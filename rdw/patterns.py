"""Quality patterns built purely on the public Workflow API.

These are thin combinators over ``agent`` + ``parallel`` — nothing here
touches the SDK directly, so everything the engine guarantees (budgets,
journaling, resume, None-on-failure) applies automatically. They formalize
the highest-value patterns observed in real hand-rolled orchestration
logs: adversarial red-team panels, independent judge panels,
review-until-clean loops, and spend-the-budget improvement loops.

Each helper takes an optional ``wf`` argument; when omitted, the Workflow
bound to the current async context is used (i.e. they Just Work inside a
``rdw run`` script).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Hashable, Sequence

from pydantic import BaseModel, Field

from .engine import Workflow, current_workflow
from .errors import AgentLimitExceeded

# --------------------------------------------------------------------------
# adversarial_verify
# --------------------------------------------------------------------------


class SkepticVote(BaseModel):
    """One skeptic's schema-forced verdict on a claim."""

    claim_holds: bool = Field(description="True if the claim survived your attack")
    reasoning: str = Field(description="The strongest argument you found, and why it did or did not defeat the claim")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5, description="Confidence in your verdict")


@dataclass
class VerifyResult:
    """Outcome of :func:`adversarial_verify`.

    ``passed`` is a strict majority of the skeptics that actually responded;
    if every skeptic failed (all ``None``), ``passed`` is False.
    """

    passed: bool
    upheld: int
    rejected: int
    votes: list[SkepticVote]

    def __bool__(self) -> bool:
        return self.passed


async def adversarial_verify(
    claim: str,
    n: int = 3,
    *,
    evidence: str | None = None,
    wf: Workflow | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> VerifyResult:
    """Fan out ``n`` independent skeptics told to *destroy* the claim.

    Each skeptic is explicitly instructed to find reasons the claim is FALSE
    and only concede if its attack fails — perspective-diverse phrasing keeps
    the panel from collapsing into one argument. The claim passes on a strict
    majority of respondents.

    Args:
        claim: The statement to verify.
        n: Number of skeptics (default 3).
        evidence: Optional supporting material shown to every skeptic.
        wf: Explicit workflow (ambient one otherwise).
        model / effort: Per-skeptic overrides.
    """
    wf = wf or current_workflow()
    angles = [
        "hunt for counterexamples and edge cases",
        "attack the hidden assumptions it rests on",
        "stress-test it against real-world failure modes",
        "check it for internal contradictions and vague terms",
        "look for evidence that directly contradicts it",
    ]
    body = f"CLAIM:\n{claim}\n"
    if evidence:
        body += f"\nEVIDENCE PROVIDED:\n{evidence}\n"

    def make_prompt(i: int) -> str:
        angle = angles[i % len(angles)]
        return (
            f"You are skeptic {i + 1} of {n}. Your job is to prove the claim "
            f"below is FALSE — {angle}. Argue against it as hard as you can, "
            "then render an honest verdict: claim_holds=true ONLY if your "
            f"attack failed.\n\n{body}"
        )

    votes = await wf.parallel(
        [
            (lambda i=i: wf.agent(
                make_prompt(i),
                schema=SkepticVote,
                label=f"skeptic-{i + 1}",
                model=model,
                effort=effort,
            ))
            for i in range(n)
        ]
    )
    valid = [v for v in votes if isinstance(v, SkepticVote)]
    upheld = sum(1 for v in valid if v.claim_holds)
    rejected = len(valid) - upheld
    passed = bool(valid) and upheld > rejected
    return VerifyResult(passed=passed, upheld=upheld, rejected=rejected, votes=valid)


# --------------------------------------------------------------------------
# judge_panel
# --------------------------------------------------------------------------


class _CandidateScore(BaseModel):
    index: int = Field(ge=0, description="Zero-based index of the candidate being scored")
    score: float = Field(ge=0.0, le=10.0, description="Score from 0 (worst) to 10 (best)")
    rationale: str = Field(description="One or two sentences justifying the score")


class _Scorecard(BaseModel):
    scores: list[_CandidateScore] = Field(description="One entry per candidate, in any order")


@dataclass
class RankedCandidate:
    """One entry of a :func:`judge_panel` ranking (best first)."""

    index: int
    candidate: Any
    score: float
    by_lens: dict[str, float]


async def judge_panel(
    candidates: Sequence[Any],
    lenses: Sequence[str],
    *,
    wf: Workflow | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> list[RankedCandidate]:
    """Score ``candidates`` through independent judging ``lenses`` and rank.

    One judge agent per lens scores *every* candidate 0-10 through that lens
    alone (e.g. ``"robustness"``, ``"implementation risk"``); a candidate's
    final score is its mean across the lenses whose judge responded. Judges
    that fail resolve to ``None`` and simply drop out of the average.

    Returns:
        Candidates ranked best-first as :class:`RankedCandidate`. Candidates
        no judge managed to score sort last with score 0.
    """
    wf = wf or current_workflow()
    listing = "\n\n".join(
        f"### Candidate {i}\n{c if isinstance(c, str) else repr(c)}"
        for i, c in enumerate(candidates)
    )

    def make_prompt(lens: str) -> str:
        return (
            f"You are one judge on an independent panel. Score EVERY candidate "
            f"below from 0 to 10 through exactly one lens: **{lens}**. Ignore "
            "all other qualities. Be discriminating — use the full range.\n\n"
            f"{listing}"
        )

    cards = await wf.parallel(
        [
            (lambda lens=lens: wf.agent(
                make_prompt(lens),
                schema=_Scorecard,
                label=f"judge-{lens}",
                model=model,
                effort=effort,
            ))
            for lens in lenses
        ]
    )

    by_lens: dict[int, dict[str, float]] = {i: {} for i in range(len(candidates))}
    for lens, card in zip(lenses, cards):
        if not isinstance(card, _Scorecard):
            continue
        for entry in card.scores:
            if 0 <= entry.index < len(candidates):
                by_lens[entry.index][str(lens)] = entry.score

    ranked = [
        RankedCandidate(
            index=i,
            candidate=candidates[i],
            score=(sum(scores.values()) / len(scores)) if scores else 0.0,
            by_lens=scores,
        )
        for i, scores in by_lens.items()
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


# --------------------------------------------------------------------------
# loop_until_dry
# --------------------------------------------------------------------------


async def loop_until_dry(
    finder: Callable[[int], Awaitable[Sequence[Any] | None]],
    key: Callable[[Any], Hashable],
    dry_rounds: int = 2,
    *,
    max_rounds: int = 10,
    wf: Workflow | None = None,
) -> list[Any]:
    """Repeat ``finder`` until it stops producing *new* findings.

    The review-until-clean loop: ``finder(round_number)`` returns an iterable
    of findings (typically it runs a reviewer agent, and may also fix what the
    previous round found); ``key(finding)`` is a finding's stable identity
    used for deduplication. The loop ends after ``dry_rounds`` consecutive
    rounds surface nothing new, or after ``max_rounds`` total.

    Args:
        finder: Async callable receiving the zero-based round number.
        key: Maps a finding to a hashable identity (e.g.
            ``lambda f: (f.file, f.line, f.summary)``).
        dry_rounds: Consecutive no-new-findings rounds required to stop
            (default 2 — one clean pass can be luck; two is a signal).
        max_rounds: Hard cap on total rounds.
        wf: Workflow used only for logging (ambient one otherwise; logging is
            skipped when no workflow is active).

    Returns:
        Every unique finding observed, in first-seen order.
    """
    try:
        wf = wf or current_workflow()
    except Exception:
        wf = None
    seen: dict[Hashable, Any] = {}
    dry = 0
    for round_no in range(max_rounds):
        findings = await finder(round_no) or []
        new = [f for f in findings if key(f) not in seen]
        for f in new:
            seen[key(f)] = f
        if wf is not None:
            wf.log(
                f"loop_until_dry round {round_no + 1}: "
                f"{len(new)} new / {len(seen)} total finding(s)"
            )
        dry = dry + 1 if not new else 0
        if dry >= dry_rounds:
            break
    return list(seen.values())


# --------------------------------------------------------------------------
# loop_until_budget
# --------------------------------------------------------------------------


async def loop_until_budget(
    step: Callable[[int], Awaitable[Any]],
    *,
    floor: float = 0.0,
    max_rounds: int | None = None,
    wf: Workflow | None = None,
) -> list[Any]:
    """Repeat ``step`` until the run budget is spent down to ``floor``.

    The keep-improving-while-credits-last loop: ``step(round_number)`` does one
    round of work (typically an agent or a small wave) and its result is
    collected; before each round the loop checks
    ``wf.budget.remaining()`` and stops once it is at or below ``floor``.
    A failed round is absorbed to ``None`` like a ``parallel`` branch — except
    :class:`~rdw.errors.AgentLimitExceeded`, which propagates (the run-level
    cap firing inside a budget loop is exactly the runaway this pattern could
    otherwise mask).

    Args:
        step: Async callable receiving the zero-based round number.
        floor: Stop once remaining credits are at or below this (default 0 —
            run until the ceiling; leave headroom for a final synthesis agent
            by setting it higher).
        max_rounds: Hard cap on rounds. **Required when the budget is
            unlimited** (``remaining()`` is ``None``) — otherwise the loop
            would have no stopping condition at all.
        wf: Explicit workflow (ambient one otherwise).

    Returns:
        Every round's result in order (``None`` for absorbed failures).
    """
    wf = wf or current_workflow()
    results: list[Any] = []
    round_no = 0
    while max_rounds is None or round_no < max_rounds:
        remaining = wf.budget.remaining()
        if remaining is None and max_rounds is None:
            raise ValueError(
                "loop_until_budget with an unlimited budget requires max_rounds "
                "(there is no ceiling to loop toward)"
            )
        if remaining is not None and remaining <= floor:
            wf.log(
                f"loop_until_budget stopped at floor: {remaining:.2f} AIU "
                f"remaining <= floor {floor:.2f} after {round_no} round(s)"
            )
            break
        try:
            results.append(await step(round_no))
        except AgentLimitExceeded:
            raise  # run-level misconfiguration — same taxonomy as parallel()
        except Exception as exc:
            wf.log(f"loop_until_budget round {round_no + 1} failed: {exc}")
            results.append(None)
        round_no += 1
    return results
