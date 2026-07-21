"""Quality patterns: adversarial_verify, judge_panel, loop_until_dry."""

from __future__ import annotations

import pytest

from rdw.patterns import (
    RankedCandidate,
    SkepticVote,
    VerifyResult,
    adversarial_verify,
    judge_panel,
    loop_until_dry,
)

from conftest import FakeRuntime, Turn


def vote(holds: bool, why: str = "because") -> dict:
    return {"claim_holds": holds, "reasoning": why, "confidence": 0.9}


def scorecard(*pairs: tuple[int, float]) -> dict:
    return {
        "scores": [
            {"index": i, "score": s, "rationale": f"cand {i}"} for i, s in pairs
        ]
    }


# ------------------------------------------------------- adversarial_verify


@pytest.mark.asyncio
async def test_adversarial_verify_strict_majority_passes(make_wf):
    rt = FakeRuntime(
        [
            [Turn(submit=[vote(True)])],
            [Turn(submit=[vote(True)])],
            [Turn(submit=[vote(False, "found a counterexample")])],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        result = await adversarial_verify("the sky is blue", n=3, wf=wf)
    assert isinstance(result, VerifyResult)
    assert result.passed and bool(result)
    assert (result.upheld, result.rejected) == (2, 1)
    assert len(result.votes) == 3 and all(isinstance(v, SkepticVote) for v in result.votes)
    # three independent skeptic sessions, each schema-forced
    assert len(rt.created) == 3
    assert all(kw["available_tools"] == ["submit_result"] for kw in rt.create_kwargs)


@pytest.mark.asyncio
async def test_adversarial_verify_minority_fails(make_wf):
    rt = FakeRuntime(
        [
            [Turn(submit=[vote(True)])],
            [Turn(submit=[vote(False)])],
            [Turn(submit=[vote(False)])],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        result = await adversarial_verify("dubious claim", n=3, wf=wf)
    assert not result.passed
    assert (result.upheld, result.rejected) == (1, 2)


@pytest.mark.asyncio
async def test_adversarial_verify_tie_among_respondents_fails(make_wf):
    """One skeptic dies; 1-1 among respondents is not a strict majority."""
    rt = FakeRuntime(
        [
            [Turn(submit=[vote(True)])],
            [Turn(error=RuntimeError("skeptic crashed"))],
            [Turn(submit=[vote(False)])],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        result = await adversarial_verify("claim", n=3, wf=wf)
    assert not result.passed
    assert len(result.votes) == 2


@pytest.mark.asyncio
async def test_adversarial_verify_all_skeptics_failing_fails_closed(make_wf):
    rt = FakeRuntime(
        [
            [Turn(error=RuntimeError("x"))],
            [Turn(error=RuntimeError("y"))],
            [Turn(error=RuntimeError("z"))],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        result = await adversarial_verify("claim", n=3, wf=wf)
    assert not result.passed
    assert result.votes == []


# --------------------------------------------------------------- judge_panel


@pytest.mark.asyncio
async def test_judge_panel_ranks_by_mean_across_lenses(make_wf):
    rt = FakeRuntime(
        [
            [Turn(submit=[scorecard((0, 2.0), (1, 8.0))])],  # lens: speed
            [Turn(submit=[scorecard((0, 4.0), (1, 6.0))])],  # lens: risk
        ]
    )
    async with make_wf(runtime=rt) as wf:
        ranked = await judge_panel(["cand-A", "cand-B"], ["speed", "risk"], wf=wf)
    assert [type(r) for r in ranked] == [RankedCandidate, RankedCandidate]
    assert ranked[0].candidate == "cand-B" and ranked[0].score == pytest.approx(7.0)
    assert ranked[1].candidate == "cand-A" and ranked[1].score == pytest.approx(3.0)
    assert set(ranked[0].by_lens) == {"speed", "risk"}
    assert len(rt.created) == 2  # one judge per lens


@pytest.mark.asyncio
async def test_judge_panel_failed_judge_drops_out_of_average(make_wf):
    rt = FakeRuntime(
        [
            [Turn(submit=[scorecard((0, 9.0), (1, 1.0))])],
            [Turn(error=RuntimeError("judge asleep"))],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        ranked = await judge_panel(["A", "B"], ["quality", "cost"], wf=wf)
    assert ranked[0].candidate == "A" and ranked[0].score == pytest.approx(9.0)
    assert len(ranked[0].by_lens) == 1  # only the surviving lens contributes


@pytest.mark.asyncio
async def test_judge_panel_unscored_candidate_sorts_last(make_wf):
    rt = FakeRuntime([[Turn(submit=[scorecard((0, 5.0))])]])  # never scores cand 1
    async with make_wf(runtime=rt) as wf:
        ranked = await judge_panel(["A", "B"], ["only-lens"], wf=wf)
    assert ranked[-1].candidate == "B" and ranked[-1].score == 0.0


# ------------------------------------------------------------ loop_until_dry


@pytest.mark.asyncio
async def test_loop_until_dry_counts_consecutive_dry_rounds():
    rounds = [["a", "b"], ["b"], [], []]
    calls: list[int] = []

    async def finder(round_no: int):
        calls.append(round_no)
        return rounds[round_no] if round_no < len(rounds) else []

    findings = await loop_until_dry(finder, key=lambda f: f, dry_rounds=2)
    assert findings == ["a", "b"]  # deduped, first-seen order
    # round 1 found nothing NEW (only the duplicate "b") -> dry streak starts
    # there; round 2 is the second consecutive dry round -> stop.
    assert calls == [0, 1, 2]


@pytest.mark.asyncio
async def test_loop_until_dry_new_finding_resets_streak():
    rounds = [["a"], [], ["c"], [], []]

    async def finder(round_no: int):
        return rounds[round_no]

    findings = await loop_until_dry(finder, key=lambda f: f, dry_rounds=2)
    assert findings == ["a", "c"]  # round 2's new finding reset the dry streak


@pytest.mark.asyncio
async def test_loop_until_dry_max_rounds_cap():
    calls = []

    async def gusher(round_no: int):
        calls.append(round_no)
        return [f"finding-{round_no}"]  # always something new

    findings = await loop_until_dry(gusher, key=lambda f: f, dry_rounds=2, max_rounds=4)
    assert len(calls) == 4
    assert findings == [f"finding-{i}" for i in range(4)]


@pytest.mark.asyncio
async def test_loop_until_dry_logs_inside_workflow(make_wf):
    async def finder(round_no: int):
        return []

    async with make_wf() as wf:
        assert await loop_until_dry(finder, key=lambda f: f, dry_rounds=1, wf=wf) == []
    # the round summary was journaled as a log note
    text = wf.journal.path.read_text()
    assert "loop_until_dry round 1" in text
