"""05 — Budget loop: iterate until the work is done OR the money runs out.

rdw budgets are a *hard admission gate*, which makes open-ended loops safe:

* ``wf.budget`` tracks real spend from the SDK's usage events
  (``spent()`` / ``remaining()`` / ``total``), in AI credits (AIU).
* Before every live agent starts, the engine checks the ceiling; at or past
  it, ``agent()`` raises ``BudgetExceeded`` instead of starting a session.
* Each session is additionally created with a runtime-side
  ``max_ai_credits`` cap (defense in depth), so even a runaway in-flight
  agent can't blow far past the ceiling.
* Inside ``wf.parallel``, a branch refused by the gate resolves to ``None``
  like any other failure — a wave *degrades* when money runs low; it never
  crashes the run.

The demo runs an improve-until-good-or-broke loop: draft an explanation, then
repeatedly (critic -> rewrite) until the critic scores it >= 9/10, the loop
hits its round cap, or ``BudgetExceeded`` ends it gracefully. The loop also
checks ``remaining()`` and stops *before* starting a round it likely can't
finish — politer than slamming into the gate mid-round.

Run it (the small ceiling is the point — try lowering it further)::

    RDW_LIVE=1 rdw run examples/05_budget_loop.py --budget 6

Expected cost: exactly what you give it — the loop spends until the ceiling
or convergence, whichever comes first (with --budget 6, at most 6 AI
credits). Without ``RDW_LIVE=1`` this script only prints an explanation and
exits 0.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RDW_LIVE") != "1":
    print(
        "05_budget_loop: live example skipped (RDW_LIVE is not set).\n"
        "This example loops critic->rewrite sessions until quality converges\n"
        "or the --budget ceiling stops it (cost == whatever ceiling you set).\n"
        "To run it:\n\n"
        "    RDW_LIVE=1 rdw run examples/05_budget_loop.py --budget 6\n"
    )
    sys.exit(0)

from pydantic import BaseModel, Field

from rdw import BudgetExceeded

MAX_ROUNDS = 6
TARGET_SCORE = 9.0
# Don't start a critic+rewrite round with less than this left in the tank.
MIN_CREDITS_PER_ROUND = 1.0

TASK = (
    "Explain, in one paragraph a busy engineer will actually read, why "
    "append-only journals make workflow re-runs resumable."
)


class Draft(BaseModel):
    text: str = Field(description="The explanation paragraph")


class Critique(BaseModel):
    score: float = Field(ge=0.0, le=10.0, description="Overall quality, 0-10")
    worst_problem: str = Field(description="The single biggest problem to fix next")


async def workflow(wf):
    async with wf.phase("draft"):
        draft = await wf.agent(TASK, schema=Draft, label="draft-0")

    best = draft.text
    try:
        for round_no in range(1, MAX_ROUNDS + 1):
            remaining = wf.budget.remaining()
            if remaining is not None and remaining < MIN_CREDITS_PER_ROUND:
                wf.log(
                    f"stopping before round {round_no}: only {remaining:.2f} "
                    f"AIU left (spent {wf.budget.spent():.2f})"
                )
                break

            async with wf.phase(f"round-{round_no}"):
                critique = await wf.agent(
                    f"Critique this explanation harshly. TASK: {TASK}\n\n"
                    f"DRAFT:\n{best}",
                    schema=Critique,
                    label=f"critic-{round_no}",
                )
                wf.log(
                    f"round {round_no}: score {critique.score:.1f}/10 — "
                    f"{critique.worst_problem}"
                )
                if critique.score >= TARGET_SCORE:
                    wf.log(f"converged at {critique.score:.1f}/10")
                    break

                rewrite = await wf.agent(
                    f"Rewrite the draft to fix exactly this problem: "
                    f"{critique.worst_problem}\nKeep it one paragraph. "
                    f"TASK: {TASK}\n\nDRAFT:\n{best}",
                    schema=Draft,
                    label=f"rewrite-{round_no}",
                )
                best = rewrite.text
    except BudgetExceeded as exc:
        # The admission gate ended the loop for us — the run stays healthy.
        wf.log(f"budget gate closed the loop: {exc.spent:.2f}/{exc.total:.2f} AIU")

    wf.log(f"final spend: {wf.budget.summary()}")
    wf.log(f"final draft:\n{best}")


if __name__ == "__main__":
    import asyncio

    from rdw import Workflow

    async def _main() -> None:
        wf = Workflow.open(budget=6.0)
        async with wf:
            await workflow(wf)
        print(wf.report())

    asyncio.run(_main())
