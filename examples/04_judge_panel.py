"""04 — Judge panel: generate candidates, rank them through independent lenses.

The generate-then-judge pattern beats "ask once and hope":

1. **Generate** N candidates in parallel — each generator is a hermetic
   session seeded with a different stance, so you get genuine diversity
   instead of four paraphrases of the same idea.
2. **Judge** with ``judge_panel(candidates, lenses)``: one judge agent per
   *lens* (e.g. clarity, honesty, memorability) scores EVERY candidate 0–10
   through that lens alone. A candidate's final score is its mean across the
   lenses whose judge responded; a crashed judge simply drops out of the
   average. The panel returns ``RankedCandidate`` objects, best first, with
   the per-lens breakdown in ``by_lens``.

Per-lens judges (rather than one judge scoring everything holistically) is
deliberate: a single judge's biases dominate a holistic score, while
independent narrow lenses force the tradeoffs into the open — you can see a
candidate that wins on memorability but loses on honesty.

Run it::

    RDW_LIVE=1 rdw run examples/04_judge_panel.py --budget 12

Expected cost: 4 generators + 3 judges = 7 sessions, typically 4–8 AI
credits. Without ``RDW_LIVE=1`` this script only prints an explanation and
exits 0.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RDW_LIVE") != "1":
    print(
        "04_judge_panel: live example skipped (RDW_LIVE is not set).\n"
        "This example would run 7 sessions (4 generators + 3 lens judges,\n"
        "typically 4-8 AI credits). To run it:\n\n"
        "    RDW_LIVE=1 rdw run examples/04_judge_panel.py --budget 12\n"
    )
    sys.exit(0)

from pydantic import BaseModel, Field

from rdw import judge_panel

STANCES = [
    "ruthlessly plain and literal",
    "playful and slightly irreverent",
    "aimed at skeptical senior engineers",
    "aimed at first-time CLI users",
]

LENSES = ["clarity", "honesty", "memorability"]


class Tagline(BaseModel):
    text: str = Field(max_length=90, description="The tagline itself, under 90 chars")
    audience: str = Field(description="Who this tagline is written for")


async def workflow(wf):
    # ---- generate: 4 stance-diverse candidates in parallel -----------------
    async with wf.phase("generate"):
        candidates = await wf.parallel(
            [
                (lambda s=s: wf.agent(
                    "Write ONE tagline for `rapp-dynamic-workflows`, a "
                    "Python library that turns GitHub Copilot sessions into "
                    "deterministic multi-agent workflows (parallel fan-outs, "
                    "budgets, resume). Your stance: be " + s + ".",
                    schema=Tagline,
                    label=f"gen-{i}",
                ))
                for i, s in enumerate(STANCES, start=1)
            ]
        )

    texts = [c.text for c in candidates if c is not None]
    wf.log(f"{len(texts)}/{len(STANCES)} candidates generated")
    if not texts:
        wf.log("no candidates survived generation — nothing to judge")
        return

    # ---- judge: one independent judge per lens, mean-ranked ----------------
    async with wf.phase("judge"):
        ranked = await judge_panel(texts, LENSES, wf=wf)

    wf.log("ranking (best first):")
    for r in ranked:
        lenses = ", ".join(f"{k}={v:.1f}" for k, v in sorted(r.by_lens.items()))
        wf.log(f"  {r.score:.2f}  {r.candidate!r}  ({lenses or 'no judge scored it'})")

    winner = ranked[0]
    wf.log(f"winner: {winner.candidate!r} at {winner.score:.2f}/10")


if __name__ == "__main__":
    import asyncio

    from rdw import Workflow

    async def _main() -> None:
        wf = Workflow.open(budget=12.0)
        async with wf:
            await workflow(wf)
        print(wf.report())

    asyncio.run(_main())
