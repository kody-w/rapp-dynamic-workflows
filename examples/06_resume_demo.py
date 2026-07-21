"""06 — Resume: kill a run halfway, resume it, and pay only for the new work.

Every ``wf.agent(...)`` call is journaled to
``.rdw/runs/<run-id>/journal.jsonl`` with a fingerprint of
``(prompt, normalized options)`` plus an occurrence number (the Nth call
with that fingerprint). On ``--resume``, a call whose fingerprint and
occurrence match a recorded result is *replayed from the journal* —
instantly, at zero cost, re-validated against its schema. The keying is
content-addressed rather than call-order-positional, so parallel/pipeline
scheduling differences between runs never bust the cache. The first call
that does NOT match marks a divergence: a loud ``DivergenceWarning`` is
emitted, a divergence marker is appended to the journal, and everything
from that point on runs live.

=========================================================================
WALKTHROUGH — try the kill/resume cycle yourself
=========================================================================

1. Start the run and note the run id it prints (also visible in `rdw runs`):

       RDW_LIVE=1 rdw run examples/06_resume_demo.py --budget 10

2. KILL IT mid-flight — press Ctrl-C while phase "expand" is running.
   rdw prints the exact resume command on interrupt, e.g.:

       interrupted — resume with: rdw run examples/06_resume_demo.py --resume 20260721-...

3. Resume it:

       RDW_LIVE=1 rdw run examples/06_resume_demo.py --resume <RUN_ID> --budget 10

   Watch the progress tree: the agents that finished before the kill flash
   by as "cached" (0 credits, no session created), and execution goes live
   exactly where the first run died. `rdw show <RUN_ID>` afterwards shows
   the whole stitched history.

4. (Optional) See divergence handling: after a completed run, edit
   OUTLINE_TOPIC below and resume again. The first agent's fingerprint no
   longer matches, so rdw warns, marks the divergence in the journal, and
   re-runs everything live from that point — cached results are never
   silently reused for a prompt that changed.

Why this matters: long multi-agent runs die for boring reasons (laptop lid,
network, a bug in stage 7 of 9). With fingerprinted replay, "fix and re-run"
costs only the un-run tail, not the whole workflow again.

Expected cost: ~3–6 AI credits for a full cold run; a resume after a late
kill costs only the remaining agents. Without ``RDW_LIVE=1`` this script
only prints an explanation and exits 0.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RDW_LIVE") != "1":
    print(
        "06_resume_demo: live example skipped (RDW_LIVE is not set).\n"
        "This example demonstrates kill/resume: run it, Ctrl-C it mid-run,\n"
        "then resume — finished agents replay from the journal for free\n"
        "(~3-6 AI credits for a full cold run). To run it:\n\n"
        "    RDW_LIVE=1 rdw run examples/06_resume_demo.py --budget 10\n"
        "    # Ctrl-C while it runs, then:\n"
        "    RDW_LIVE=1 rdw run examples/06_resume_demo.py --resume <RUN_ID> --budget 10\n"
    )
    sys.exit(0)

from pydantic import BaseModel, Field

# Edit this AFTER a completed run and resume to watch divergence handling
# (step 4 in the walkthrough above).
OUTLINE_TOPIC = "how append-only journals give multi-agent workflows resume"


class Outline(BaseModel):
    title: str
    sections: list[str] = Field(min_length=3, max_length=5, description="Section headings")


class Section(BaseModel):
    heading: str
    paragraph: str = Field(description="One tight paragraph for this section")


async def workflow(wf):
    # Phase 1: one agent. After any kill later than this, resuming replays
    # this outline from the journal — same position, same fingerprint.
    async with wf.phase("outline"):
        outline = await wf.agent(
            f"Write a short article outline (3-5 section headings) on: "
            f"{OUTLINE_TOPIC}",
            schema=Outline,
            label="outliner",
        )
    wf.log(f"outline: {outline.title!r} with {len(outline.sections)} sections")

    # Phase 2: one agent per section, sequential ON PURPOSE so a Ctrl-C lands
    # between agents and the kill point is easy to see in `rdw show`.
    # (In production you'd use wf.parallel here — replay works there too;
    # cache keys are content-addressed, so completion order never matters.)
    drafts: list[Section] = []
    async with wf.phase("expand"):
        for i, heading in enumerate(outline.sections, start=1):
            section = await wf.agent(
                f"Write one tight paragraph for the section {heading!r} of an "
                f"article titled {outline.title!r}. Topic: {OUTLINE_TOPIC}",
                schema=Section,
                label=f"section-{i}",
            )
            drafts.append(section)
            wf.log(f"section {i}/{len(outline.sections)} done: {heading}")

    async with wf.phase("assemble"):
        article = "\n\n".join(f"## {s.heading}\n{s.paragraph}" for s in drafts)
        wf.log(f"assembled {len(article)} chars, {len(drafts)} sections")
        wf.log(f"# {outline.title}\n\n{article}")


if __name__ == "__main__":
    import asyncio

    from rdw import Workflow

    # Standalone resume: python examples/06_resume_demo.py [RUN_ID]
    async def _main() -> None:
        run_id = sys.argv[1] if len(sys.argv) > 1 else None
        wf = Workflow.open(run_id=run_id, resume=run_id is not None, budget=10.0)
        print(f"run id: {wf.run_id}  (resume with: python {sys.argv[0]} {wf.run_id})")
        async with wf:
            await workflow(wf)
        print(wf.report())

    asyncio.run(_main())
