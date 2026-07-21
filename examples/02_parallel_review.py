"""02 — Parallel review + adversarial verification (the canonical pattern).

This is the pattern rdw was built around, distilled from real hand-rolled
multi-agent orchestration logs:

1. **Fan out** one reviewer per *dimension* (correctness, security,
   performance, readability) with ``wf.parallel``. Each reviewer is a hermetic
   session that sees only its own prompt and returns schema-forced findings.
   A reviewer that crashes or times out resolves to ``None`` — the wave
   degrades, it never explodes.
2. **Adversarially verify** each finding with ``adversarial_verify``: three
   independent skeptics are told to *destroy* the claim from different angles
   (counterexamples, hidden assumptions, real-world failure modes) and only
   concede if their attack fails. A finding survives on strict majority.

Step 2 is what separates this from naive fan-out: single-agent findings have
a high false-positive rate; findings that survive a hostile panel are worth a
human's time.

Run it::

    RDW_LIVE=1 rdw run examples/02_parallel_review.py --budget 20

Expected cost: 4 reviewers + 3 skeptics per surviving finding — typically
8–20 AI credits depending on how many findings the reviewers raise. The
``--budget`` ceiling hard-caps the run regardless.
Without ``RDW_LIVE=1`` this script only prints an explanation and exits 0.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RDW_LIVE") != "1":
    print(
        "02_parallel_review: live example skipped (RDW_LIVE is not set).\n"
        "This example would run 4 parallel reviewer sessions plus 3 skeptic\n"
        "sessions per finding (typically 8-20 AI credits). To run it:\n\n"
        "    RDW_LIVE=1 rdw run examples/02_parallel_review.py --budget 20\n"
    )
    sys.exit(0)

from pydantic import BaseModel, Field

from rdw import adversarial_verify

# The code under review — inlined so the example is hermetic. In a real
# workflow you would read files from disk and pass ``cwd=`` to the agents.
CODE_UNDER_REVIEW = '''
import sqlite3

def get_user(db_path, username):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name = '%s'" % username)
    row = cur.fetchone()
    return {"name": row[0], "email": row[1]}

def average(scores):
    total = 0
    for s in scores:
        total += s
    return total / len(scores)
'''

DIMENSIONS = ["correctness", "security", "performance", "readability"]


class Finding(BaseModel):
    summary: str = Field(description="One-sentence statement of the defect")
    severity: str = Field(description="One of: low, medium, high")
    evidence: str = Field(description="The exact code fragment that demonstrates it")


class Review(BaseModel):
    dimension: str = Field(description="The dimension you were asked to review")
    findings: list[Finding] = Field(description="Real defects only; empty list if clean")


async def workflow(wf):
    # ---- wave 1: one hermetic reviewer per dimension -----------------------
    async with wf.phase("review"):
        reviews = await wf.parallel(
            [
                (lambda d=d: wf.agent(
                    f"You are a strict {d} reviewer. Review ONLY for {d} "
                    "problems — ignore everything else. Report real defects "
                    f"with exact evidence.\n\nCODE:\n{CODE_UNDER_REVIEW}",
                    schema=Review,
                    label=f"review-{d}",
                ))
                for d in DIMENSIONS
            ]
        )

    findings = [f for r in reviews if r is not None for f in r.findings]
    wf.log(f"{len(findings)} raw finding(s) from {sum(r is not None for r in reviews)} reviewer(s)")

    # ---- wave 2: adversarial verification of every finding -----------------
    verified = []
    async with wf.phase("verify"):
        for finding in findings:
            claim = f"This code has a {finding.severity}-severity defect: {finding.summary}"
            result = await adversarial_verify(claim, n=3, evidence=finding.evidence, wf=wf)
            wf.log(
                f"{'SURVIVED' if result.passed else 'rejected'} "
                f"({result.upheld}-{result.rejected}): {finding.summary}"
            )
            if result.passed:
                verified.append(finding)

    wf.log(f"final: {len(verified)}/{len(findings)} finding(s) survived the skeptic panel")
    for f in verified:
        wf.log(f"  [{f.severity}] {f.summary}")


if __name__ == "__main__":
    import asyncio

    from rdw import Workflow

    async def _main() -> None:
        wf = Workflow.open(budget=20.0)
        async with wf:
            await workflow(wf)
        print(wf.report())

    asyncio.run(_main())
