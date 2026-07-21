"""03 — Pipeline: per-item flow over files with no barrier between stages.

``wf.pipeline(items, *stages)`` is the streaming counterpart to ``parallel``:
every item flows through the stages independently — item 3 can still be in
stage 1 while item 1 is already in stage 3. There is **no barrier** between
stages, so total wall time is bounded by the slowest *item*, not the sum of
the slowest stage at each step.

Failure semantics: a stage that raises (or returns ``None``) drops that item
to ``None`` and skips its remaining stages — one poisoned file never blocks
the rest of the batch. Results keep input order, so ``zip(items, results)``
always lines up.

The demo migrates three legacy Python snippets through a 3-stage pipeline:

    analyze  -> what legacy constructs does the snippet use?
    migrate  -> rewrite it as modern Python 3.11+
    verify   -> does the rewrite preserve behavior? (drops the item if not)

Run it::

    RDW_LIVE=1 rdw run examples/03_pipeline_migration.py --budget 15

Expected cost: 3 items x 3 stages = up to 9 sessions, typically 5–10 AI
credits. Without ``RDW_LIVE=1`` this script only prints an explanation and
exits 0.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RDW_LIVE") != "1":
    print(
        "03_pipeline_migration: live example skipped (RDW_LIVE is not set).\n"
        "This example would run up to 9 sessions (3 items x 3 stages,\n"
        "typically 5-10 AI credits). To run it:\n\n"
        "    RDW_LIVE=1 rdw run examples/03_pipeline_migration.py --budget 15\n"
    )
    sys.exit(0)

from pydantic import BaseModel, Field

# Legacy snippets standing in for files on disk. In a real migration you would
# build ``items`` from ``pathlib.Path(...).glob("**/*.py")`` and have stages
# read/write the files themselves via ``tools=``/``cwd=``.
LEGACY_FILES = {
    "report.py": (
        "def render(rows):\n"
        "    out = ''\n"
        "    for i in range(len(rows)):\n"
        "        out = out + '%s: %s\\n' % (i, rows[i])\n"
        "    return out\n"
    ),
    "config.py": (
        "def load(path):\n"
        "    f = open(path)\n"
        "    data = f.read()\n"
        "    f.close()\n"
        "    d = {}\n"
        "    for line in data.split('\\n'):\n"
        "        if line != '':\n"
        "            parts = line.split('=')\n"
        "            d[parts[0]] = parts[1]\n"
        "    return d\n"
    ),
    "compat.py": (
        "def merge(a, b):\n"
        "    c = dict(a.items() + b.items())  # Py2-only: dict.items() lists\n"
        "    return c\n"
    ),
}


class Analysis(BaseModel):
    filename: str
    legacy_constructs: list[str] = Field(description="Each legacy pattern found")
    source: str = Field(description="The original source, unchanged")


class Migration(BaseModel):
    filename: str
    modern_source: str = Field(description="The rewritten, modern Python source")
    original_source: str = Field(description="The original source, unchanged")


class Verification(BaseModel):
    filename: str
    behavior_preserved: bool
    notes: str


async def workflow(wf):
    items = list(LEGACY_FILES.items())  # (filename, source) pairs

    async def analyze(item):
        name, src = item
        return await wf.agent(
            f"Analyze this legacy Python file and list every outdated or "
            f"unidiomatic construct. Echo the source back unchanged.\n\n"
            f"FILE {name}:\n{src}",
            schema=Analysis,
            label=f"analyze-{name}",
        )

    async def migrate(analysis: Analysis):
        constructs = ", ".join(analysis.legacy_constructs) or "none listed"
        return await wf.agent(
            f"Rewrite this file as clean modern Python 3.11+ (f-strings, "
            f"context managers, comprehensions, pathlib where natural). "
            f"Known legacy constructs: {constructs}. Preserve behavior "
            f"exactly. Echo the original source back unchanged too.\n\n"
            f"FILE {analysis.filename}:\n{analysis.source}",
            schema=Migration,
            label=f"migrate-{analysis.filename}",
        )

    async def verify(migration: Migration):
        result = await wf.agent(
            "Compare the original and migrated source. Decide whether the "
            "migration preserves behavior for all inputs the original "
            "handled.\n\n"
            f"FILE {migration.filename}\nORIGINAL:\n{migration.original_source}\n"
            f"MIGRATED:\n{migration.modern_source}",
            schema=Verification,
            label=f"verify-{migration.filename}",
        )
        # Returning None drops the item — a rejected migration falls out of
        # the pipeline instead of being reported as a success.
        if not result.behavior_preserved:
            wf.log(f"{migration.filename}: verification REJECTED — {result.notes}")
            return None
        return migration

    async with wf.phase("migration-pipeline"):
        results = await wf.pipeline(items, analyze, migrate, verify)

    for (name, _), migrated in zip(items, results):
        if migrated is None:
            wf.log(f"{name}: DROPPED (a stage failed or verification rejected it)")
        else:
            wf.log(f"{name}: migrated OK ({len(migrated.modern_source)} chars)")


if __name__ == "__main__":
    import asyncio

    from rdw import Workflow

    async def _main() -> None:
        wf = Workflow.open(budget=15.0)
        async with wf:
            await workflow(wf)
        print(wf.report())

    asyncio.run(_main())
