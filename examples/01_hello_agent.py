"""01 — Hello, agent: one schema-forced agent call.

The smallest possible rdw workflow: a single ``wf.agent(...)`` call whose
``schema=`` argument is a Pydantic model. The GitHub Copilot SDK has no native
``response_format``, so rdw forces structure with the *submit-tool pattern*:

* the session gets exactly one tool, ``submit_result``, whose JSON schema is
  compiled from your Pydantic model;
* an appended system message instructs the model to finish by calling it;
* invalid arguments are bounced back by the SDK so the model retries, and a
  short "nudge ladder" re-prompts a model that ends its turn without calling
  the tool.

What you get back from ``wf.agent`` is the *validated Pydantic instance* — not
text you have to parse. If the model never submits, ``AgentSchemaError`` is
raised instead of returning garbage.

Run it::

    RDW_LIVE=1 rdw run examples/01_hello_agent.py --budget 2

    # or standalone:
    RDW_LIVE=1 python examples/01_hello_agent.py

Expected cost: roughly 0.5–1 AI credit (one small session, one turn).
Without ``RDW_LIVE=1`` this script only prints an explanation and exits 0 —
no session is created and nothing is spent.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RDW_LIVE") != "1":
    print(
        "01_hello_agent: live example skipped (RDW_LIVE is not set).\n"
        "This example would start ONE real GitHub Copilot session and spend\n"
        "roughly 0.5-1 AI credit. To run it for real:\n\n"
        "    RDW_LIVE=1 rdw run examples/01_hello_agent.py --budget 2\n"
    )
    sys.exit(0)

from pydantic import BaseModel, Field


class Haiku(BaseModel):
    """The structured result the agent MUST submit."""

    lines: list[str] = Field(min_length=3, max_length=3, description="Exactly three lines")
    syllables: list[int] = Field(description="Syllable count of each line, ideally 5/7/5")
    theme: str = Field(description="One or two words naming the theme")


async def workflow(wf):
    haiku = await wf.agent(
        "Write a haiku about append-only journals. Count the syllables of "
        "each line yourself and report them honestly.",
        schema=Haiku,
        label="haiku-poet",
    )
    wf.log(f"theme: {haiku.theme}, syllables: {haiku.syllables}")
    for line in haiku.lines:
        wf.log(f"  {line}")


if __name__ == "__main__":
    import asyncio

    from rdw import Workflow

    async def _main() -> None:
        wf = Workflow.open(budget=2.0)
        async with wf:
            await workflow(wf)
        print(wf.report())

    asyncio.run(_main())
