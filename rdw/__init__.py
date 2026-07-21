"""rapp-dynamic-workflows — deterministic multi-agent orchestration for the
GitHub Copilot SDK.

Inspired by Claude Code's Workflow tool, built for GitHub Copilot CLI users:
hermetic per-call agent sessions, schema-forced structured output (the
submit-tool pattern — the SDK has no native ``response_format``), barriered
``parallel`` fan-outs, per-item ``pipeline`` flows, AI-credit budgets with a
hard ceiling, and a fingerprinted append-only journal that makes re-runs
resume instead of respawn.

Quickstart::

    # wave.py
    from pydantic import BaseModel

    class Idea(BaseModel):
        approach: str
        confidence: float

    async def workflow(wf):
        async with wf.phase("design"):
            ideas = await wf.parallel([
                (lambda i=i: wf.agent(f"You are strategist {i} of 4. Propose a design.",
                                      schema=Idea, label=f"strategy-{i}"))
                for i in range(1, 5)
            ])
        wf.log(f"{sum(x is not None for x in ideas)} ideas survived")

    # $ rdw run wave.py --budget 50
"""

from __future__ import annotations

from .budget import Budget
from .engine import (
    Workflow,
    agent,
    current_workflow,
    log,
    new_run_id,
    parallel,
    phase,
    pipeline,
)
from .errors import (
    AgentError,
    AgentSchemaError,
    AgentTimeout,
    BudgetExceeded,
    RdwError,
    DivergenceWarning,
    JournalError,
    JournalWarning,
    WorkflowContextError,
)
from .journal import AgentRecord, Journal, fingerprint
from .patterns import (
    RankedCandidate,
    SkepticVote,
    VerifyResult,
    adversarial_verify,
    judge_panel,
    loop_until_dry,
)
from .progress import Progress
from .runtime import BaseRuntime, CopilotRuntime, Runtime, SessionHandle
from .schema import SUBMIT_TOOL_NAME, SchemaSpec, SubmitCapture, build_submit_tool

__version__ = "0.1.0"

__all__ = [
    # engine
    "Workflow",
    "agent",
    "parallel",
    "pipeline",
    "phase",
    "log",
    "current_workflow",
    "new_run_id",
    # budget
    "Budget",
    # journal
    "Journal",
    "AgentRecord",
    "fingerprint",
    # runtime
    "Runtime",
    "BaseRuntime",
    "CopilotRuntime",
    "SessionHandle",
    # schema
    "SchemaSpec",
    "SubmitCapture",
    "build_submit_tool",
    "SUBMIT_TOOL_NAME",
    # progress
    "Progress",
    # patterns
    "adversarial_verify",
    "judge_panel",
    "loop_until_dry",
    "VerifyResult",
    "SkepticVote",
    "RankedCandidate",
    # errors
    "RdwError",
    "AgentError",
    "AgentTimeout",
    "AgentSchemaError",
    "BudgetExceeded",
    "JournalError",
    "JournalWarning",
    "DivergenceWarning",
    "WorkflowContextError",
    "__version__",
]
