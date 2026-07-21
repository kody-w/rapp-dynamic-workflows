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
    MAX_AGENTS_PER_RUN,
    MAX_WAVE_ITEMS,
    Workflow,
    agent,
    current_workflow,
    log,
    new_run_id,
    now,
    parallel,
    phase,
    pipeline,
    random,
    uuid,
)
from .errors import (
    AgentError,
    AgentLimitExceeded,
    AgentSchemaError,
    AgentTimeout,
    BudgetExceeded,
    RdwError,
    RdwWarning,
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
    loop_until_budget,
    loop_until_dry,
)
from .progress import Progress
from .runtime import BaseRuntime, CopilotRuntime, Runtime, SessionHandle
from .schema import SUBMIT_TOOL_NAME, SchemaSpec, SubmitCapture, build_submit_tool
from .transcripts import TranscriptWriter, UsageTap

__version__ = "0.2.0"

__all__ = [
    # engine
    "Workflow",
    "agent",
    "parallel",
    "pipeline",
    "phase",
    "log",
    "now",
    "random",
    "uuid",
    "current_workflow",
    "new_run_id",
    "MAX_AGENTS_PER_RUN",
    "MAX_WAVE_ITEMS",
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
    # transcripts / telemetry
    "TranscriptWriter",
    "UsageTap",
    # patterns
    "adversarial_verify",
    "judge_panel",
    "loop_until_budget",
    "loop_until_dry",
    "VerifyResult",
    "SkepticVote",
    "RankedCandidate",
    # errors
    "RdwError",
    "AgentError",
    "AgentTimeout",
    "AgentSchemaError",
    "AgentLimitExceeded",
    "BudgetExceeded",
    "JournalError",
    "JournalWarning",
    "DivergenceWarning",
    "RdwWarning",
    "WorkflowContextError",
    "__version__",
]
