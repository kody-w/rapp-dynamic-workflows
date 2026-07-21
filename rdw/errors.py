"""Typed errors for rapp-dynamic-workflows.

Every failure mode an orchestration script can observe has a dedicated class,
so workflows can react precisely (``except AgentTimeout``) instead of string
matching. ``parallel()`` and ``pipeline()`` swallow ``Exception``-derived
failures into ``None`` results by contract, so most of these surface only when
you call ``agent()`` directly.
"""

from __future__ import annotations


class RdwError(Exception):
    """Base class for all rapp-dynamic-workflows errors."""


class AgentError(RdwError):
    """An agent session failed to produce a usable result.

    Attributes:
        label: Human-readable agent label (as shown in progress output).
    """

    def __init__(self, message: str, *, label: str | None = None) -> None:
        super().__init__(message)
        self.label = label


class AgentTimeout(AgentError):
    """The agent did not go idle within its wall-clock ``timeout``.

    The underlying session is aborted before this is raised, so the turn does
    not keep burning credits in the background.
    """

    def __init__(self, message: str, *, label: str | None = None, timeout: float | None = None) -> None:
        super().__init__(message, label=label)
        self.timeout = timeout


class AgentSchemaError(AgentError):
    """The agent ended its turn without ever calling ``submit_result``.

    Raised only after the nudge ladder is exhausted: the SDK already bounces
    Pydantic validation failures back to the model automatically, and the
    engine re-prompts an idle-without-submit session before giving up.
    """


class ReplayedAgentError(AgentError):
    """An error record replayed from the journal (informational; the engine
    re-executes errored positions live instead of raising this, but it is kept
    public for tooling that inspects journals)."""


class AgentLimitExceeded(RdwError):
    """The run exceeded its lifetime agent-call cap (``max_agents``).

    A runaway loop spawning agents forever is a *run-level misconfiguration*,
    not a per-branch failure — so unlike :class:`AgentError` (which
    ``parallel()``/``pipeline()`` absorb into ``None``), this error propagates
    out of waves and crashes the workflow. Journal-cached replays count toward
    the cap too: the cap bounds calls, not spend (the budget bounds spend).
    """


class BudgetExceeded(RdwError):
    """The run's hard AI-credit ceiling was reached.

    New ``agent()`` calls are refused with this error; already-running agents
    are additionally capped per-session via the SDK's experimental
    ``session_limits.max_ai_credits``.

    Attributes:
        spent: Credits (AIU) spent so far.
        total: The configured ceiling.
    """

    def __init__(self, spent: float, total: float, *, label: str | None = None) -> None:
        super().__init__(
            f"budget exceeded: spent {spent:.2f} of {total:.2f} AI credits"
            + (f" (refusing agent {label!r})" if label else "")
        )
        self.spent = spent
        self.total = total
        self.label = label


class JournalError(RdwError):
    """The run journal is unreadable or structurally invalid.

    Raised only for damage in the *interior* of ``journal.jsonl`` — a torn
    final line (crash mid-append) is skipped with :class:`JournalWarning`
    instead, so the crash the journal exists to recover from can never
    permanently disable ``--resume``."""


class WorkflowContextError(RdwError):
    """A module-level helper (``rdw.agent`` etc.) was called with no active
    ``Workflow`` context. Run the script via ``rdw run`` or enter a
    ``Workflow`` with ``async with``."""


class RdwWarning(UserWarning):
    """Base category for all rdw warnings (``-W error::rdw.RdwWarning`` turns
    every rdw warning into a hard failure at once)."""


class DivergenceWarning(RdwWarning):
    """Emitted when a resumed run's call stream stops matching the journal.

    Not an error: by contract the run simply goes live from the first
    divergent call, but the warning makes silent cache invalidation loud.
    """


class JournalWarning(RdwWarning):
    """Emitted when a resume skips a torn final journal line left by a crash
    mid-append. The rest of the journal still replays normally."""
