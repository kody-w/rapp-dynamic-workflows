"""The Workflow engine: ``agent`` / ``parallel`` / ``pipeline`` / ``phase`` / ``log``.

A :class:`Workflow` is plain Python driving hermetic Copilot SDK sessions —
the orchestrator itself spends zero tokens. Each ``agent()`` call is one fresh
``create_session`` (isolated system prompt, own working directory, own
model/effort, own credit cap), so agents behave like deterministic functions
of their prompt and options. That per-call hermeticity is what makes the
journal's fingerprinted replay meaningful.

Contract semantics implemented here (Workflow-tool parity):

* ``agent(prompt, ...)`` — one hermetic session; structured output via the
  submit-tool pattern when ``schema`` is given; wall-clock ``timeout``
  enforced with ``session.abort()`` so timed-out turns never keep burning.
* ``parallel(thunks)`` — a barrier; a failing branch resolves to ``None``,
  never poisons its siblings, and ``parallel`` itself never raises.
* ``pipeline(items, *stages)`` — per-item flow with **no barrier between
  stages**: item 3 can be in stage 1 while item 1 is in stage 3. A stage
  exception drops that item to ``None``.
* ``phase(title)`` — a (sync or async) context manager that scopes journal
  grouping and progress display via a ``ContextVar``, so concurrently running
  tasks inherit the phase they were spawned under.
* ``log(msg)`` — progress line plus a non-replayable journal note.

Module-level ``agent``/``parallel``/... helpers delegate to the workflow bound
to the current async context, matching the ``from rdw import agent`` ergonomics
of the design sketch.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from .budget import Budget
from .errors import (
    AgentError,
    AgentSchemaError,
    AgentTimeout,
    BudgetExceeded,
    WorkflowContextError,
)
from .journal import AgentRecord, Journal, fingerprint
from .progress import Progress
from .runtime import CopilotRuntime, Runtime, SessionHandle
from .schema import (
    NUDGE_PROMPT,
    SUBMIT_INSTRUCTION,
    SUBMIT_TOOL_NAME,
    SchemaSpec,
    SubmitCapture,
    build_submit_tool,
    dump_value,
    load_value,
    schema_fingerprint,
)

DEFAULT_TIMEOUT = 600.0
MAX_SCHEMA_NUDGES = 2

Thunk = Callable[[], Awaitable[Any]] | Awaitable[Any]
"""A parallel branch: a zero-arg callable returning an awaitable, or an
awaitable directly (coroutines work, but callables replay-safely defer
creation until the branch actually runs)."""

_current_workflow: ContextVar["Workflow | None"] = ContextVar("rdw_workflow", default=None)
_current_phase: ContextVar[str | None] = ContextVar("rdw_phase", default=None)


def current_workflow() -> "Workflow":
    """The Workflow bound to the current async context (set by ``async with``)."""
    wf = _current_workflow.get()
    if wf is None:
        raise WorkflowContextError(
            "no active Workflow — run the script with `rdw run` or wrap your "
            "code in `async with Workflow.open(...) as wf:`"
        )
    return wf


def new_run_id() -> str:
    """Sortable, collision-resistant run id: ``YYYYmmdd-HHMMSS-xxxxxx``."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


class _Phase:
    """``phase(title)`` context manager — usable with ``with`` or ``async with``."""

    def __init__(self, wf: "Workflow", title: str) -> None:
        self._wf = wf
        self._title = title
        self._token: Any = None

    def __enter__(self) -> "_Phase":
        self._token = _current_phase.set(self._title)
        self._wf.progress.phase_started(self._title)
        self._wf.journal.note(f"phase started: {self._title}", phase=self._title)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._wf.journal.note(f"phase ended: {self._title}", phase=self._title)
        if self._token is not None:
            _current_phase.reset(self._token)
            self._token = None

    async def __aenter__(self) -> "_Phase":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__(*exc)


class Workflow:
    """One orchestration run: owns the runtime, budget, journal, and progress.

    Use :meth:`open` for the batteries-included construction (run directory,
    journal, defaults), or pass components explicitly for tests::

        wf = Workflow(run_id="t", runtime=FakeRuntime(), budget=Budget(),
                      journal=Journal(tmp_path), progress=Progress(force_plain=True))
        async with wf:
            out = await wf.agent("hello", label="a")
    """

    def __init__(
        self,
        *,
        run_id: str,
        runtime: Runtime,
        budget: Budget,
        journal: Journal,
        progress: Progress,
        model: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.runtime = runtime
        self.budget = budget
        self.journal = journal
        self.progress = progress
        self.default_model = model
        self.default_effort = effort
        self.default_cwd = cwd
        self._ctx_token: Any = None

    # ------------------------------------------------------------ construction

    @classmethod
    def open(
        cls,
        *,
        run_id: str | None = None,
        root: str | Path = ".rdw",
        resume: bool = False,
        budget: float | Budget | None = None,
        runtime: Runtime | None = None,
        progress: Progress | None = None,
        model: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        concurrency: int | None = None,
    ) -> "Workflow":
        """Create a Workflow with its run directory under ``<root>/runs/<id>``.

        Args:
            run_id: Reuse an existing id (required for ``resume=True``);
                a fresh sortable id is generated otherwise.
            resume: Load the run's journal as a replay cache.
            budget: A credit ceiling (float), a prebuilt :class:`Budget`,
                or ``None`` for unlimited-with-accounting.
            runtime: Session factory; defaults to a shared-client
                :class:`CopilotRuntime`. Tests pass a fake here.
        """
        rid = run_id or new_run_id()
        run_dir = Path(root) / "runs" / rid
        b = budget if isinstance(budget, Budget) else Budget(total=budget)
        journal = Journal(run_dir, resume=resume)
        rt = runtime or CopilotRuntime(working_directory=cwd, concurrency=concurrency)
        prog = progress or Progress(rid, budget=b)
        return cls(
            run_id=rid,
            runtime=rt,
            budget=b,
            journal=journal,
            progress=prog,
            model=model,
            effort=effort,
            cwd=cwd,
        )

    # ---------------------------------------------------------------- lifecycle

    async def __aenter__(self) -> "Workflow":
        self._ctx_token = _current_workflow.set(self)
        self.progress.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        try:
            await self.runtime.close()
        finally:
            self.progress.stop()
            if self._ctx_token is not None:
                _current_workflow.reset(self._ctx_token)
                self._ctx_token = None

    # -------------------------------------------------------------------- agent

    async def agent(
        self,
        prompt: str,
        *,
        schema: SchemaSpec | None = None,
        label: str | None = None,
        phase: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        tools: Sequence[Any] | None = None,
        cwd: str | None = None,
    ) -> Any:
        """Run one hermetic agent and return its result.

        Args:
            prompt: The agent's task. Everything the agent should know goes
                here — sessions are isolated and share no ambient state.
            schema: A Pydantic model class or raw JSON-schema dict. When set,
                the result is the validated instance (or dict) the agent
                passed to the forced ``submit_result`` tool; when ``None``,
                the result is the final assistant message text.
            label: Display/journal name (auto ``agent-<n>`` otherwise).
            phase: Overrides the ambient ``phase(...)`` title.
            model / effort: Per-agent model and reasoning-effort overrides
                (workflow defaults apply otherwise).
            timeout: Wall-clock seconds before the session is aborted and
                :class:`AgentTimeout` raised.
            tools: Extra SDK ``Tool`` objects for this agent. When omitted and
                ``schema`` is set, the session's tool catalog is narrowed to
                just ``submit_result`` (tool-choice pressure).
            cwd: Per-agent working directory.

        Raises:
            BudgetExceeded: Refused at admission — the run ceiling is spent.
            AgentTimeout: The turn exceeded ``timeout`` (session aborted).
            AgentSchemaError: The model never called ``submit_result`` after
                the nudge ladder.
            AgentError: The session errored.
        """
        index = self.journal.next_index()
        label = label or f"agent-{index}"
        phase = phase or _current_phase.get()
        tool_names = sorted(str(getattr(t, "name", t)) for t in (tools or []))
        opts = {
            "schema": schema_fingerprint(schema),
            "model": model or self.default_model,
            "effort": effort or self.default_effort,
            "tools": tool_names,
            "cwd": cwd or self.default_cwd,
        }
        # Replay identity is (fingerprint, occurrence) — content-addressed, so
        # the scheduling order of concurrent branches (which varies run to run
        # under parallel/pipeline) never busts the cache on resume.
        fp = fingerprint(prompt, opts)
        seq = self.journal.next_occurrence(fp)

        cached = self.journal.lookup(fp, seq, index=index, label=label)
        if cached is not None:
            self.progress.agent_started(label, phase)
            self.progress.agent_finished(label, "cached")
            return load_value(schema, cached.result or {})

        self.budget.ensure_available(label=label)
        self.progress.agent_started(label, phase)

        started = time.time()
        session_id: str | None = None
        try:
            async with self.runtime.slot():
                # Reserve inside the slot so outstanding grants track sessions
                # that can actually run, not branches queued on the semaphore.
                reservation = self.budget.reserve(label=label)
                try:
                    value, session_id = await self._run_session(
                        prompt,
                        schema=schema,
                        label=label,
                        model=opts["model"],
                        effort=opts["effort"],
                        timeout=timeout,
                        tools=tools,
                        cwd=opts["cwd"],
                        session_limits=reservation.limits(),
                    )
                finally:
                    reservation.release()
        except BudgetExceeded as exc:
            # Refused at admission: no session, no journal record.
            self.progress.agent_finished(label, "error", str(exc))
            raise
        except AgentError as exc:
            self.journal.record(
                AgentRecord(
                    index=index,
                    fp=fp,
                    seq=seq,
                    label=label,
                    phase=phase,
                    status="error",
                    error=str(exc),
                    session_id=session_id,
                    credits=self.budget.session_spent(session_id) if session_id else 0.0,
                    started=started,
                    ended=time.time(),
                )
            )
            self.progress.agent_finished(
                label, "timeout" if isinstance(exc, AgentTimeout) else "error", str(exc)
            )
            raise

        credits = self.budget.session_spent(session_id) if session_id else 0.0
        self.journal.record(
            AgentRecord(
                index=index,
                fp=fp,
                seq=seq,
                label=label,
                phase=phase,
                status="ok",
                result=dump_value(value),
                session_id=session_id,
                credits=credits,
                started=started,
                ended=time.time(),
            )
        )
        self.progress.agent_finished(label, "ok", f"{credits:.2f} AIU" if credits else "")
        return value

    async def _run_session(
        self,
        prompt: str,
        *,
        schema: SchemaSpec | None,
        label: str,
        model: str | None,
        effort: str | None,
        timeout: float,
        tools: Sequence[Any] | None,
        cwd: str | None,
        session_limits: dict[str, float] | None,
    ) -> tuple[Any, str]:
        """Create the session, drive it to a result, and always disconnect."""
        capture: SubmitCapture | None = None
        tool_list: list[Any] = list(tools or [])
        system_message: dict[str, Any] | None = None
        available_tools: list[str] | None = None
        if schema is not None:
            capture = SubmitCapture()
            tool_list.append(build_submit_tool(schema, capture))
            system_message = {"mode": "append", "content": SUBMIT_INSTRUCTION}
            if not tools:
                # Pure-extraction agent: the only tool it can reach for is submit.
                available_tools = [SUBMIT_TOOL_NAME]

        kwargs: dict[str, Any] = {
            "model": model,
            "reasoning_effort": effort,
            "working_directory": cwd,
            "tools": tool_list or None,
            "system_message": system_message,
            "available_tools": available_tools,
            "session_limits": session_limits,
            "skip_custom_instructions": True,
            "include_sub_agent_streaming_events": False,
        }
        session = await self.runtime.create_session(
            **{k: v for k, v in kwargs.items() if v is not None}
        )
        session_id = session.session_id
        unsubscribes = [
            session.on(self.budget.tap(session_id)),
            session.on(self._progress_tap(label)),
        ]
        try:
            event = await self._send(session, prompt, timeout=timeout, label=label)
            if capture is None:
                return _event_text(event), session_id
            for _ in range(MAX_SCHEMA_NUDGES):
                if capture.called:
                    break
                event = await self._send(session, NUDGE_PROMPT, timeout=timeout, label=label)
            if not capture.called:
                raise AgentSchemaError(
                    f"agent {label!r} ended its turn without calling "
                    f"{SUBMIT_TOOL_NAME} after {MAX_SCHEMA_NUDGES} nudges",
                    label=label,
                )
            return capture.value, session_id
        finally:
            for unsub in unsubscribes:
                with contextlib.suppress(Exception):
                    unsub()
            with contextlib.suppress(Exception):
                await session.disconnect()

    async def _send(
        self, session: SessionHandle, prompt: str, *, timeout: float, label: str
    ) -> Any:
        """``send_and_wait`` with abort-on-timeout and typed errors.

        The raw SDK ``send_and_wait`` raises ``TimeoutError`` *without*
        aborting the in-flight turn; here the session is aborted first so a
        timed-out agent stops spending immediately.
        """
        try:
            return await session.send_and_wait(prompt, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            with contextlib.suppress(Exception):
                await session.abort()
            raise AgentTimeout(
                f"agent {label!r} exceeded {timeout:.0f}s and was aborted",
                label=label,
                timeout=timeout,
            ) from None
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"agent {label!r} session error: {exc}", label=label) from exc

    def _progress_tap(self, label: str) -> Callable[[Any], None]:
        """Feed output-token counts from ``assistant.usage`` into the tree."""

        def handler(event: Any) -> None:
            try:
                etype = getattr(getattr(event, "type", None), "value", None)
                if etype == "assistant.usage":
                    n = getattr(getattr(event, "data", None), "output_tokens", None)
                    if n:
                        self.progress.agent_tokens(label, int(n))
            except Exception:
                pass

        return handler

    # ---------------------------------------------------------------- parallel

    async def parallel(self, thunks: Sequence[Thunk]) -> list[Any]:
        """Run branches concurrently; a failed branch becomes ``None``.

        Never raises for ``Exception``-derived failures (including
        ``BudgetExceeded`` — a capped wave degrades instead of crashing);
        cancellation still propagates. Results keep input order.
        """

        async def run(thunk: Thunk) -> Any:
            try:
                aw = thunk() if callable(thunk) else thunk
                return await aw
            except Exception as exc:
                self.log(f"parallel branch failed: {exc}")
                return None

        return list(await asyncio.gather(*(run(t) for t in thunks)))

    # ---------------------------------------------------------------- pipeline

    async def pipeline(
        self, items: Sequence[Any], *stages: Callable[[Any], Awaitable[Any]]
    ) -> list[Any]:
        """Flow each item through ``stages`` with no barrier between stages.

        Every item advances to its next stage the moment the previous one
        returns — item 3 can be in stage 1 while item 1 is in stage 3. A stage
        exception (or a stage returning ``None``) drops the item to ``None``
        and skips its remaining stages. Results keep input order.
        """

        async def flow(item: Any) -> Any:
            current = item
            for stage in stages:
                if current is None:
                    return None
                try:
                    current = await stage(current)
                except Exception as exc:
                    self.log(f"pipeline stage failed for item {item!r}: {exc}")
                    return None
            return current

        return list(await asyncio.gather(*(flow(i) for i in items)))

    # ------------------------------------------------------------ phase & log

    def phase(self, title: str) -> _Phase:
        """Scope journal grouping and progress under ``title``.

        Works with both ``with`` and ``async with``; concurrent tasks spawned
        inside inherit the phase through the async context.
        """
        return _Phase(self, title)

    def log(self, message: str) -> None:
        """Emit a progress line and a non-replayable journal note."""
        self.progress.log(message)
        self.journal.note(message, phase=_current_phase.get())

    # ---------------------------------------------------------------- reporting

    def report(self) -> str:
        """Human summary: spend, cache hits, and per-agent credit table."""
        lines = [
            f"run {self.run_id}: {self.budget.summary()}, "
            f"{self.journal.cache_hits} cache hit(s)"
            + (", DIVERGED" if self.journal.diverged else "")
        ]
        for rec in self.journal.records():
            mark = "✓" if rec.status == "ok" else "✗"
            loc = f"[{rec.phase}] " if rec.phase else ""
            wall = max(0.0, rec.ended - rec.started)
            lines.append(
                f"  {mark} #{rec.index:<3} {loc}{rec.label}: {rec.status}, "
                f"{rec.credits:.2f} AIU, {wall:.1f}s"
            )
        return "\n".join(lines)


def _event_text(event: Any) -> str:
    """Final assistant message text from a ``send_and_wait`` return value."""
    if event is None:
        return ""
    content = getattr(getattr(event, "data", None), "content", None)
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Module-level convenience API (bound to the ambient Workflow)
# ---------------------------------------------------------------------------


async def agent(prompt: str, **kwargs: Any) -> Any:
    """``current_workflow().agent(...)`` — see :meth:`Workflow.agent`."""
    return await current_workflow().agent(prompt, **kwargs)


async def parallel(thunks: Sequence[Thunk]) -> list[Any]:
    """``current_workflow().parallel(...)`` — see :meth:`Workflow.parallel`."""
    return await current_workflow().parallel(thunks)


async def pipeline(items: Sequence[Any], *stages: Callable[[Any], Awaitable[Any]]) -> list[Any]:
    """``current_workflow().pipeline(...)`` — see :meth:`Workflow.pipeline`."""
    return await current_workflow().pipeline(items, *stages)


def phase(title: str) -> _Phase:
    """``current_workflow().phase(...)`` — see :meth:`Workflow.phase`."""
    return current_workflow().phase(title)


def log(message: str) -> None:
    """``current_workflow().log(...)`` — see :meth:`Workflow.log`."""
    current_workflow().log(message)
