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
* ``now()`` / ``random()`` / ``uuid()`` — journaled nondeterminism: the first
  run records the real value, a resume replays it, so timestamps and
  randomness are deterministic under replay instead of forbidden.

Safety caps (contract parity): a run is bounded to ``max_agents`` total agent
calls (default :data:`MAX_AGENTS_PER_RUN`) and a single ``parallel``/
``pipeline`` wave to ``max_wave`` items (default :data:`MAX_WAVE_ITEMS`), so a
bugged while-loop without ``--budget`` cannot burn credits until killed. The
caps complement — never replace — the reservation budget.

Module-level ``agent``/``parallel``/... helpers delegate to the workflow bound
to the current async context, matching the ``from rdw import agent`` ergonomics
of the design sketch.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import random as _random
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence
from uuid import uuid4

from .budget import Budget
from .errors import (
    AgentError,
    AgentLimitExceeded,
    AgentSchemaError,
    AgentTimeout,
    BudgetExceeded,
    WorkflowContextError,
)
from .journal import AgentRecord, Journal, fingerprint
from .progress import Progress, fmt_tokens
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
from .transcripts import TRANSCRIPT_DIR, TranscriptWriter, UsageTap, transcript_filename

DEFAULT_TIMEOUT = 600.0
MAX_SCHEMA_NUDGES = 2

MAX_AGENTS_PER_RUN = 1000
"""Default lifetime cap on ``agent()`` calls per run (cached replays count —
the cap bounds calls, not spend). Raise it via ``max_agents=``/``--max-agents``
for genuinely huge runs; it exists so a runaway loop under the default
unlimited budget fails loudly instead of billing until killed."""

MAX_WAVE_ITEMS = 4096
"""Default cap on items in one ``parallel()``/``pipeline()`` wave. Exceeding it
raises ``ValueError`` before any branch runs — an explicit error, never silent
truncation."""

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
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]


def _stage_arity(stage: Callable[..., Awaitable[Any]]) -> int:
    """Positional arity of a pipeline stage, clamped to 1..3.

    Contract parity: Claude's Workflow tool calls stages with
    ``(prev, originalItem, index)``. Legacy 1-arg callables keep working —
    the engine passes only as many arguments as the stage declares. ``*args``
    counts as "wants everything" (3); un-inspectable callables (builtins,
    some partials) fall back to the legacy single argument.
    """
    try:
        sig = inspect.signature(stage)
    except (TypeError, ValueError):
        return 1
    count = 0
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            count += 1
        elif param.kind is param.VAR_POSITIONAL:
            return 3
    return max(1, min(count, 3))


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
        args: dict[str, Any] | None = None,
        max_agents: int = MAX_AGENTS_PER_RUN,
        max_wave: int = MAX_WAVE_ITEMS,
        transcripts: bool = False,
    ) -> None:
        self.run_id = run_id
        self.runtime = runtime
        self.budget = budget
        self.journal = journal
        self.progress = progress
        self.transcripts = transcripts
        self._replay_saved = 0.0  # AIU served from the journal instead of live
        self.default_model = model
        self.default_effort = effort
        self.default_cwd = cwd
        self.args: dict[str, Any] = dict(args or {})
        # Args parameterize the run, so they are part of replay identity —
        # but ONLY when present: an empty dict contributes nothing to the
        # fingerprint, keeping every pre-args journal replayable byte-for-byte.
        self._args_fp: str | None = (
            hashlib.sha256(
                json.dumps(self.args, sort_keys=True, ensure_ascii=False, default=str).encode(
                    "utf-8"
                )
            ).hexdigest()
            if self.args
            else None
        )
        self._max_agents = max_agents
        self._max_wave = max_wave
        self.declared_phases: list[str] = []
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
        args: dict[str, Any] | None = None,
        max_agents: int = MAX_AGENTS_PER_RUN,
        max_wave: int = MAX_WAVE_ITEMS,
        transcripts: bool = False,
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
            args: Run parameters, readable as ``wf.args`` — the sanctioned
                channel for run-scoped values like timestamps. Non-empty args
                are folded (as one hash) into every agent fingerprint, so the
                same script with different args is a different run identity.
            max_agents: Lifetime cap on agent calls this run (safety net).
            max_wave: Cap on items per ``parallel``/``pipeline`` wave.
            transcripts: Write per-agent session transcripts (filtered event
                JSONL) under ``<run-dir>/agents/`` for turn-by-turn forensics.
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
            args=args,
            max_agents=max_agents,
            max_wave=max_wave,
            transcripts=transcripts,
        )

    # ---------------------------------------------------------------- lifecycle

    async def __aenter__(self) -> "Workflow":
        self._ctx_token = _current_workflow.set(self)
        self.journal.run_boundary(
            event="resume" if self.journal.resume else "start",
            info={
                "ts": time.time(),
                "pid": os.getpid(),
                "budget_total": self.budget.total,
                "model": self.default_model,
                "effort": self.default_effort,
                "rdw_version": _rdw_version(),
                "cache_records_loaded": self.journal.cache_size,
            },
        )
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
        # One loop tick so the plain-mode heartbeat task (cancelled in
        # progress.stop) actually finishes instead of dying with the loop.
        await asyncio.sleep(0)

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
        explore: bool = False,
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
            explore: Keep the full built-in tool catalog (``bash``, ``view``,
                ``rg``, …) even when ``schema`` is set — for agents that must
                investigate before submitting. Narrowing is what makes pure
                extraction reliable, so this is opt-in.
            cwd: Per-agent working directory.

        Raises:
            AgentLimitExceeded: The run hit its ``max_agents`` lifetime cap
                (propagates out of ``parallel``/``pipeline`` — a run-level
                misconfiguration, not a branch failure).
            BudgetExceeded: Refused at admission — the run ceiling is spent.
                Journaled as a ``refusal`` line with a budget snapshot.
            AgentTimeout: The turn exceeded ``timeout`` (session aborted).
            AgentSchemaError: The model never called ``submit_result`` after
                the nudge ladder.
            AgentError: The session errored.
        """
        index = self.journal.next_index()
        if index >= self._max_agents:
            raise AgentLimitExceeded(
                f"run exceeded {self._max_agents} agent calls "
                f"(raise max_agents/--max-agents if this run is legitimately that large)"
            )
        label = label or f"agent-{index}"
        phase = phase or _current_phase.get()
        tool_names = sorted(str(getattr(t, "name", t)) for t in (tools or []))
        opts = {
            "schema": schema_fingerprint(schema),
            "model": model or self.default_model,
            "effort": effort or self.default_effort,
            "tools": tool_names,
            "explore": explore,
            "cwd": cwd or self.default_cwd,
        }
        if self._args_fp is not None:
            # Only when args exist: absent key == pre-args fingerprint, so
            # existing journals replay unchanged.
            opts["args"] = self._args_fp
        # Replay identity is (fingerprint, occurrence) — content-addressed, so
        # the scheduling order of concurrent branches (which varies run to run
        # under parallel/pipeline) never busts the cache on resume.
        fp = fingerprint(prompt, opts)
        seq = self.journal.next_occurrence(fp)

        cached = self.journal.lookup(fp, seq, index=index, label=label)
        if cached is not None:
            self._replay_saved += cached.credits  # the AIU this replay avoided
            self.progress.agent_started(label, phase)
            self.progress.agent_finished(label, "cached")
            return load_value(schema, cached.result or {})

        # Forensic request context, journaled with the outcome (ok or error —
        # the error path is the one that pays for this). Never fingerprinted.
        request: dict[str, Any] = {
            "model": opts["model"],
            "effort": opts["effort"],
            "cwd": opts["cwd"],
            "tools": tool_names,
            "explore": explore,
            "timeout": timeout,
            "prompt_chars": len(prompt),
        }
        usage = UsageTap()  # token/tool telemetry, journaled on ok AND error
        transcript_path: Path | None = None
        if self.transcripts:
            rel = f"{TRANSCRIPT_DIR}/{transcript_filename(index, label)}"
            transcript_path = self.journal.run_dir / rel
            request["transcript"] = rel
        started = time.time()
        session_id: str | None = None
        try:
            self.budget.ensure_available(label=label)
            self.progress.agent_started(label, phase)
            async with self.runtime.slot():
                # Reserve inside the slot so outstanding grants track sessions
                # that can actually run, not branches queued on the semaphore.
                reservation = self.budget.reserve(label=label)
                request["session_limits"] = reservation.limits()
                request["budget"] = self._budget_snapshot()
                try:
                    value, session_id = await self._run_session(
                        prompt,
                        schema=schema,
                        label=label,
                        model=opts["model"],
                        effort=opts["effort"],
                        timeout=timeout,
                        tools=tools,
                        explore=explore,
                        cwd=opts["cwd"],
                        session_limits=reservation.limits(),
                        usage=usage,
                        transcript_path=transcript_path,
                    )
                finally:
                    reservation.release()
        except BudgetExceeded as exc:
            # Refused at admission: no session ran, but the refusal itself is
            # evidence — journal who was refused and the budget state that
            # refused them. lookup() ignores refusal lines, so a resume under
            # a raised budget re-runs this call live.
            self.journal.refusal(
                index=index,
                fp=fp,
                seq=seq,
                label=label,
                phase=phase,
                budget=self._budget_snapshot(),
            )
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
                    usage=usage.snapshot(),
                    request=request,
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
                usage=usage.snapshot(),
                request=request,
            )
        )
        self.progress.agent_finished(label, "ok", f"{credits:.2f} AIU" if credits else "")
        return value

    def _budget_snapshot(self) -> dict[str, Any]:
        """Point-in-time budget state for forensic journal lines."""
        return {
            "total": self.budget.total,
            "spent": self.budget.spent(),
            "outstanding": self.budget.outstanding(),
        }

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
        explore: bool,
        cwd: str | None,
        session_limits: dict[str, float] | None,
        usage: UsageTap | None = None,
        transcript_path: Path | None = None,
    ) -> tuple[Any, str]:
        """Create the session, drive it to a result, and always disconnect.

        ``usage`` and ``transcript_path`` wire the observability taps: both
        subscribe next to the budget/progress taps and only observe — spend
        accounting stays the Budget's job alone.
        """
        capture: SubmitCapture | None = None
        tool_list: list[Any] = list(tools or [])
        system_message: dict[str, Any] | None = None
        available_tools: list[str] | None = None
        if schema is not None:
            capture = SubmitCapture()
            tool_list.append(build_submit_tool(schema, capture))
            system_message = {"mode": "append", "content": SUBMIT_INSTRUCTION}
            if not tools and not explore:
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
        try:
            session = await self.runtime.create_session(
                **{k: v for k, v in kwargs.items() if v is not None}
            )
        except Exception as exc:
            # SDK-level rejections (JSON-RPC errors, transport failures) must
            # surface as AgentError so parallel() branches resolve to None
            # instead of crashing the whole workflow.
            raise AgentError(
                f"session create failed for {label!r}: {exc}", label=label
            ) from exc
        session_id = session.session_id
        unsubscribes = [
            session.on(self.budget.tap(session_id)),
            session.on(self._progress_tap(label)),
        ]
        if usage is not None:
            unsubscribes.append(session.on(usage.tap()))
        transcript: TranscriptWriter | None = None
        if transcript_path is not None:
            transcript = TranscriptWriter(transcript_path)
            unsubscribes.append(session.on(transcript.tap()))
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
            if transcript is not None:
                transcript.close()
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
        """Feed token counts and tool activity into the progress board.

        ``assistant.usage`` drives the output-token counter;
        ``tool.execution_start`` stamps the agent's last-activity marker so
        the heartbeat / rich tree can show ``last: bash 3s ago`` — the signal
        that separates a working agent from a wedged one inside the timeout
        window.
        """

        def handler(event: Any) -> None:
            try:
                etype = getattr(getattr(event, "type", None), "value", None)
                if etype == "assistant.usage":
                    n = getattr(getattr(event, "data", None), "output_tokens", None)
                    if n:
                        self.progress.agent_tokens(label, int(n))
                elif etype == "tool.execution_start":
                    name = getattr(getattr(event, "data", None), "tool_name", None)
                    self.progress.agent_activity(label, str(name) if name else "tool")
            except Exception:
                pass

        return handler

    # ---------------------------------------------------------------- parallel

    async def parallel(self, thunks: Sequence[Thunk]) -> list[Any]:
        """Run branches concurrently; a failed branch becomes ``None``.

        Never raises for ``Exception``-derived failures — including
        ``BudgetExceeded``, deliberately: a capped wave *degrades* instead of
        crashing, because budget exhaustion is an expected runtime condition.
        The one asymmetric exception is :class:`AgentLimitExceeded`, which
        propagates: the ``max_agents`` cap firing means the *run* is
        misconfigured (usually a runaway loop), and silently degrading it to
        ``None`` branches would hide exactly the bug the cap exists to catch.
        Cancellation still propagates. Results keep input order.

        Raises:
            ValueError: More than ``max_wave`` branches (before any runs).
            AgentLimitExceeded: A branch hit the run's ``max_agents`` cap.
        """
        thunks = list(thunks)
        if len(thunks) > self._max_wave:
            raise ValueError(f"{len(thunks)} items exceeds max_wave={self._max_wave}")

        async def run(thunk: Thunk) -> Any:
            try:
                aw = thunk() if callable(thunk) else thunk
                return await aw
            except AgentLimitExceeded:
                raise  # run-level misconfiguration — never degrade to None
            except Exception as exc:
                self.log(f"parallel branch failed: {exc}")
                return None

        return list(await asyncio.gather(*(run(t) for t in thunks)))

    # ---------------------------------------------------------------- pipeline

    async def pipeline(
        self, items: Sequence[Any], *stages: Callable[..., Awaitable[Any]]
    ) -> list[Any]:
        """Flow each item through ``stages`` with no barrier between stages.

        Every item advances to its next stage the moment the previous one
        returns — item 3 can be in stage 1 while item 1 is in stage 3. A stage
        exception (or a stage returning ``None``) drops the item to ``None``
        and skips its remaining stages — except :class:`AgentLimitExceeded`,
        which propagates (same run-level taxonomy as :meth:`parallel`; budget
        exhaustion still degrades). Results keep input order.

        Stages are called by declared positional arity (contract parity with
        ``(prev, originalItem, index)``): 1-arg stages get ``stage(current)``
        (the legacy shape), 2-arg get ``stage(current, item)``, 3-arg get
        ``stage(current, item, index)`` where ``item`` is the *original* input
        item and ``index`` its position.

        Raises:
            ValueError: More than ``max_wave`` items (before any stage runs).
            AgentLimitExceeded: A stage hit the run's ``max_agents`` cap.
        """
        items = list(items)
        if len(items) > self._max_wave:
            raise ValueError(f"{len(items)} items exceeds max_wave={self._max_wave}")
        arities = [_stage_arity(stage) for stage in stages]  # inspect once, not per item

        async def flow(index: int, item: Any) -> Any:
            current = item
            for stage, arity in zip(stages, arities):
                if current is None:
                    return None
                try:
                    if arity >= 3:
                        current = await stage(current, item, index)
                    elif arity == 2:
                        current = await stage(current, item)
                    else:
                        current = await stage(current)
                except AgentLimitExceeded:
                    raise  # run-level misconfiguration — never degrade to None
                except Exception as exc:
                    self.log(f"pipeline stage failed for item {item!r}: {exc}")
                    return None
            return current

        return list(await asyncio.gather(*(flow(i, item) for i, item in enumerate(items))))

    # ------------------------------------------------------------ phase & log

    def phase(self, title: str) -> _Phase:
        """Scope journal grouping and progress under ``title``.

        Works with both ``with`` and ``async with``; concurrent tasks spawned
        inside inherit the phase through the async context.
        """
        return _Phase(self, title)

    def declare_phases(self, titles: Sequence[str]) -> None:
        """Pre-declare the run's phases for progress display.

        Called by ``rdw run`` when the script exposes a module-level
        ``PHASES = [...]`` list, or directly as the first call in a workflow.
        Display-only: declared phases render as pending branches in the
        progress tree and land in ``meta.json`` — they are never part of any
        fingerprint, so adding or reordering them cannot bust the cache.
        """
        self.declared_phases = [str(t) for t in titles]
        self.progress.declare_phases(self.declared_phases)

    def log(self, message: str) -> None:
        """Emit a progress line and a non-replayable journal note."""
        self.progress.log(message)
        self.journal.note(message, phase=_current_phase.get())

    # ------------------------------------------- journaled nondeterminism

    def now(self) -> float:
        """Wall-clock ``time.time()``, journaled for deterministic replay.

        The first run records the real timestamp; a resume replays the
        recorded one — strictly stronger than forbidding wall-clock (Claude's
        Workflow tool makes ``Date.now()`` throw): timestamps become
        deterministic under replay instead of banned.
        """
        return self._journaled_value("now", time.time)

    def random(self) -> float:
        """A float in ``[0, 1)`` like ``random.random()``, journaled for
        deterministic replay (see :meth:`now`)."""
        return self._journaled_value("random", _random.random)

    def uuid(self) -> str:
        """A ``uuid4`` string, journaled for deterministic replay (see
        :meth:`now`)."""
        return self._journaled_value("uuid", lambda: str(uuid4()))

    def _journaled_value(self, kind: str, produce: Callable[[], Any]) -> Any:
        """Record-or-replay one nondeterministic value, keyed by
        ``(kind, occurrence)`` exactly like agent records — so value replay is
        scheduling-independent too."""
        seq = self.journal.next_value_occurrence(kind)
        cached = self.journal.value_lookup(kind, seq)
        if cached is not None:
            return cached
        value = produce()
        self.journal.value_record(kind, seq, value)
        return value

    # ---------------------------------------------------------------- reporting

    def report(self) -> str:
        """Human summary: spend, cache hits, per-agent table, phase rollups.

        Per-phase rollups (agents, AIU, tokens, wall time) appear when the run
        used phases; a ``replay saved`` line appears when journal replay
        avoided live spend this run.
        """
        records = self.journal.records()
        lines = [
            f"run {self.run_id}: {self.budget.summary()}, "
            f"{self.journal.cache_hits} cache hit(s)"
            + (", DIVERGED" if self.journal.diverged else "")
        ]
        for rec in records:
            mark = "✓" if rec.status == "ok" else "✗"
            loc = f"[{rec.phase}] " if rec.phase else ""
            wall = max(0.0, rec.ended - rec.started)
            lines.append(
                f"  {mark} #{rec.index:<3} {loc}{rec.label}: {rec.status}, "
                f"{rec.credits:.2f} AIU, {wall:.1f}s"
            )
        if any(rec.phase for rec in records):
            lines.append("  phases:")
            lines.extend(f"    {line}" for line in phase_rollup_lines(records))
        if self._replay_saved > 0:
            lines.append(f"  replay saved ~{self._replay_saved:.2f} AIU")
        return "\n".join(lines)


def _event_text(event: Any) -> str:
    """Final assistant message text from a ``send_and_wait`` return value."""
    if event is None:
        return ""
    content = getattr(getattr(event, "data", None), "content", None)
    return content if isinstance(content, str) else ""


def _usage_sum(records: Sequence[AgentRecord], *fields: str) -> int:
    """Sum ``AgentRecord.usage`` counters across records (missing → 0)."""
    total = 0
    for rec in records:
        usage = rec.usage or {}
        total += sum(int(usage.get(field) or 0) for field in fields)
    return total


def phase_rollup_lines(records: Sequence[AgentRecord]) -> list[str]:
    """Per-phase rollup lines, first-seen phase order.

    One line per phase — ``design: 3 agent(s) (3 ok), 211.20 AIU, 148.0k tok,
    402.1s wall`` — with the token figure (input + output from journaled
    ``usage`` telemetry) shown only when any was recorded. Shared by
    ``Workflow.report()`` and ``rdw show --stats`` so the arithmetic lives in
    exactly one place.
    """
    groups: dict[str | None, list[AgentRecord]] = {}
    for rec in records:
        groups.setdefault(rec.phase, []).append(rec)
    lines: list[str] = []
    for phase, recs in groups.items():
        credits = sum(r.credits for r in recs)
        tokens = _usage_sum(recs, "input_tokens", "output_tokens")
        wall = sum(max(0.0, r.ended - r.started) for r in recs)
        ok = sum(1 for r in recs if r.status == "ok")
        line = f"{phase or '(no phase)'}: {len(recs)} agent(s) ({ok} ok), {credits:.2f} AIU"
        if tokens:
            line += f", {fmt_tokens(tokens)}"
        line += f", {wall:.1f}s wall"
        lines.append(line)
    return lines


def _rdw_version() -> str:
    """The installed rdw version for run-boundary lines (lazy import: the
    package is fully initialized by the time a Workflow is entered)."""
    try:
        from rdw import __version__

        return __version__
    except Exception:  # pragma: no cover - only on exotic import setups
        return "unknown"


# ---------------------------------------------------------------------------
# Module-level convenience API (bound to the ambient Workflow)
# ---------------------------------------------------------------------------


async def agent(prompt: str, **kwargs: Any) -> Any:
    """``current_workflow().agent(...)`` — see :meth:`Workflow.agent`."""
    return await current_workflow().agent(prompt, **kwargs)


async def parallel(thunks: Sequence[Thunk]) -> list[Any]:
    """``current_workflow().parallel(...)`` — see :meth:`Workflow.parallel`."""
    return await current_workflow().parallel(thunks)


async def pipeline(items: Sequence[Any], *stages: Callable[..., Awaitable[Any]]) -> list[Any]:
    """``current_workflow().pipeline(...)`` — see :meth:`Workflow.pipeline`."""
    return await current_workflow().pipeline(items, *stages)


def phase(title: str) -> _Phase:
    """``current_workflow().phase(...)`` — see :meth:`Workflow.phase`."""
    return current_workflow().phase(title)


def log(message: str) -> None:
    """``current_workflow().log(...)`` — see :meth:`Workflow.log`."""
    current_workflow().log(message)


def now() -> float:
    """``current_workflow().now()`` — journaled wall-clock, see :meth:`Workflow.now`."""
    return current_workflow().now()


def random() -> float:
    """``current_workflow().random()`` — journaled RNG, see :meth:`Workflow.random`."""
    return current_workflow().random()


def uuid() -> str:
    """``current_workflow().uuid()`` — journaled uuid4, see :meth:`Workflow.uuid`."""
    return current_workflow().uuid()
