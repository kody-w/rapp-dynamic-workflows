"""Shared test scaffolding: a FakeCopilot layer that never spawns the CLI.

Everything here impersonates the slice of the github-copilot-sdk that
``rdw`` actually touches (verified against the installed SDK source):

* :class:`FakeSession` — implements the ``SessionHandle`` protocol
  (``session_id``, ``send_and_wait``, ``on``, ``abort``, ``disconnect``) and
  executes a script of :class:`Turn` objects. A turn can emit usage events to
  registered handlers, call the ``submit_result`` tool handler with a sequence
  of payloads (retrying on ``result_type == "failure"`` exactly like the real
  model does when the SDK bounces invalid arguments), return final message
  text, or raise.
* :class:`FakeRuntime` — a ``rdw.runtime.BaseRuntime`` subclass whose
  ``create_session`` pops scripted sessions from a queue and records every
  kwarg for assertions.
* :class:`FakeClient` — stands in for ``copilot.CopilotClient`` behind
  ``CopilotRuntime``'s ``client_factory`` seam.

An autouse fixture monkeypatches ``CopilotRuntime._default_factory`` to raise,
so no test can ever construct a real client (and thereby spawn the ``copilot``
child process) by accident.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Run the tests against the checked-out package without requiring an install
# (the repo has no README yet, which blocks editable builds).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from copilot.tools import ToolInvocation  # noqa: E402  (pure dataclass import)

from rdw.budget import Budget  # noqa: E402
from rdw.engine import Workflow  # noqa: E402
from rdw.journal import Journal  # noqa: E402
from rdw.progress import Progress  # noqa: E402
from rdw.runtime import BaseRuntime, CopilotRuntime  # noqa: E402

_ids = itertools.count(1)


# ---------------------------------------------------------------------------
# Fake SDK events (shapes verified against copilot/generated/session_events)
# ---------------------------------------------------------------------------


def event(etype: str, **data: Any) -> Any:
    """A generic fake SessionEvent: ``event("tool.execution_start", tool_name="bash")``."""
    return SimpleNamespace(type=SimpleNamespace(value=etype), data=SimpleNamespace(**data))


def usage_event(nano_aiu: float, output_tokens: int | None = None) -> Any:
    """An ``assistant.usage`` event: per-call cost in nano-AIU."""
    return SimpleNamespace(
        type=SimpleNamespace(value="assistant.usage"),
        data=SimpleNamespace(
            copilot_usage=SimpleNamespace(total_nano_aiu=nano_aiu),
            output_tokens=output_tokens,
        ),
    )


def checkpoint_event(nano_aiu: float) -> Any:
    """A ``session.usage_checkpoint`` event: cumulative session spend."""
    return SimpleNamespace(
        type=SimpleNamespace(value="session.usage_checkpoint"),
        data=SimpleNamespace(total_nano_aiu=nano_aiu),
    )


# ---------------------------------------------------------------------------
# Scripted sessions
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One scripted ``send_and_wait`` exchange.

    Attributes:
        text: Final assistant message content returned for this turn.
        submit: Payloads the fake model passes to ``submit_result`` in order;
            it stops at the first ``success`` result (mirroring the SDK's
            invalid-arguments retry loop).
        events: Events emitted to every ``session.on`` handler first.
        error: Raised instead of completing (e.g. ``TimeoutError()``).
    """

    text: str = ""
    submit: list[Any] | None = None
    events: list[Any] = field(default_factory=list)
    error: BaseException | None = None


class FakeSession:
    """SessionHandle-conformant scripted session."""

    def __init__(self, turns: list[Turn] | None = None, session_id: str | None = None):
        self.session_id = session_id or f"fake-session-{next(_ids)}"
        self.turns = list(turns or [])
        self.tools: list[Any] = []  # populated by the fake runtime/client
        self.prompts: list[str] = []
        self.handlers: list[Any] = []
        self.submit_results: list[Any] = []
        self.aborted = False
        self.disconnected = False

    def on(self, handler):
        self.handlers.append(handler)

        def unsubscribe():
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    def _emit(self, event: Any) -> None:
        for handler in list(self.handlers):
            handler(event)

    def emit(self, event: Any) -> None:
        """Drive every registered ``session.on`` handler with ``event`` —
        lets tests feed taps directly, outside any scripted turn."""
        self._emit(event)

    def _submit_tool(self) -> Any:
        for tool in self.tools:
            if getattr(tool, "name", None) == "submit_result":
                return tool
        raise AssertionError("scripted submit but session has no submit_result tool")

    async def send_and_wait(self, prompt: str, *, timeout: float = 60.0) -> Any:
        self.prompts.append(prompt)
        turn = self.turns.pop(0) if self.turns else Turn()
        if turn.error is not None:
            raise turn.error
        for event in turn.events:
            self._emit(event)
        if turn.submit is not None:
            tool = self._submit_tool()
            for payload in turn.submit:
                result = await tool.handler(
                    ToolInvocation(
                        session_id=self.session_id,
                        tool_call_id=f"call-{next(_ids)}",
                        tool_name="submit_result",
                        arguments=payload,
                    )
                )
                self.submit_results.append(result)
                if result.result_type == "success":
                    break
        return SimpleNamespace(
            type=SimpleNamespace(value="session.idle"),
            data=SimpleNamespace(content=turn.text),
        )

    async def abort(self) -> None:
        self.aborted = True

    async def disconnect(self) -> None:
        self.disconnected = True


# ---------------------------------------------------------------------------
# Fake runtime / client
# ---------------------------------------------------------------------------


class FakeRuntime(BaseRuntime):
    """BaseRuntime subclass serving scripted sessions from a queue.

    Queue items may be :class:`FakeSession` instances or plain ``list[Turn]``
    scripts. An empty queue serves default single-turn text sessions.
    """

    def __init__(self, sessions: list[Any] | None = None, concurrency: int = 8):
        super().__init__(concurrency)
        self.queue: list[Any] = list(sessions or [])
        self.created: list[FakeSession] = []
        self.create_kwargs: list[dict[str, Any]] = []
        self.closed = False

    async def create_session(self, **kwargs: Any) -> FakeSession:
        if self.queue:
            item = self.queue.pop(0)
            session = item if isinstance(item, FakeSession) else FakeSession(item)
        else:
            session = FakeSession([Turn(text="default")])
        session.tools = list(kwargs.get("tools") or [])
        self.create_kwargs.append(kwargs)
        self.created.append(session)
        return session

    async def close(self) -> None:
        self.closed = True


class FakeClient:
    """Stands in for ``copilot.CopilotClient`` behind the client_factory seam."""

    def __init__(self, sessions: list[Any] | None = None):
        self.queue: list[Any] = list(sessions or [])
        self.create_kwargs: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def create_session(self, **kwargs: Any) -> FakeSession:
        if self.queue:
            item = self.queue.pop(0)
            session = item if isinstance(item, FakeSession) else FakeSession(item)
        else:
            session = FakeSession([Turn(text="default")])
        session.tools = list(kwargs.get("tools") or [])
        self.create_kwargs.append(kwargs)
        return session

    async def stop(self) -> None:
        self.stopped = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_real_copilot(monkeypatch):
    """Hard guard: no test may ever build a real CopilotClient (child spawn)."""

    def _blocked(self):  # pragma: no cover - only fires on a test bug
        raise AssertionError(
            "test attempted to construct a real CopilotClient — use FakeRuntime "
            "or CopilotRuntime(client_factory=FakeClient())"
        )

    monkeypatch.setattr(CopilotRuntime, "_default_factory", _blocked)


@pytest.fixture
def make_wf(tmp_path):
    """Factory for fully-faked Workflows sharing ``tmp_path`` run storage."""

    def make(
        *,
        runtime: Any = None,
        budget: Budget | None = None,
        run_dir: Path | None = None,
        resume: bool = False,
        run_id: str = "test-run",
        model: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        transcripts: bool = False,
    ) -> Workflow:
        return Workflow(
            run_id=run_id,
            runtime=runtime if runtime is not None else FakeRuntime(),
            budget=budget if budget is not None else Budget(),
            journal=Journal(run_dir or tmp_path / "run", resume=resume),
            progress=Progress(run_id, force_plain=True),
            model=model,
            effort=effort,
            cwd=cwd,
            transcripts=transcripts,
        )

    return make
