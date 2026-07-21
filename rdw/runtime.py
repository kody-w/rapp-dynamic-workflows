"""Runtime layer: a lazy, shared CopilotClient behind a small protocol.

The github-copilot-sdk multiplexes many concurrent sessions over one spawned
``copilot --headless --stdio`` child process (``CopilotClient._sessions``), so
the cheapest fan-out architecture is exactly one client per workflow run.
:class:`CopilotRuntime` wraps that client with:

* **lazy start** — the node child process only spawns on the first
  ``create_session`` call (a resumed run that serves every result from the
  journal never pays the startup cost);
* **a concurrency semaphore** — ``agent()`` acquires a slot before creating a
  session, capping simultaneous live sessions at
  ``min(16, cpu_count - 2)`` by default;
* **clean shutdown** — ``close()`` stops the client (and its child process)
  exactly once.

Everything above this module talks to the :class:`Runtime` protocol, so tests
(and CI, which must never run live inference) substitute a fake by subclassing
:class:`BaseRuntime` and implementing ``create_session``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable


def default_concurrency() -> int:
    """Default cap on simultaneous live agent sessions: ``min(16, cpu-2)``, >= 1."""
    cpus = os.cpu_count() or 4
    return max(1, min(16, cpus - 2))


@runtime_checkable
class SessionHandle(Protocol):
    """The slice of ``copilot.CopilotSession`` the engine actually uses.

    Verified against the installed SDK (session.py): ``session_id`` attribute,
    ``send_and_wait(prompt, *, timeout=...)`` (waits for ``session.idle``),
    ``on(handler) -> unsubscribe``, ``abort()``, ``disconnect()``.
    """

    session_id: str

    async def send_and_wait(self, prompt: str, *, timeout: float = 60.0) -> Any: ...

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]: ...

    async def abort(self) -> None: ...

    async def disconnect(self) -> None: ...


class Runtime(Protocol):
    """What the engine needs from a session factory."""

    async def create_session(self, **kwargs: Any) -> SessionHandle: ...

    def slot(self) -> "contextlib.AbstractAsyncContextManager[None]": ...

    async def close(self) -> None: ...


class BaseRuntime:
    """Shared semaphore machinery for real and fake runtimes.

    Subclasses implement :meth:`create_session`. The semaphore is created
    lazily so a runtime can be constructed outside an event loop (e.g. at CLI
    argument-parsing time).
    """

    def __init__(self, concurrency: int | None = None) -> None:
        self.concurrency = concurrency or default_concurrency()
        self._sem: asyncio.Semaphore | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        return self._sem

    @contextlib.asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        sem = self._semaphore()
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def slot(self) -> contextlib.AbstractAsyncContextManager[None]:
        """Acquire one concurrency slot for the duration of a live agent."""
        return self._slot()

    async def create_session(self, **kwargs: Any) -> SessionHandle:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:
        """Default shutdown is a no-op; real runtimes stop their client."""


class CopilotRuntime(BaseRuntime):
    """Production runtime: one lazy singleton ``CopilotClient`` per instance.

    The ``copilot`` package is imported lazily so that ``import rdw`` works in
    environments where only fakes are exercised, and so no child process is
    spawned until a live session is genuinely needed.

    Args:
        working_directory: Default working directory for the runtime process.
        base_directory: Overrides ``COPILOT_HOME`` for isolation (optional).
        concurrency: Max simultaneous live sessions (default ``min(16, cpu-2)``).
        client_factory: Test seam — a callable returning a client object with
            ``start()``, ``create_session(**kw)`` and ``stop()`` coroutines.
    """

    def __init__(
        self,
        *,
        working_directory: str | None = None,
        base_directory: str | None = None,
        concurrency: int | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(concurrency)
        self.working_directory = working_directory
        self.base_directory = base_directory
        self._client_factory = client_factory
        self._client: Any = None
        self._client_lock = asyncio.Lock()
        self._closed = False

    def _default_factory(self) -> Any:
        from copilot import CopilotClient  # lazy: spawns nothing until start()

        return CopilotClient(
            working_directory=self.working_directory,
            base_directory=self.base_directory,
        )

    async def _ensure_client(self) -> Any:
        if self._closed:
            raise RuntimeError("CopilotRuntime is closed")
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    factory = self._client_factory or self._default_factory
                    client = factory()
                    await client.start()
                    self._client = client
        return self._client

    async def create_session(self, **kwargs: Any) -> SessionHandle:
        """Create one hermetic session, defaulting permissions to approve-all.

        Headless runs otherwise fail their first tool call with
        ``denied-no-approval-rule-and-could-not-request-from-user`` (observed
        in production logs), so the policy is installed *before* the first
        turn. Callers can pass their own ``on_permission_request`` to narrow.
        """
        client = await self._ensure_client()
        if "on_permission_request" not in kwargs:
            from copilot import PermissionHandler

            kwargs["on_permission_request"] = PermissionHandler.approve_all
        return await client.create_session(**kwargs)

    async def close(self) -> None:
        """Stop the shared client (and its ``copilot`` child process) once."""
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.stop()
