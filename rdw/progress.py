"""Live run progress: a rich tree grouped by phase, with a plain fallback.

When stdout is a TTY and ``rich`` is importable, a ``rich.live.Live`` display
renders the run as::

    run 20260721-1  (12.4 / 100 AIU)
    ├── design
    │   ├── ● strategy-1  running · 84s · 8.2k tok · last: bash 3s ago
    │   ├── ✓ strategy-2  done (cache)
    │   └── ✗ challenge   failed
    └── implement
        └── ● implementer running · 41.0k tok

When not a TTY (CI, piped output) — or when ``rich`` is missing — every state
transition prints as one plain line instead, so logs stay grep-able and the
package imports cleanly without rich installed. Because a wedged agent is
otherwise indistinguishable from a working one for the whole ``timeout``
window (600 s by default), plain mode also runs a **heartbeat**: a single
bounded summary line every :data:`HEARTBEAT_INTERVAL` seconds while agents are
running::

    · 3 running: strategy-1 84s/8.2k last: bash 3s ago, strategy-2 80s, ...

All update methods are thread-safe: SDK event handlers fire from the client's
receive thread while the engine mutates state on the event loop. The heartbeat
task is created in :meth:`Progress.start` (needs a running loop) and cancelled
in :meth:`Progress.stop`.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

HEARTBEAT_INTERVAL = 30.0
"""Seconds between plain-mode heartbeat lines while agents are running."""

HEARTBEAT_MAX_AGENTS = 4
"""Agents listed per heartbeat line before collapsing to ``+N more`` — keeps
the line bounded no matter how wide the wave is."""

_STATUS_GLYPH = {
    "running": "●",
    "ok": "✓",
    "cached": "✓",
    "error": "✗",
    "timeout": "✗",
}


def _fmt_count(n: int) -> str:
    """``8200 -> '8.2k'`` — short token count for tight displays."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def fmt_tokens(n: int) -> str:
    """``148000 -> '148.0k tok'`` — used by the tree and the stats rollups."""
    return f"{_fmt_count(n)} tok"


@dataclass
class _AgentState:
    label: str
    phase: str | None
    status: str = "running"
    tokens: int = 0
    detail: str = ""
    started_ts: float = 0.0
    last_event: str = ""  # e.g. the tool name from tool.execution_start
    last_event_ts: float = 0.0


class Progress:
    """Run-scoped progress board.

    Args:
        run_id: Displayed at the tree root.
        budget: Optional :class:`~rdw.budget.Budget` whose live spend is shown
            in the header.
        force_plain: Force the non-TTY line renderer (used by tests/CI).
        heartbeat: Seconds between plain-mode liveness lines while agents are
            running (``None`` disables the heartbeat entirely).
    """

    def __init__(
        self,
        run_id: str = "run",
        *,
        budget: Any = None,
        force_plain: bool | None = None,
        heartbeat: float | None = HEARTBEAT_INTERVAL,
    ) -> None:
        self.run_id = run_id
        self.budget = budget
        self._lock = threading.Lock()
        self._phases: list[str] = []
        self._agents: dict[str, _AgentState] = {}
        self._live: Any = None
        self._heartbeat_interval = heartbeat
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._now = time.time  # test seam: swap for a fake clock
        if force_plain is not None:
            self._plain = force_plain
        else:
            self._plain = not sys.stdout.isatty()
        if not self._plain:
            try:
                import rich.live  # noqa: F401
            except ImportError:
                self._plain = True

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Begin rendering; in plain mode this starts the heartbeat task."""
        if self._plain:
            self._start_heartbeat()
            return
        if self._live is not None:
            return
        from rich.live import Live

        self._live = Live(
            get_renderable=self._render,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display (and heartbeat), leaving the final tree on
        screen."""
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None:
            task.cancel()
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    def _start_heartbeat(self) -> None:
        """Spawn the heartbeat task if enabled and a loop is running.

        Constructed lazily because ``Progress`` may be built outside any event
        loop (CLI argument-parsing time); without a running loop the heartbeat
        is simply off — state transitions still print as discrete lines.
        """
        if self._heartbeat_interval is None or self._heartbeat_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval or HEARTBEAT_INTERVAL)
                line = self.heartbeat_line()
                if line:
                    print(line, flush=True)
        except asyncio.CancelledError:
            pass

    # ---------------------------------------------------------------- updates

    def declare_phases(self, titles: list[str]) -> None:
        """Pre-register the run's phases so they render as pending branches.

        Called once up front (from ``PHASES = [...]`` in a script or
        ``wf.declare_phases``); the rich tree shows every declared phase
        immediately — empty ones read as "not started yet". Plain mode prints
        a single summary line instead of one line per phase.
        """
        with self._lock:
            for title in titles:
                if title not in self._phases:
                    self._phases.append(title)
        if titles:
            self._line("── phases: " + " → ".join(str(t) for t in titles))

    def phase_started(self, title: str) -> None:
        with self._lock:
            if title not in self._phases:
                self._phases.append(title)
        self._line(f"── phase: {title}")

    def agent_started(self, label: str, phase: str | None) -> None:
        with self._lock:
            self._agents[label] = _AgentState(
                label=label, phase=phase, started_ts=self._now()
            )
            if phase and phase not in self._phases:
                self._phases.append(phase)
        self._line(f"→ {self._loc(phase)}{label} started")

    def agent_tokens(self, label: str, tokens: int) -> None:
        """Add ``tokens`` output tokens to an agent's running counter."""
        with self._lock:
            state = self._agents.get(label)
            if state:
                state.tokens += tokens
        # deliberately silent in plain mode — token ticks would flood logs

    def agent_activity(self, label: str, event: str) -> None:
        """Note live session activity (e.g. a tool starting) for an agent.

        Fed by the engine's progress tap on ``tool.execution_start`` events.
        Silent per-call — the heartbeat and the rich tree summarize it as
        ``last: <event> <n>s ago`` so a hung agent becomes visible without
        flooding plain logs with per-tool lines.
        """
        with self._lock:
            state = self._agents.get(label)
            if state:
                state.last_event = event
                state.last_event_ts = self._now()

    def agent_finished(self, label: str, status: str, detail: str = "") -> None:
        with self._lock:
            state = self._agents.get(label)
            if state is None:
                state = _AgentState(label=label, phase=None)
                self._agents[label] = state
            state.status = status
            state.detail = detail
        glyph = _STATUS_GLYPH.get(status, "•")
        phase = state.phase
        suffix = f" ({detail})" if detail else ""
        self._line(f"{glyph} {self._loc(phase)}{label} {status}{suffix}")

    def log(self, message: str) -> None:
        self._line(f"· {message}")

    # -------------------------------------------------------------- rendering

    @staticmethod
    def _loc(phase: str | None) -> str:
        return f"[{phase}] " if phase else ""

    def _line(self, text: str) -> None:
        if self._plain:
            print(text, flush=True)
        # In rich mode the Live display re-renders from state; discrete lines
        # would fight the repaint.

    def heartbeat_line(self, max_agents: int = HEARTBEAT_MAX_AGENTS) -> str:
        """One bounded liveness summary, or ``""`` when nothing is running.

        Example: ``· 2 running: strategy-1 84s/8.2k last: bash 3s ago, docs 12s``
        """
        now = self._now()
        with self._lock:
            running = [s for s in self._agents.values() if s.status == "running"]
            bits: list[str] = []
            for state in running[:max_agents]:
                piece = f"{state.label} {max(0.0, now - state.started_ts):.0f}s"
                if state.tokens:
                    piece += f"/{_fmt_count(state.tokens)}"
                if state.last_event:
                    ago = max(0.0, now - state.last_event_ts)
                    piece += f" last: {state.last_event} {ago:.0f}s ago"
                bits.append(piece)
        if not running:
            return ""
        if len(running) > max_agents:
            bits.append(f"+{len(running) - max_agents} more")
        return f"· {len(running)} running: " + ", ".join(bits)

    def _header(self) -> str:
        if self.budget is not None:
            try:
                return f"{self.run_id}  ({self.budget.summary()})"
            except Exception:
                pass
        return self.run_id

    def _render(self) -> Any:
        from rich.tree import Tree

        now = self._now()
        with self._lock:
            tree = Tree(self._header())
            by_phase: dict[str | None, list[_AgentState]] = {}
            for state in self._agents.values():
                by_phase.setdefault(state.phase, []).append(state)
            ordered: list[str | None] = list(self._phases)
            if None in by_phase:
                ordered.append(None)
            for phase in ordered:
                agents = by_phase.get(phase, [])
                node = tree.add(phase or "(no phase)") if phase or agents else None
                if node is None:
                    continue
                for state in agents:
                    glyph = _STATUS_GLYPH.get(state.status, "•")
                    bits = [f"{glyph} {state.label}", state.status]
                    if state.status == "running" and state.started_ts:
                        bits.append(f"{max(0.0, now - state.started_ts):.0f}s")
                    if state.tokens:
                        bits.append(fmt_tokens(state.tokens))
                    if state.status == "running" and state.last_event:
                        ago = max(0.0, now - state.last_event_ts)
                        bits.append(f"last: {state.last_event} {ago:.0f}s ago")
                    if state.detail:
                        bits.append(state.detail)
                    node.add("  ".join(bits))
            return tree
