"""Live run progress: a rich tree grouped by phase, with a plain fallback.

When stdout is a TTY and ``rich`` is importable, a ``rich.live.Live`` display
renders the run as::

    run 20260721-1  (12.4 / 100 AIU)
    ├── design
    │   ├── ● strategy-1  running · 8.2k tok
    │   ├── ✓ strategy-2  done (cache)
    │   └── ✗ challenge   failed
    └── implement
        └── ● implementer running · 41.0k tok

When not a TTY (CI, piped output) — or when ``rich`` is missing — every state
transition prints as one plain line instead, so logs stay grep-able and the
package imports cleanly without rich installed.

All update methods are thread-safe: SDK event handlers fire from the client's
receive thread while the engine mutates state on the event loop.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any

_STATUS_GLYPH = {
    "running": "●",
    "ok": "✓",
    "cached": "✓",
    "error": "✗",
    "timeout": "✗",
}


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M tok"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k tok"
    return f"{n} tok"


@dataclass
class _AgentState:
    label: str
    phase: str | None
    status: str = "running"
    tokens: int = 0
    detail: str = ""


class Progress:
    """Run-scoped progress board.

    Args:
        run_id: Displayed at the tree root.
        budget: Optional :class:`~rdw.budget.Budget` whose live spend is shown
            in the header.
        force_plain: Force the non-TTY line renderer (used by tests/CI).
    """

    def __init__(
        self,
        run_id: str = "run",
        *,
        budget: Any = None,
        force_plain: bool | None = None,
    ) -> None:
        self.run_id = run_id
        self.budget = budget
        self._lock = threading.Lock()
        self._phases: list[str] = []
        self._agents: dict[str, _AgentState] = {}
        self._live: Any = None
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
        """Begin rendering (no-op in plain mode)."""
        if self._plain or self._live is not None:
            return
        from rich.live import Live

        self._live = Live(
            get_renderable=self._render,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display, leaving the final tree on screen."""
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    # ---------------------------------------------------------------- updates

    def phase_started(self, title: str) -> None:
        with self._lock:
            if title not in self._phases:
                self._phases.append(title)
        self._line(f"── phase: {title}")

    def agent_started(self, label: str, phase: str | None) -> None:
        with self._lock:
            self._agents[label] = _AgentState(label=label, phase=phase)
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

    def _header(self) -> str:
        if self.budget is not None:
            try:
                return f"{self.run_id}  ({self.budget.summary()})"
            except Exception:
                pass
        return self.run_id

    def _render(self) -> Any:
        from rich.tree import Tree

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
                    if state.tokens:
                        bits.append(_fmt_tokens(state.tokens))
                    if state.detail:
                        bits.append(state.detail)
                    node.add("  ".join(bits))
            return tree
