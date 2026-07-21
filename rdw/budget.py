"""AI-credit budget accounting and enforcement.

Copilot bills sessions in AI credits (AIU); the runtime reports spend through
two session-event channels (verified against the installed SDK's generated
event types):

* ``assistant.usage`` — per model call; ``data.copilot_usage.total_nano_aiu``
  is that call's cost in nano-AIU (1 AIU = 1e9 nano-AIU).
* ``session.usage_checkpoint`` — ``data.total_nano_aiu`` is the session's
  *cumulative* spend, emitted at idle.

A :class:`Budget` taps both per session and treats a session's spend as
``max(latest checkpoint, sum of per-call deltas)`` — robust to either channel
being missing or lagging, and never double-counting.

Enforcement is two-tier, and **concurrency-aware**:

1. **Admission control with reservations** (always): before each live agent
   the engine calls :meth:`Budget.reserve`, which considers not just spend
   already reported by usage events but every *outstanding* reservation held
   by an in-flight session. Past the ceiling — or when outstanding grants have
   committed the ceiling — new ``agent()`` calls raise
   :class:`~rdw.errors.BudgetExceeded`.
2. **Defense in depth** (when a ceiling is set): each admitted session is
   created with ``session_limits={"max_ai_credits": grant}`` where the grant
   is **half of the uncommitted remainder** at admission time. Grants shrink
   geometrically as concurrent sessions stack up, so the caps handed to any
   number of simultaneous sessions always sum to less than the remaining
   budget — the worst case is bounded by ``total``, never
   ``total + N × remaining``. (The SDK marks ``session_limits`` Experimental;
   admission control does not depend on it.)

Accounting granularity is honest: usage events arrive *after* a model step
completes, so one in-flight step can overshoot its grant before the gate sees
it. Hard real-time cost guarantees are impossible at this layer.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import BudgetExceeded

NANO_PER_CREDIT = 1_000_000_000.0

MIN_ADMISSION = 0.01
"""Uncommitted credits below which admission refuses new sessions outright
(prevents an endless tail of dust-capped sessions near the ceiling)."""

MIN_GRANT = 0.005
"""Floor for a session's ``max_ai_credits`` so the limit stays positive for a
session admitted right at the boundary."""

PROVIDER_MIN_SESSION_LIMIT = 30.0
"""The Copilot API rejects ``session.create`` outright when
``session_limits.max_ai_credits`` is below 30 ("Minimum session limit is 30
AI credits", observed live 2026-07). Grants below this floor are clamped UP
to it: a 30-credit provider-side leash beats an uncapped session. (Observed
live: with sub-floor caps omitted entirely, a 3-agent wave admitted against
a 40-credit budget spent 267 credits — admission gates alone cannot stop
sessions that are already running.) Consequence: a capped run's worst-case
overshoot is bounded by ~30 credits per concurrently admitted session."""


def _event_type(event: Any) -> str:
    """Extract the string event type from an SDK SessionEvent (or a fake)."""
    etype = getattr(event, "type", None)
    value = getattr(etype, "value", etype)
    return value if isinstance(value, str) else ""


@dataclass
class _SessionSpend:
    """Per-session accumulator (nano-AIU)."""

    checkpoint: float = 0.0  # highest cumulative checkpoint seen
    usage_sum: float = 0.0  # sum of per-call deltas

    @property
    def nano(self) -> float:
        return max(self.checkpoint, self.usage_sum)


@dataclass
class Reservation:
    """One in-flight session's admission grant.

    Attributes:
        granted: The session's credit cap (AIU), or ``None`` when the budget
            is unlimited.
    """

    granted: float | None
    _budget: "Budget" = field(repr=False)
    _key: int = field(repr=False)
    _released: bool = field(default=False, repr=False)

    def limits(self) -> dict[str, float] | None:
        """``session_limits`` kwargs for this session (``None`` = no cap sent).

        Returns ``None`` both for unlimited budgets and for grants below the
        provider's 30-credit floor, which the API would reject.
        """
        if self.granted is None:
            return None
        grant = max(self.granted, MIN_GRANT, PROVIDER_MIN_SESSION_LIMIT)
        return {"max_ai_credits": grant}

    def release(self) -> None:
        """Return the unspent grant to the pool (session finished; its actual
        spend is now tracked by the usage-event accounting). Idempotent."""
        if self._released:
            return
        self._released = True
        self._budget._release(self._key)


@dataclass
class Budget:
    """A hard AI-credit ceiling for one workflow run.

    Args:
        total: The ceiling in AI credits (AIU). ``None`` means unlimited —
            accounting still runs, enforcement is disabled.

    The object is thread-safe: SDK event handlers may fire from the client's
    receive thread while the engine reads totals on the event loop.
    """

    total: float | None = None
    _sessions: dict[str, _SessionSpend] = field(default_factory=dict, repr=False)
    _reservations: dict[int, float] = field(default_factory=dict, repr=False)
    _res_ids: Any = field(default_factory=lambda: itertools.count(), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def spent(self) -> float:
        """Credits (AIU) spent so far across every session this run started."""
        with self._lock:
            return self._spent_locked()

    def _spent_locked(self) -> float:
        return sum(s.nano for s in self._sessions.values()) / NANO_PER_CREDIT

    def outstanding(self) -> float:
        """Credits currently reserved by in-flight (unfinished) sessions."""
        with self._lock:
            return sum(self._reservations.values())

    def remaining(self) -> float | None:
        """Credits left under the ceiling, or ``None`` when unlimited.

        Reflects reported spend only; in-flight reservations are accounted by
        the admission gate, not here."""
        if self.total is None:
            return None
        return max(0.0, self.total - self.spent())

    def session_spent(self, session_id: str) -> float:
        """Credits attributed to a single session (== one agent)."""
        with self._lock:
            spend = self._sessions.get(session_id)
            return (spend.nano / NANO_PER_CREDIT) if spend else 0.0

    def _available_locked(self) -> float | None:
        """Uncommitted credits: ceiling minus spend minus outstanding grants."""
        if self.total is None:
            return None
        return self.total - self._spent_locked() - sum(self._reservations.values())

    def ensure_available(self, *, label: str | None = None) -> None:
        """Admission gate: raise :class:`BudgetExceeded` when the ceiling is
        spent *or* fully committed to in-flight sessions.

        Called by the engine before every live agent start (a cheap fail-fast
        ahead of :meth:`reserve`). Replayed (journal-cached) results bypass
        this — they cost nothing.
        """
        if self.total is None:
            return
        with self._lock:
            available = self._available_locked()
            spent = self._spent_locked()
        assert available is not None
        if available < MIN_ADMISSION:
            raise BudgetExceeded(spent, self.total, label=label)

    def reserve(self, *, label: str | None = None) -> Reservation:
        """Admit one session, granting it a credit cap from the uncommitted
        remainder.

        The grant is half of what is neither spent nor already granted to
        other in-flight sessions, so concurrent grants shrink geometrically
        and always sum to less than the remaining budget — the run's
        worst-case spend is bounded by ``total`` even if every session
        exhausts its cap. Raises :class:`BudgetExceeded` when the uncommitted
        remainder is (nearly) gone. Callers must :meth:`Reservation.release`
        when the session finishes.
        """
        with self._lock:
            if self.total is None:
                key = next(self._res_ids)
                self._reservations[key] = 0.0
                return Reservation(granted=None, _budget=self, _key=key)
            available = self._available_locked()
            assert available is not None
            if available < MIN_ADMISSION:
                raise BudgetExceeded(self._spent_locked(), self.total, label=label)
            granted = available / 2.0
            key = next(self._res_ids)
            self._reservations[key] = granted
            return Reservation(granted=granted, _budget=self, _key=key)

    def _release(self, key: int) -> None:
        with self._lock:
            self._reservations.pop(key, None)

    def session_limits(self) -> dict[str, float] | None:
        """Snapshot ``session_limits`` for the *whole remaining* budget.

        Informational/back-compat API: the engine caps each session via
        :meth:`reserve` (which divides the remainder among concurrent
        sessions) rather than handing every session the full remainder.
        """
        remaining = self.remaining()
        if remaining is None:
            return None
        grant = max(remaining, MIN_GRANT * 2, PROVIDER_MIN_SESSION_LIMIT)
        return {"max_ai_credits": grant}

    def tap(self, session_id: str) -> Callable[[Any], None]:
        """Build a ``session.on()`` handler that accounts this session's spend.

        The handler is synchronous, cheap, and exception-safe (a malformed
        event must never kill the SDK's dispatch loop).
        """
        with self._lock:
            spend = self._sessions.setdefault(session_id, _SessionSpend())

        def handler(event: Any) -> None:
            try:
                etype = _event_type(event)
                data = getattr(event, "data", None)
                if etype == "assistant.usage":
                    usage = getattr(data, "copilot_usage", None)
                    nano = getattr(usage, "total_nano_aiu", None)
                    if nano:
                        with self._lock:
                            spend.usage_sum += float(nano)
                elif etype == "session.usage_checkpoint":
                    nano = getattr(data, "total_nano_aiu", None)
                    if nano:
                        with self._lock:
                            spend.checkpoint = max(spend.checkpoint, float(nano))
            except Exception:
                # Accounting must never break event delivery.
                pass

        return handler

    def summary(self) -> str:
        """One-line human summary for run-end reporting."""
        spent = self.spent()
        if self.total is None:
            return f"{spent:.2f} AI credits spent (no ceiling)"
        return f"{spent:.2f} / {self.total:.2f} AI credits spent"
