"""Append-only run journal with fingerprinted resume.

Every ``agent()`` call gets a **fingerprint** — ``sha256(prompt, normalized
opts)`` — plus a per-fingerprint **occurrence number** (the Nth call with this
exact fingerprint in the run). The replay cache is keyed by
``(fingerprint, occurrence)``, which is deliberately *scheduling-independent*:
under ``parallel()`` and ``pipeline()`` the order agents *start* varies run to
run with live session latency, so a global call-order position would spuriously
diverge on resume. Content-addressed keys make an identical resumed run replay
fully no matter how the event loop interleaved the original. Records are
appended to ``.rdw/runs/<run-id>/journal.jsonl`` as they complete (each call
also carries a monotonically increasing display ``index`` for reporting).

Resume semantics (the Workflow-tool contract):

* same ``(fingerprint, occurrence)`` with an ``ok`` record → **replay** the
  cached result instantly (no session, no credits);
* an ``error`` record at a matching key → re-execute live (fingerprints match,
  so this is a retry, not a divergence);
* a cache miss while unreplayed cached records remain → the run is
  **diverged**: everything after that call runs live, a
  :class:`~rdw.errors.DivergenceWarning` is emitted, and a divergence marker
  line is appended. (A miss *after* every cached record has been consumed is
  just new work appended to the script — no divergence.)

Beyond agent records, the journal carries forensic and determinism lines:

* ``{"type": "value", "kind": "now"|"random"|"uuid", "seq": N, "value": …}`` —
  a journaled nondeterministic value (``wf.now()`` / ``wf.random()`` /
  ``wf.uuid()``), keyed by ``(kind, occurrence)`` exactly like agent records
  so timestamps and randomness replay deterministically on resume;
* ``{"type": "refusal", …}`` — an agent call refused at the budget ceiling,
  with label, key, and a budget snapshot (never replayable: a retried call
  after raising the budget runs live);
* ``{"type": "boundary", "event": "start"|"resume", "info": …}`` — one line
  per process attempt, so a journal read cold shows which records belong to
  which invocation;
* ``{"type": "divergence"}`` / ``{"type": "log"}`` — informational history.

Only ``agent`` and ``value`` lines participate in replay; everything else is
ignored by the loader's cache build.

The file is genuinely append-only and crash-tolerant: superseding is
event-sourced (a later line for the same key wins), a torn final line from a
crash mid-append is skipped with a warning instead of poisoning every future
resume, and appends repair a missing trailing newline so two records can never
merge into one corrupt line.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DivergenceWarning, JournalError, JournalWarning

JOURNAL_NAME = "journal.jsonl"

CacheKey = tuple[str, int]
"""``(fingerprint, occurrence)`` — the replay cache key."""


def fingerprint(prompt: str, opts: dict[str, Any]) -> str:
    """``sha256(prompt, normalized opts)`` — the replay cache key's hash part.

    ``opts`` should contain only result-affecting options (model, effort,
    schema hash, tool names, cwd). Cosmetic options (label, timeout) are
    deliberately excluded by the engine so tweaking them never busts the
    cache. Call position is deliberately **not** part of the fingerprint:
    agents are hermetic functions of (prompt, opts), and concurrent
    scheduling must not affect replay identity.
    """
    body = json.dumps(
        {"prompt": prompt, "opts": opts},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class AgentRecord:
    """One journaled agent call (one line of journal.jsonl)."""

    index: int
    fp: str
    label: str
    phase: str | None
    status: str  # "ok" | "error"
    seq: int = 0  # Nth call with this fingerprint (occurrence number)
    result: dict[str, Any] | None = None  # schema.dump_value payload when ok
    error: str | None = None
    session_id: str | None = None
    credits: float = 0.0
    started: float = 0.0
    ended: float = 0.0
    usage: dict[str, Any] | None = None  # token/tool telemetry (UsageTap snapshot)
    request: dict[str, Any] | None = None
    """Forensic request context (model, effort, tools, session_limits, budget
    snapshot, prompt size) — recorded for debugging, deliberately **excluded**
    from the fingerprint so replay identity never depends on it."""

    @property
    def key(self) -> CacheKey:
        return (self.fp, self.seq)

    def to_line(self) -> str:
        return json.dumps(
            {
                "type": "agent",
                "index": self.index,
                "fp": self.fp,
                "seq": self.seq,
                "label": self.label,
                "phase": self.phase,
                "status": self.status,
                "result": self.result,
                "error": self.error,
                "session_id": self.session_id,
                "credits": self.credits,
                "started": self.started,
                "ended": self.ended,
                "usage": self.usage,
                "request": self.request,
            },
            ensure_ascii=False,
            default=str,
        )

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "AgentRecord":
        return cls(
            index=int(obj["index"]),
            fp=str(obj["fp"]),
            seq=int(obj.get("seq") or 0),
            label=str(obj.get("label") or ""),
            phase=obj.get("phase"),
            status=str(obj.get("status") or "error"),
            result=obj.get("result"),
            error=obj.get("error"),
            session_id=obj.get("session_id"),
            credits=float(obj.get("credits") or 0.0),
            started=float(obj.get("started") or 0.0),
            ended=float(obj.get("ended") or 0.0),
            usage=obj.get("usage"),
            request=obj.get("request"),
        )


@dataclass
class Journal:
    """Append-only journal for one run directory.

    Args:
        run_dir: ``.rdw/runs/<run-id>``; created if missing.
        resume: When True, prior records are loaded as the replay cache.
            When False the cache starts empty (records are still appended, so
            a *later* ``--resume`` of this run id can reuse them).
    """

    run_dir: Path
    resume: bool = False
    _cache: dict[CacheKey, AgentRecord] = field(default_factory=dict, repr=False)
    _values: dict[CacheKey, Any] = field(default_factory=dict, repr=False)
    _pending: set[CacheKey] = field(default_factory=set, repr=False)
    _occurrences: dict[str, int] = field(default_factory=dict, repr=False)
    _value_occurrences: dict[str, int] = field(default_factory=dict, repr=False)
    _counter: int = field(default=0, repr=False)
    _diverged: bool = field(default=False, repr=False)
    _hits: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.resume:
            self._load()
            # Agent keys are (64-hex fingerprint, seq); value keys are
            # (kind, seq) with kind in {"now", "random", "uuid"} — the key
            # spaces cannot collide, so one pending set covers both: a miss of
            # either sort while *anything* cached remains unreplayed is a
            # divergence.
            self._pending = set(self._cache) | set(self._values)

    @property
    def path(self) -> Path:
        return self.run_dir / JOURNAL_NAME

    @property
    def cache_hits(self) -> int:
        """Number of agent calls served from the journal this run."""
        return self._hits

    @property
    def diverged(self) -> bool:
        return self._diverged

    @property
    def cache_size(self) -> int:
        """Replayable entries currently held (agent records + values)."""
        with self._lock:
            return len(self._cache) + len(self._values)

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        """Replay journal lines in order: last record per (fp, seq) wins.

        Fills ``_cache`` (agent records) and ``_values`` (journaled
        nondeterministic values) in place. A torn **final** line (crash
        mid-append — the exact failure the journal exists to recover from) is
        skipped with a :class:`~rdw.errors.JournalWarning`; corruption
        anywhere *else* in the file still raises
        :class:`~rdw.errors.JournalError`, because a damaged interior means
        the history can't be trusted.
        """
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise JournalError(f"cannot read {self.path}: {exc}") from exc
        lines = raw.split("\n")
        last = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if i == last:
                    warnings.warn(
                        f"{self.path}:{i + 1}: skipping torn final journal line "
                        f"(crash mid-append?): {exc}",
                        JournalWarning,
                        stacklevel=4,
                    )
                    continue
                raise JournalError(f"{self.path}:{i + 1}: invalid JSON: {exc}") from exc
            if obj.get("type") == "agent":
                rec = AgentRecord.from_obj(obj)
                self._cache[rec.key] = rec
            elif obj.get("type") == "value":
                key = (str(obj.get("kind")), int(obj.get("seq") or 0))
                self._values[key] = obj.get("value")
            # "refusal"/"boundary"/"divergence"/"log"/unknown lines are
            # informational history — never replayable, so a refused call
            # retried after raising the budget always runs live.

    # ---------------------------------------------------------------- lookup

    def next_index(self) -> int:
        """Allocate the next display position (monotonic, in call order).

        Display/labeling only — never part of replay identity, because call
        order under concurrency is scheduling-dependent.
        """
        with self._lock:
            index = self._counter
            self._counter += 1
            return index

    def next_occurrence(self, fp: str) -> int:
        """Allocate the occurrence number for the Nth call with fingerprint
        ``fp`` this run (0-based). Identical concurrent calls may claim their
        occurrences in either order; their results are interchangeable by
        construction (same prompt, same opts, hermetic sessions)."""
        with self._lock:
            seq = self._occurrences.get(fp, 0)
            self._occurrences[fp] = seq + 1
            return seq

    def lookup(
        self, fp: str, seq: int, *, index: int = 0, label: str = ""
    ) -> AgentRecord | None:
        """Return the replayable record for ``(fp, seq)`` or ``None``.

        Applies the contract: an ``ok`` record replays; an ``error`` record
        re-executes live (retry, not divergence); a miss while unreplayed
        cached records remain marks the run diverged (loudly, once) and
        everything after runs live.
        """
        with self._lock:
            if self._diverged:
                return None
            rec = self._cache.get((fp, seq))
            if rec is not None:
                self._pending.discard((fp, seq))
                if rec.status != "ok":
                    return None  # matching retry of a failed call — go live
                self._hits += 1
                return rec
            if self._pending:
                self._diverged = True
                self._append_line(
                    json.dumps(
                        {
                            "type": "divergence",
                            "index": index,
                            "fp": fp,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )
                warnings.warn(
                    f"journal divergence at agent {label or index!r}: no cached "
                    f"record matches this call; running live from here",
                    DivergenceWarning,
                    stacklevel=3,
                )
            return None

    # ----------------------------------------------------------------- values

    def next_value_occurrence(self, kind: str) -> int:
        """Allocate the occurrence number for the Nth ``kind`` value this run
        (0-based) — the value-channel twin of :meth:`next_occurrence`."""
        with self._lock:
            seq = self._value_occurrences.get(kind, 0)
            self._value_occurrences[kind] = seq + 1
            return seq

    def value_lookup(self, kind: str, seq: int) -> Any | None:
        """Return the cached value for ``(kind, seq)`` or ``None`` on a miss.

        Recorded values are floats (``now``/``random``) or uuid strings, never
        ``None``, so ``None`` unambiguously means "record a fresh value". A
        miss while unreplayed cached entries remain marks the run diverged —
        the same contract as :meth:`lookup`, because a shifted value stream
        shifts everything downstream of it.
        """
        with self._lock:
            if self._diverged:
                return None
            value = self._values.get((kind, seq))
            if value is not None:
                self._pending.discard((kind, seq))
                return value
            if self._pending:
                self._diverged = True
                self._append_line(
                    json.dumps(
                        {"type": "divergence", "kind": kind, "seq": seq, "ts": time.time()},
                        ensure_ascii=False,
                    )
                )
                warnings.warn(
                    f"journal divergence at value {kind}[{seq}]: no cached "
                    f"value matches this call; running live from here",
                    DivergenceWarning,
                    stacklevel=4,
                )
            return None

    def value_record(self, kind: str, seq: int, value: Any) -> None:
        """Append a freshly produced nondeterministic value for replay."""
        with self._lock:
            self._values[(kind, seq)] = value
            self._pending.discard((kind, seq))
            self._append_line(
                json.dumps(
                    {"type": "value", "kind": kind, "seq": seq, "value": value},
                    ensure_ascii=False,
                    default=str,
                )
            )

    # ---------------------------------------------------------------- append

    def _append_line(self, line: str) -> None:
        """Durably append one record line.

        If a previous crash left the file without a trailing newline, a
        newline is written first so this record can never merge into the torn
        one. Each append is flushed and fsynced — the journal is the crash
        recovery story, so a record must survive the process dying right
        after the call it describes."""
        payload = (line + "\n").encode("utf-8")
        with self.path.open("a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    payload = b"\n" + payload
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    def record(self, rec: AgentRecord) -> None:
        """Append a completed agent record and update the in-memory view."""
        with self._lock:
            self._cache[rec.key] = rec
            self._pending.discard(rec.key)
            self._append_line(rec.to_line())

    def note(self, message: str, *, phase: str | None = None) -> None:
        """Append a non-replayable log line (``wf.log``, phase transitions)."""
        with self._lock:
            self._append_line(
                json.dumps(
                    {"type": "log", "message": message, "phase": phase, "ts": time.time()},
                    ensure_ascii=False,
                )
            )

    def refusal(
        self,
        *,
        index: int,
        fp: str,
        seq: int,
        label: str,
        phase: str | None,
        budget: dict[str, Any],
    ) -> None:
        """Append a budget-refusal line: which call was refused, and the exact
        budget snapshot (total/spent/outstanding) that refused it.

        Refusals are deliberately *not* agent records: :meth:`lookup` ignores
        them, so a refused call retried under a raised budget runs live
        instead of replaying the refusal.
        """
        with self._lock:
            self._append_line(
                json.dumps(
                    {
                        "type": "refusal",
                        "index": index,
                        "fp": fp,
                        "seq": seq,
                        "label": label,
                        "phase": phase,
                        "budget": budget,
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )

    def run_boundary(self, *, event: str, info: dict[str, Any]) -> None:
        """Append a run-boundary line (``event`` is ``"start"``/``"resume"``).

        One line per process attempt, so a journal read cold shows which
        records belong to which invocation — the forensic gap that made a
        three-attempt failure look like one run. Ignored by the replay loader.
        """
        with self._lock:
            self._append_line(
                json.dumps(
                    {"type": "boundary", "event": event, "info": info},
                    ensure_ascii=False,
                    default=str,
                )
            )

    # ----------------------------------------------------------------- tools

    def records(self) -> list[AgentRecord]:
        """Current effective records, ordered by display position."""
        with self._lock:
            return sorted(self._cache.values(), key=lambda r: (r.index, r.started))


def read_journal_lines(run_dir: Path) -> list[dict[str, Any]]:
    """Raw journal lines for ``rdw show`` (tolerant of unknown types)."""
    path = Path(run_dir) / JOURNAL_NAME
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"type": "corrupt", "raw": line[:200]})
    return out
