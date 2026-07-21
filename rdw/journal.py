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
    _pending: set[CacheKey] = field(default_factory=set, repr=False)
    _occurrences: dict[str, int] = field(default_factory=dict, repr=False)
    _counter: int = field(default=0, repr=False)
    _diverged: bool = field(default=False, repr=False)
    _hits: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.resume:
            self._cache = self._load()
            self._pending = set(self._cache)

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

    # ------------------------------------------------------------------ load

    def _load(self) -> dict[CacheKey, AgentRecord]:
        """Replay journal lines in order: last record per (fp, seq) wins.

        A torn **final** line (crash mid-append — the exact failure the
        journal exists to recover from) is skipped with a
        :class:`~rdw.errors.JournalWarning`; corruption anywhere *else* in
        the file still raises :class:`~rdw.errors.JournalError`, because a
        damaged interior means the history can't be trusted.
        """
        cache: dict[CacheKey, AgentRecord] = {}
        if not self.path.exists():
            return cache
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
                cache[rec.key] = rec
            # "divergence"/"log"/unknown lines are informational history.
        return cache

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
