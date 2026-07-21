"""Opt-in observability taps: per-agent transcripts and token/tool telemetry.

Two small subscribers that ride *alongside* the engine's budget and progress
taps on every live session (``session.on(handler)``). Neither touches credit
accounting — :class:`~rdw.budget.Budget` remains the single source of truth
for spend; these taps only observe.

* :class:`TranscriptWriter` — appends a filtered stream of session events to
  ``<run-dir>/agents/<index>-<label>.jsonl`` when the workflow is opened with
  ``transcripts=True`` (or ``rdw run --transcripts``). Filtering happens *at
  the tap*, deliberately: raw session ``events.jsonl`` logs grow to 50–150 MB
  on long sessions and interleave base64 ``session.binary_asset`` blobs, so
  only the five turn-forensics event types are kept and streaming deltas are
  dropped. Tool arguments are truncated so one giant ``bash`` heredoc cannot
  bloat the transcript.
* :class:`UsageTap` — accumulates per-agent token and tool-call counters from
  the same event stream (``assistant.usage`` carries the token splits;
  ``tool.execution_start`` marks each tool call). The engine snapshots this
  onto :class:`~rdw.journal.AgentRecord.usage` so ``wf.report()`` and
  ``rdw show --stats`` can roll telemetry up per phase.

Handler discipline (docs/architecture.md): SDK event handlers may fire on the
client's receive thread, so both taps are synchronous, lock-guarded, and
swallow *all* of their own exceptions — a full disk or malformed event must
never raise into the SDK's dispatch loop.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, TextIO

TRANSCRIPT_DIR = "agents"
"""Subdirectory of the run dir where per-agent transcripts land."""

TRANSCRIPT_EVENT_TYPES = frozenset(
    {
        "assistant.message",
        "tool.execution_start",
        "tool.execution_complete",
        "assistant.usage",
        "session.error",
    }
)
"""The turn-forensics events a transcript keeps. Everything else — streaming
``assistant.message_delta`` ticks, ``session.binary_asset`` blobs, lifecycle
chatter — is dropped at the tap."""

MAX_TOOL_ARG_CHARS = 2000
"""Truncation limit for serialized tool arguments/results in transcripts."""


def _event_type(event: Any) -> str:
    """Extract the string event type from an SDK SessionEvent (or a fake)."""
    etype = getattr(event, "type", None)
    value = getattr(etype, "value", etype)
    return value if isinstance(value, str) else ""


def _truncate(text: str, limit: int = MAX_TOOL_ARG_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit} chars]"


def _compact_json(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return _truncate(text)


def _compact(etype: str, data: Any) -> dict[str, Any]:
    """Reduce an event's data payload to the fields worth keeping on disk.

    Field names verified against the installed SDK's generated event types
    (``AssistantMessageData``, ``ToolExecutionStartData``,
    ``ToolExecutionCompleteData``, ``AssistantUsageData``,
    ``SessionErrorData``). Everything is ``getattr``-tolerant so a fake or a
    future SDK shape degrades to partial data, never an exception.
    """
    if etype == "assistant.message":
        return {"content": getattr(data, "content", None)}
    if etype == "tool.execution_start":
        return {
            "tool_name": getattr(data, "tool_name", None),
            "arguments": _compact_json(getattr(data, "arguments", None)),
        }
    if etype == "tool.execution_complete":
        out: dict[str, Any] = {
            "tool_call_id": getattr(data, "tool_call_id", None),
            "success": getattr(data, "success", None),
        }
        error = getattr(data, "error", None)
        if error is not None:
            out["error"] = _truncate(str(getattr(error, "message", error)))
        result = getattr(data, "result", None)
        if result is not None:
            out["result"] = _truncate(str(getattr(result, "content", result)))
        return out
    if etype == "assistant.usage":
        out = {"model": getattr(data, "model", None)}
        for field in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens"):
            value = getattr(data, field, None)
            if isinstance(value, int):
                out[field] = value
        nano = getattr(getattr(data, "copilot_usage", None), "total_nano_aiu", None)
        if nano is not None:
            out["total_nano_aiu"] = nano
        return out
    if etype == "session.error":
        return {
            "error_type": getattr(data, "error_type", None),
            "message": getattr(data, "message", None),
        }
    return {}


def transcript_filename(index: int, label: str) -> str:
    """``000-strategy_1.jsonl`` — index keeps call order, label keeps identity.

    The label is sanitized because it is user-supplied and becomes a path
    segment; anything outside ``[\\w.-]`` would otherwise create directories
    or escape the run dir.
    """
    safe = re.sub(r"[^\w.-]", "_", label) or "agent"
    return f"{index:03d}-{safe}.jsonl"


class TranscriptWriter:
    """Append-only JSONL transcript for one agent session.

    The file (and its parent directory) is created lazily on the first kept
    event, so an agent that emits nothing transcript-worthy leaves no file
    behind — and ``transcripts=False`` runs never even construct one of these.

    Args:
        path: Destination, conventionally
            ``<run-dir>/agents/<index>-<label>.jsonl``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fh: TextIO | None = None

    def tap(self) -> Callable[[Any], None]:
        """Build a ``session.on()`` handler that writes kept events.

        The handler swallows every exception it can raise (I/O errors
        included): the transcript is best-effort forensics and must never
        break event delivery to the budget/progress taps.
        """

        def handler(event: Any) -> None:
            try:
                etype = _event_type(event)
                if etype not in TRANSCRIPT_EVENT_TYPES:
                    return
                line = json.dumps(
                    {
                        "ts": time.time(),
                        "type": etype,
                        "data": _compact(etype, getattr(event, "data", None)),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                with self._lock:
                    if self._fh is None:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        self._fh = self.path.open("a", encoding="utf-8")
                    self._fh.write(line + "\n")
                    self._fh.flush()
            except Exception:
                pass  # best-effort: never raise into the SDK dispatch loop

        return handler

    def close(self) -> None:
        """Close the file if it was ever opened. Idempotent."""
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is not None:
            with contextlib.suppress(Exception):
                fh.close()


class UsageTap:
    """Per-agent token/tool-call accumulator (read-only sibling of Budget.tap).

    Counters, all summed over the session's lifetime:

    * ``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` — from
      ``assistant.usage`` events (a key is present only once observed);
    * ``model_calls`` — one per ``assistant.usage`` event (one per model step);
    * ``tool_calls`` — one per ``tool.execution_start``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def tap(self) -> Callable[[Any], None]:
        """Build a ``session.on()`` handler feeding the counters."""

        def handler(event: Any) -> None:
            try:
                etype = _event_type(event)
                if etype == "assistant.usage":
                    data = getattr(event, "data", None)
                    with self._lock:
                        self._counts["model_calls"] = self._counts.get("model_calls", 0) + 1
                        for field in ("input_tokens", "output_tokens", "cache_read_tokens"):
                            value = getattr(data, field, None)
                            if isinstance(value, int) and value:
                                self._counts[field] = self._counts.get(field, 0) + value
                elif etype == "tool.execution_start":
                    with self._lock:
                        self._counts["tool_calls"] = self._counts.get("tool_calls", 0) + 1
            except Exception:
                pass  # telemetry must never break event delivery

        return handler

    def snapshot(self) -> dict[str, int] | None:
        """Current totals, or ``None`` when nothing was observed — so journal
        lines for sessions without telemetry stay byte-identical to v0.1."""
        with self._lock:
            return dict(self._counts) if self._counts else None
