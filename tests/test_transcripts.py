"""Opt-in per-agent transcripts: filtered tap, engine wiring, CLI flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdw import cli
from rdw.transcripts import (
    MAX_TOOL_ARG_CHARS,
    TranscriptWriter,
    transcript_filename,
)

from conftest import FakeRuntime, Turn, event


def _read_types(path: Path) -> list[str]:
    return [json.loads(line)["type"] for line in path.read_text().splitlines()]


# ------------------------------------------------------------- writer unit


def test_writer_keeps_only_transcript_event_types(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TranscriptWriter(path)
    tap = writer.tap()
    tap(event("assistant.message", content="hello", message_id="m1"))
    tap(event("assistant.message_delta", content="h"))  # streaming: dropped
    tap(event("tool.execution_start", tool_name="bash", arguments={"cmd": "ls"}))
    tap(event("tool.execution_complete", tool_call_id="c1", success=True))
    tap(event("assistant.usage", model="m", input_tokens=10, output_tokens=5))
    tap(event("session.binary_asset", blob="x" * 100))  # blobs: dropped
    tap(event("session.error", error_type="boom", message="it broke"))
    writer.close()
    assert _read_types(path) == [
        "assistant.message",
        "tool.execution_start",
        "tool.execution_complete",
        "assistant.usage",
        "session.error",
    ]


def test_writer_compacts_payloads(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TranscriptWriter(path)
    tap = writer.tap()
    tap(event("assistant.message", content="hi", message_id="m1"))
    tap(event("assistant.usage", model="gpt-x", input_tokens=100, output_tokens=40))
    writer.close()
    msg, usage = [json.loads(line) for line in path.read_text().splitlines()]
    assert msg["data"] == {"content": "hi"}
    assert usage["data"]["model"] == "gpt-x"
    assert usage["data"]["input_tokens"] == 100
    assert usage["data"]["output_tokens"] == 40
    assert "ts" in msg


def test_tool_arguments_truncated(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TranscriptWriter(path)
    writer.tap()(event("tool.execution_start", tool_name="bash", arguments={"cmd": "x" * 5000}))
    writer.close()
    [line] = path.read_text().splitlines()
    arguments = json.loads(line)["data"]["arguments"]
    assert len(arguments) < MAX_TOOL_ARG_CHARS + 30  # payload + truncation marker
    assert "…[+" in arguments


def test_no_file_until_first_kept_event(tmp_path):
    path = tmp_path / "agents" / "t.jsonl"
    writer = TranscriptWriter(path)
    writer.tap()(event("assistant.message_delta", content="h"))
    writer.close()
    assert not path.exists()
    assert not path.parent.exists()  # even the directory is lazy


def test_writer_never_raises_into_dispatch(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory must go")
    writer = TranscriptWriter(blocker / "sub" / "t.jsonl")  # mkdir will fail
    tap = writer.tap()
    tap(event("assistant.message", content="hi", message_id="m1"))  # must not raise
    tap(object())  # garbage event: must not raise either
    writer.close()


def test_transcript_filename_sanitizes_label():
    assert transcript_filename(0, "strategy-1") == "000-strategy-1.jsonl"
    assert transcript_filename(12, "re/view agent") == "012-re_view_agent.jsonl"
    assert transcript_filename(3, "") == "003-agent.jsonl"


# ---------------------------------------------------------- engine wiring


@pytest.mark.asyncio
async def test_engine_writes_transcript_per_agent(make_wf):
    rt = FakeRuntime(
        [
            [
                Turn(
                    text="done",
                    events=[
                        event("assistant.message", content="working", message_id="m1"),
                        event("assistant.message_delta", content="w"),
                        event("tool.execution_start", tool_name="bash", arguments={}),
                    ],
                )
            ]
        ]
    )
    async with make_wf(runtime=rt, transcripts=True) as wf:
        await wf.agent("do it", label="writer")
    path = wf.journal.run_dir / "agents" / "000-writer.jsonl"
    assert path.exists()
    assert _read_types(path) == ["assistant.message", "tool.execution_start"]


@pytest.mark.asyncio
async def test_transcripts_off_by_default(make_wf):
    rt = FakeRuntime(
        [[Turn(text="done", events=[event("assistant.message", content="x", message_id="m")])]]
    )
    async with make_wf(runtime=rt) as wf:
        await wf.agent("do it", label="quiet")
    assert not (wf.journal.run_dir / "agents").exists()


@pytest.mark.asyncio
async def test_transcript_path_recorded_in_request(make_wf):
    rt = FakeRuntime([[Turn(text="ok")]])
    async with make_wf(runtime=rt, transcripts=True) as wf:
        await wf.agent("do it", label="tracked")
    [record] = wf.journal.records()
    assert record.request is not None
    assert record.request["transcript"] == "agents/000-tracked.jsonl"


# ------------------------------------------------------------- CLI plumbing

TRANSCRIPT_SCRIPT = '''
from types import SimpleNamespace

from rdw.runtime import BaseRuntime


def _ev(etype, **data):
    return SimpleNamespace(type=SimpleNamespace(value=etype), data=SimpleNamespace(**data))


class _Session:
    session_id = "cli-transcript-1"

    def __init__(self):
        self.handlers = []

    def on(self, handler):
        self.handlers.append(handler)
        return lambda: None

    async def send_and_wait(self, prompt, *, timeout=60.0):
        for handler in list(self.handlers):
            handler(_ev("assistant.message", content="hi", message_id="m1"))
            handler(_ev("assistant.message_delta", content="h"))
        return SimpleNamespace(data=SimpleNamespace(content="done"))

    async def abort(self):
        pass

    async def disconnect(self):
        pass


class _FakeRuntime(BaseRuntime):
    def __init__(self):
        super().__init__(2)

    async def create_session(self, **kwargs):
        return _Session()


async def workflow(wf):
    wf.runtime = _FakeRuntime()  # test seam: never touch the real client
    await wf.agent("alpha", label="writer")
'''


def test_cli_transcripts_flag_writes_agent_files(tmp_path, capsys):
    script = tmp_path / "wf_transcripts.py"
    script.write_text(TRANSCRIPT_SCRIPT)
    root = tmp_path / "root"
    assert cli.main(["--root", str(root), "run", str(script), "--transcripts"]) == 0
    run_dir = next((root / "runs").iterdir())
    path = run_dir / "agents" / "000-writer.jsonl"
    assert path.exists()
    assert _read_types(path) == ["assistant.message"]  # delta filtered out


def test_cli_transcripts_default_off(tmp_path, capsys):
    script = tmp_path / "wf_transcripts.py"
    script.write_text(TRANSCRIPT_SCRIPT)
    root = tmp_path / "root"
    assert cli.main(["--root", str(root), "run", str(script)]) == 0
    run_dir = next((root / "runs").iterdir())
    assert not (run_dir / "agents").exists()
