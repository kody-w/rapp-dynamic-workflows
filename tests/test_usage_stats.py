"""Token/tool usage persistence, phase rollups, and `rdw show --stats`."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rdw import cli
from rdw.errors import AgentSchemaError
from rdw.journal import AgentRecord
from rdw.transcripts import UsageTap

from conftest import FakeRuntime, Turn, event

NANO = 1_000_000_000


def _full_usage_event(
    nano_aiu: float, *, input_tokens: int | None = None, output_tokens: int | None = None
):
    """An ``assistant.usage`` event carrying both cost and token splits."""
    return event(
        "assistant.usage",
        copilot_usage=SimpleNamespace(total_nano_aiu=nano_aiu),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ------------------------------------------------------------ UsageTap unit


def test_usage_tap_accumulates_tokens_and_calls():
    usage = UsageTap()
    tap = usage.tap()
    tap(event("assistant.usage", input_tokens=100, output_tokens=40))
    tap(event("assistant.usage", input_tokens=50, output_tokens=10, cache_read_tokens=30))
    tap(event("tool.execution_start", tool_name="bash", arguments={}))
    tap(event("tool.execution_start", tool_name="view", arguments={}))
    assert usage.snapshot() == {
        "model_calls": 2,
        "input_tokens": 150,
        "output_tokens": 50,
        "cache_read_tokens": 30,
        "tool_calls": 2,
    }


def test_usage_tap_empty_snapshot_is_none():
    usage = UsageTap()
    usage.tap()(event("session.idle"))  # irrelevant event type
    usage.tap()(object())  # malformed event: swallowed
    assert usage.snapshot() is None


# ------------------------------------------------- journal persistence


def test_agent_record_usage_round_trips():
    rec = AgentRecord(
        index=0,
        fp="f",
        label="a",
        phase=None,
        status="ok",
        usage={"input_tokens": 10, "output_tokens": 5, "model_calls": 1},
    )
    parsed = AgentRecord.from_obj(json.loads(rec.to_line()))
    assert parsed.usage == rec.usage
    bare = AgentRecord.from_obj({"index": 1, "fp": "g", "status": "ok"})
    assert bare.usage is None


@pytest.mark.asyncio
async def test_usage_recorded_on_ok_record(make_wf):
    rt = FakeRuntime(
        [
            [
                Turn(
                    text="done",
                    events=[
                        _full_usage_event(1.5 * NANO, input_tokens=1000, output_tokens=500),
                        event("tool.execution_start", tool_name="bash", arguments={}),
                    ],
                )
            ]
        ]
    )
    async with make_wf(runtime=rt) as wf:
        await wf.agent("work", label="worker")
    [record] = wf.journal.records()
    assert record.usage == {
        "model_calls": 1,
        "input_tokens": 1000,
        "output_tokens": 500,
        "tool_calls": 1,
    }
    assert record.credits == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_usage_recorded_on_error_record(make_wf):
    """The error path is the one that pays for telemetry — a schema failure
    still journals what the session consumed before it stonewalled."""
    rt = FakeRuntime(
        [[Turn(text="no submit", events=[event("assistant.usage", output_tokens=40)])]]
    )
    async with make_wf(runtime=rt) as wf:
        with pytest.raises(AgentSchemaError):
            await wf.agent("extract", schema={"type": "object"}, label="stonewalled")
    [record] = wf.journal.records()
    assert record.status == "error"
    assert record.usage == {"model_calls": 1, "output_tokens": 40}


# -------------------------------------------------------- report rollups


@pytest.mark.asyncio
async def test_report_includes_phase_rollups(make_wf):
    rt = FakeRuntime(
        [
            [Turn(text="a", events=[_full_usage_event(1.5 * NANO, input_tokens=1000, output_tokens=500)])],
            [Turn(text="b", events=[_full_usage_event(0.5 * NANO, output_tokens=100)])],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        with wf.phase("design"):
            await wf.agent("one", label="one")
            await wf.agent("two", label="two")
    report = wf.report()
    assert "phases:" in report
    assert "design: 2 agent(s) (2 ok), 2.00 AIU, 1.6k tok" in report


@pytest.mark.asyncio
async def test_report_replay_saved_on_resume(make_wf):
    rt = FakeRuntime([[Turn(text="hi", events=[_full_usage_event(1.5 * NANO)])]])
    async with make_wf(runtime=rt) as wf1:
        await wf1.agent("alpha", label="a")
    assert "replay saved" not in wf1.report()  # nothing cached on a fresh run

    async with make_wf(runtime=FakeRuntime(), resume=True) as wf2:
        await wf2.agent("alpha", label="a")  # identical call: journal replay
    assert wf2.journal.cache_hits == 1
    assert "replay saved ~1.50 AIU" in wf2.report()


# ------------------------------------------------------- rdw show --stats


def _write_journal(root, run_id: str, objs: list[dict]) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    run_dir.joinpath("journal.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in objs), encoding="utf-8"
    )


def test_show_stats_rolls_up_phases_and_totals(tmp_path, capsys):
    root = tmp_path / "root"
    _write_journal(
        root,
        "r1",
        [
            {"type": "log", "message": "hello", "ts": 0},
            {
                "type": "agent", "index": 0, "fp": "f1", "seq": 0, "label": "a",
                "phase": "design", "status": "ok", "credits": 1.5,
                "started": 0.0, "ended": 10.0,
                "usage": {"input_tokens": 1000, "output_tokens": 500,
                          "model_calls": 2, "tool_calls": 3},
            },
            {
                "type": "agent", "index": 1, "fp": "f2", "seq": 0, "label": "b",
                "phase": "design", "status": "error", "credits": 0.5,
                "started": 0.0, "ended": 5.0,
            },
            {
                "type": "agent", "index": 2, "fp": "f3", "seq": 0, "label": "c",
                "phase": None, "status": "ok", "credits": 1.0,
                "started": 0.0, "ended": 2.5,
            },
            # retry of f2 succeeds later: last record per (fp, seq) wins
            {
                "type": "agent", "index": 3, "fp": "f2", "seq": 0, "label": "b",
                "phase": "design", "status": "ok", "credits": 0.7,
                "started": 0.0, "ended": 3.0,
            },
        ],
    )
    assert cli.main(["--root", str(root), "show", "r1", "--stats"]) == 0
    out = capsys.readouterr().out
    assert "total: 3 agent(s) (3 ok), 3.20 AIU" in out
    assert "1500 tok (1000 in / 500 out)" in out
    assert "2 model call(s), 3 tool call(s)" in out
    assert "design: 2 agent(s) (2 ok), 2.20 AIU, 1.5k tok, 13.0s wall" in out
    assert "(no phase): 1 agent(s) (1 ok), 1.00 AIU, 2.5s wall" in out
    assert "hello" not in out  # --stats replaces the journal dump


def test_show_stats_without_agent_records(tmp_path, capsys):
    root = tmp_path / "root"
    _write_journal(root, "r2", [{"type": "log", "message": "only logs", "ts": 0}])
    assert cli.main(["--root", str(root), "show", "r2", "--stats"]) == 0
    assert "(no agent records)" in capsys.readouterr().out
