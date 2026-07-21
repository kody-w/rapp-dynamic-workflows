"""Heartbeat / liveness indicators: a wedged agent must be visible before
its 600s timeout, not after."""

from __future__ import annotations

import asyncio

import pytest

from rdw.progress import Progress

from conftest import FakeRuntime, Turn, event


# ---------------------------------------------------------- line composition


def test_heartbeat_line_shows_elapsed_tokens_and_last_activity():
    p = Progress("hb", force_plain=True)
    p._now = lambda: 1000.0
    p.agent_started("strategy-1", "design")
    p.agent_tokens("strategy-1", 8200)
    p._now = lambda: 1081.0
    p.agent_activity("strategy-1", "bash")
    p._now = lambda: 1084.0
    assert p.heartbeat_line() == "· 1 running: strategy-1 84s/8.2k last: bash 3s ago"


def test_heartbeat_line_empty_when_nothing_running():
    p = Progress(force_plain=True)
    assert p.heartbeat_line() == ""
    p.agent_started("a", None)
    p.agent_finished("a", "ok")
    assert p.heartbeat_line() == ""  # finished agents never count as running


def test_heartbeat_line_is_bounded():
    p = Progress(force_plain=True)
    p._now = lambda: 0.0
    for i in range(6):
        p.agent_started(f"agent-{i}", None)
    line = p.heartbeat_line()
    assert line.startswith("· 6 running: ")
    assert "+2 more" in line
    assert "agent-4" not in line  # collapsed, not listed


# ------------------------------------------------------------ heartbeat task


@pytest.mark.asyncio
async def test_heartbeat_prints_periodically_and_stops(capsys):
    p = Progress("hb", force_plain=True, heartbeat=0.02)
    p.start()
    p.agent_started("worker", None)
    await asyncio.sleep(0.15)
    p.stop()
    await asyncio.sleep(0)  # let the cancelled task finish
    assert capsys.readouterr().out.count("running: worker") >= 2

    await asyncio.sleep(0.06)  # well past the interval, after stop()
    assert "running:" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_heartbeat_silent_while_idle(capsys):
    p = Progress("hb", force_plain=True, heartbeat=0.02)
    p.start()
    await asyncio.sleep(0.07)
    p.stop()
    await asyncio.sleep(0)
    assert "running:" not in capsys.readouterr().out


def test_start_without_running_loop_is_safe():
    p = Progress(force_plain=True)
    p.start()  # no event loop: heartbeat is simply off
    p.stop()
    assert p._heartbeat_task is None


@pytest.mark.asyncio
async def test_heartbeat_disabled_with_none():
    p = Progress(force_plain=True, heartbeat=None)
    p.start()
    assert p._heartbeat_task is None
    p.stop()


# ------------------------------------------------------- engine tap feeding


@pytest.mark.asyncio
async def test_tool_execution_start_feeds_last_activity(make_wf):
    rt = FakeRuntime(
        [[Turn(text="ok", events=[event("tool.execution_start", tool_name="bash", arguments={})])]]
    )
    async with make_wf(runtime=rt) as wf:
        await wf.agent("dig around", label="digger")
        state = wf.progress._agents["digger"]
    assert state.last_event == "bash"
    assert state.last_event_ts > 0
