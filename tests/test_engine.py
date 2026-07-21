"""Engine semantics: parallel, pipeline, phase, timeout-abort, module API."""

from __future__ import annotations

import asyncio

import pytest

import rdw
from rdw.errors import AgentError, AgentTimeout, WorkflowContextError

from conftest import FakeRuntime, FakeSession, Turn


class _ExplodingRuntime(FakeRuntime):
    """create_session rejects like the live API does (e.g. the 30-credit
    session_limits floor)."""

    async def create_session(self, **kwargs):
        raise RuntimeError("Minimum session limit is 30 AI credits.")


@pytest.mark.asyncio
async def test_create_session_failure_wraps_as_agent_error(make_wf):
    async with make_wf(runtime=_ExplodingRuntime([])) as wf:
        with pytest.raises(AgentError, match="session create failed"):
            await wf.agent("solo")
        results = await wf.parallel([lambda: wf.agent("branch")])
    assert results == [None]


# ---------------------------------------------------------------- parallel


@pytest.mark.asyncio
async def test_parallel_failure_becomes_none_never_raises(make_wf):
    rt = FakeRuntime(
        [
            [Turn(text="first")],
            [Turn(error=RuntimeError("session exploded"))],
            [Turn(text="third")],
        ]
    )
    async with make_wf(runtime=rt) as wf:
        results = await wf.parallel(
            [
                lambda: wf.agent("a", label="a"),
                lambda: wf.agent("b", label="b"),
                lambda: wf.agent("c", label="c"),
            ]
        )
    assert results == ["first", None, "third"]


@pytest.mark.asyncio
async def test_parallel_preserves_input_order(make_wf):
    async with make_wf() as wf:

        async def slow():
            await asyncio.sleep(0.05)
            return "slow"

        async def fast():
            return "fast"

        assert await wf.parallel([slow, fast]) == ["slow", "fast"]


@pytest.mark.asyncio
async def test_parallel_accepts_bare_awaitables(make_wf):
    async with make_wf() as wf:

        async def val(x):
            return x

        assert await wf.parallel([val(1), val(2)]) == [1, 2]


# ---------------------------------------------------------------- pipeline


@pytest.mark.asyncio
async def test_pipeline_no_barrier_between_stages(make_wf):
    """Item B reaches stage 2 while item A is still inside stage 1."""
    log: list[tuple[str, str]] = []

    async def stage1(item):
        await asyncio.sleep(0.08 if item == "A" else 0.0)
        log.append(("s1", item))
        return item

    async def stage2(item):
        log.append(("s2", item))
        return item + "!"

    async with make_wf() as wf:
        results = await wf.pipeline(["A", "B"], stage1, stage2)

    assert results == ["A!", "B!"]  # order preserved despite no barrier
    assert log.index(("s2", "B")) < log.index(("s1", "A"))


@pytest.mark.asyncio
async def test_pipeline_stage_exception_drops_item(make_wf):
    calls: list[tuple[str, int]] = []

    async def double(item):
        calls.append(("double", item))
        if item == 2:
            raise ValueError("boom")
        return item * 2

    async def stringify(item):
        calls.append(("stringify", item))
        return str(item)

    async with make_wf() as wf:
        results = await wf.pipeline([1, 2, 3], double, stringify)

    assert results == ["2", None, "6"]
    # the dropped item never entered its later stages
    assert ("stringify", None) not in calls
    assert all(not (name == "stringify" and arg == 4) for name, arg in calls)


@pytest.mark.asyncio
async def test_pipeline_none_short_circuits_remaining_stages(make_wf):
    seen = []

    async def to_none(item):
        return None if item == "drop" else item

    async def record(item):
        seen.append(item)
        return item

    async with make_wf() as wf:
        results = await wf.pipeline(["keep", "drop"], to_none, record)

    assert results == ["keep", None]
    assert seen == ["keep"]


# --------------------------------------------------------------- timeout


@pytest.mark.asyncio
async def test_timeout_aborts_session_and_raises(make_wf):
    session = FakeSession([Turn(error=TimeoutError())])
    rt = FakeRuntime([session])
    async with make_wf(runtime=rt) as wf:
        with pytest.raises(AgentTimeout):
            await wf.agent("slow task", timeout=5.0, label="slow")
    assert session.aborted
    assert session.disconnected
    [record] = wf.journal.records()
    assert record.status == "error"


@pytest.mark.asyncio
async def test_session_exception_becomes_agent_error(make_wf):
    rt = FakeRuntime([[Turn(error=ConnectionError("pipe broke"))]])
    async with make_wf(runtime=rt) as wf:
        with pytest.raises(AgentError, match="pipe broke"):
            await wf.agent("x", label="broken")


# ------------------------------------------------------------ phase & log


@pytest.mark.asyncio
async def test_phase_scopes_journal_records(make_wf):
    rt = FakeRuntime([[Turn(text="a")], [Turn(text="b")], [Turn(text="c")]])
    async with make_wf(runtime=rt) as wf:
        with wf.phase("design"):
            await wf.agent("one", label="one")
        async with wf.phase("build"):
            await wf.agent("two", label="two")
        await wf.agent("three", label="three")
    phases = [r.phase for r in wf.journal.records()]
    assert phases == ["design", "build", None]


@pytest.mark.asyncio
async def test_phase_inherited_by_concurrent_tasks(make_wf):
    rt = FakeRuntime([[Turn(text="a")], [Turn(text="b")]])
    async with make_wf(runtime=rt) as wf:
        with wf.phase("wave"):
            await wf.parallel(
                [lambda: wf.agent("a", label="a"), lambda: wf.agent("b", label="b")]
            )
    assert all(r.phase == "wave" for r in wf.journal.records())


@pytest.mark.asyncio
async def test_explicit_phase_overrides_ambient(make_wf):
    rt = FakeRuntime([[Turn(text="a")]])
    async with make_wf(runtime=rt) as wf:
        with wf.phase("outer"):
            await wf.agent("x", label="x", phase="inner")
    [record] = wf.journal.records()
    assert record.phase == "inner"


# --------------------------------------------------- module-level helpers


@pytest.mark.asyncio
async def test_module_level_api_binds_to_active_workflow(make_wf):
    rt = FakeRuntime([[Turn(text="bound")]])
    async with make_wf(runtime=rt) as wf:
        with rdw.phase("p"):
            result = await rdw.agent("hello", label="m")
            rdw.log("note")
        assert rdw.current_workflow() is wf
    assert result == "bound"


@pytest.mark.asyncio
async def test_module_level_api_without_workflow_raises():
    with pytest.raises(WorkflowContextError):
        await rdw.agent("nope")
    with pytest.raises(WorkflowContextError):
        rdw.log("nope")


# ------------------------------------------------------------ concurrency


@pytest.mark.asyncio
async def test_runtime_slot_caps_concurrency():
    rt = FakeRuntime(concurrency=1)
    async with rt.slot():
        second = rt.slot()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(second.__aenter__(), timeout=0.05)


@pytest.mark.asyncio
async def test_workflow_exit_closes_runtime(make_wf):
    rt = FakeRuntime()
    async with make_wf(runtime=rt):
        pass
    assert rt.closed


# ----------------------------------------------------------- model/effort


@pytest.mark.asyncio
async def test_model_effort_cwd_overrides_reach_session(make_wf):
    rt = FakeRuntime([[Turn(text="a")], [Turn(text="b")]])
    async with make_wf(runtime=rt, model="gpt-base", effort="low", cwd="/tmp/wfdir") as wf:
        await wf.agent("defaulted", label="d")
        await wf.agent("overridden", label="o", model="gpt-big", effort="xhigh", cwd="/tmp/other")
    assert rt.create_kwargs[0]["model"] == "gpt-base"
    assert rt.create_kwargs[0]["reasoning_effort"] == "low"
    assert rt.create_kwargs[0]["working_directory"] == "/tmp/wfdir"
    assert rt.create_kwargs[1]["model"] == "gpt-big"
    assert rt.create_kwargs[1]["reasoning_effort"] == "xhigh"
    assert rt.create_kwargs[1]["working_directory"] == "/tmp/other"
