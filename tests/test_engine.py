"""Engine semantics: parallel, pipeline, phase, timeout-abort, module API,
safety caps, args identity, forensic records, and journaled nondeterminism."""

from __future__ import annotations

import asyncio
import json

import pytest

import rdw
from rdw.budget import Budget
from rdw.engine import Workflow
from rdw.errors import (
    AgentError,
    AgentLimitExceeded,
    AgentTimeout,
    BudgetExceeded,
    DivergenceWarning,
    WorkflowContextError,
)
from rdw.journal import Journal
from rdw.progress import Progress

from conftest import FakeRuntime, FakeSession, Turn, usage_event


def _wf(tmp_path, *, run_dir=None, resume=False, runtime=None, budget=None, **kwargs):
    """Direct Workflow construction for options make_wf doesn't surface
    (args, max_agents, max_wave)."""
    return Workflow(
        run_id="test-run",
        runtime=runtime if runtime is not None else FakeRuntime(),
        budget=budget if budget is not None else Budget(),
        journal=Journal(run_dir or tmp_path / "run", resume=resume),
        progress=Progress("test-run", force_plain=True),
        **kwargs,
    )


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


# ------------------------------------------------------------- safety caps


@pytest.mark.asyncio
async def test_max_agents_cap_stops_third_call(tmp_path):
    rt = FakeRuntime([[Turn(text="a")], [Turn(text="b")]])
    async with _wf(tmp_path, runtime=rt, max_agents=2) as wf:
        await wf.agent("one", label="a")
        await wf.agent("two", label="b")
        with pytest.raises(AgentLimitExceeded, match="exceeded 2 agent calls"):
            await wf.agent("three", label="c")


@pytest.mark.asyncio
async def test_agent_limit_escapes_parallel_but_agent_error_does_not(tmp_path):
    """The deliberate taxonomy asymmetry: AgentError degrades to None,
    AgentLimitExceeded (run-level misconfiguration) propagates."""
    rt = FakeRuntime([[Turn(error=RuntimeError("boom"))], [Turn(text="ok")]])
    async with _wf(tmp_path, runtime=rt, max_agents=10) as wf:
        results = await wf.parallel([lambda: wf.agent("fails", label="f")])
        assert results == [None]  # AgentError still absorbed

    rt2 = FakeRuntime()
    async with _wf(tmp_path, run_dir=tmp_path / "cap", runtime=rt2, max_agents=1) as wf2:
        await wf2.agent("one", label="a")
        with pytest.raises(AgentLimitExceeded):
            await wf2.parallel([lambda: wf2.agent("two", label="b")])


@pytest.mark.asyncio
async def test_max_wave_rejects_oversized_waves_before_running(tmp_path):
    ran: list[str] = []

    async def thunk():
        ran.append("ran")

    async def stage(item):
        ran.append("stage")
        return item

    async with _wf(tmp_path, max_wave=2) as wf:
        with pytest.raises(ValueError, match="3 items exceeds max_wave=2"):
            await wf.parallel([thunk, thunk, thunk])
        with pytest.raises(ValueError, match="3 items exceeds max_wave=2"):
            await wf.pipeline([1, 2, 3], stage)
    assert ran == []  # explicit error, nothing silently ran (or truncated)


@pytest.mark.asyncio
async def test_cached_replays_count_toward_agent_cap(tmp_path):
    """The cap bounds *calls* (they consume next_index), not just live spend."""
    run_dir = tmp_path / "capped"
    rt1 = FakeRuntime([[Turn(text="one")]])
    async with _wf(tmp_path, run_dir=run_dir, runtime=rt1) as wf1:
        await wf1.agent("p1", label="a")

    rt2 = FakeRuntime()
    async with _wf(tmp_path, run_dir=run_dir, resume=True, runtime=rt2, max_agents=1) as wf2:
        assert await wf2.agent("p1", label="a") == "one"  # cached, index 0
        with pytest.raises(AgentLimitExceeded):
            await wf2.agent("p2", label="b")  # index 1 >= cap
    assert rt2.created == []


# ---------------------------------------------------------- pipeline arity


@pytest.mark.asyncio
async def test_pipeline_stage_arity_1_2_3(make_wf):
    """Contract parity: stages may take (prev), (prev, item), or
    (prev, item, index); legacy 1-arg stages are untouched."""
    calls: list[tuple] = []

    async def legacy(current):
        calls.append(("legacy", current))
        return current + "x"

    async def with_item(current, item):
        calls.append(("with_item", current, item))
        return current + "y"

    async def with_index(current, item, index):
        calls.append(("with_index", current, item, index))
        return f"{current}@{index}"

    async with make_wf() as wf:
        results = await wf.pipeline(["a", "b"], legacy, with_item, with_index)

    assert results == ["axy@0", "bxy@1"]
    assert ("with_item", "ax", "a") in calls  # original item, not the stage-1 output
    assert ("with_index", "axy", "a", 0) in calls
    assert ("with_index", "bxy", "b", 1) in calls


@pytest.mark.asyncio
async def test_pipeline_star_args_stage_gets_all_three(make_wf):
    seen: list[tuple] = []

    async def star(*args):
        seen.append(args)
        return args[0]

    async with make_wf() as wf:
        await wf.pipeline(["i"], star)
    assert seen == [("i", "i", 0)]


# --------------------------------------------------------- forensic records


@pytest.mark.asyncio
async def test_request_context_recorded_on_ok_and_error(make_wf, tmp_path):
    rt = FakeRuntime([[Turn(text="fine")], [Turn(error=RuntimeError("died"))]])
    async with make_wf(runtime=rt, model="gpt-m", effort="low") as wf:
        await wf.agent("good prompt", label="ok-agent", timeout=42.0)
        with pytest.raises(AgentError):
            await wf.agent("bad prompt", label="err-agent")

    ok_rec, err_rec = wf.journal.records()
    for rec in (ok_rec, err_rec):
        assert rec.request is not None
        assert rec.request["model"] == "gpt-m"
        assert rec.request["effort"] == "low"
        assert rec.request["budget"] == {"total": None, "spent": 0.0, "outstanding": 0.0}
        assert rec.request["session_limits"] is None  # unlimited budget
    assert ok_rec.request["prompt_chars"] == len("good prompt")
    assert ok_rec.request["timeout"] == 42.0

    # request context is journaled on disk, not just in memory
    lines = [json.loads(ln) for ln in wf.journal.path.read_text().splitlines() if ln]
    agent_lines = [ln for ln in lines if ln["type"] == "agent"]
    assert all(ln["request"]["model"] == "gpt-m" for ln in agent_lines)


@pytest.mark.asyncio
async def test_budget_refusal_is_journaled_and_reruns_live_on_resume(make_wf, tmp_path):
    """The live-run forensic gap: a refusal must leave label + budget snapshot
    in the journal, and a resume with headroom re-runs the refused call."""
    run_dir = tmp_path / "refused"
    # First agent spends 2 AIU against a 1-AIU ceiling; the second is refused.
    rt1 = FakeRuntime([[Turn(text="pricey", events=[usage_event(2e9)])]])
    async with make_wf(runtime=rt1, run_dir=run_dir, budget=Budget(total=1.0)) as wf1:
        assert await wf1.agent("first", label="a") == "pricey"
        with pytest.raises(BudgetExceeded):
            await wf1.agent("second", label="b")

    lines = [json.loads(ln) for ln in (run_dir / "journal.jsonl").read_text().splitlines() if ln]
    [refusal] = [ln for ln in lines if ln["type"] == "refusal"]
    assert refusal["label"] == "b"
    assert refusal["budget"]["total"] == 1.0
    assert refusal["budget"]["spent"] == pytest.approx(2.0)

    # Resume with headroom: the ok record replays, the refused call runs live.
    rt2 = FakeRuntime([[Turn(text="now it runs")]])
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True, budget=Budget(total=50.0)) as wf2:
        assert await wf2.agent("first", label="a") == "pricey"
        assert await wf2.agent("second", label="b") == "now it runs"
    assert len(rt2.created) == 1
    assert wf2.journal.cache_hits == 1
    assert not wf2.journal.diverged


# ------------------------------------------------------------ args channel


@pytest.mark.asyncio
async def test_args_are_run_identity(tmp_path):
    """Same script, same args → 100% replay; different args → different
    fingerprints (divergence, live)."""
    run_dir = tmp_path / "args"
    rt1 = FakeRuntime([[Turn(text="one")]])
    async with _wf(tmp_path, run_dir=run_dir, runtime=rt1, args={"n": 3}) as wf1:
        await wf1.agent("p", label="a")

    rt2 = FakeRuntime()
    async with _wf(tmp_path, run_dir=run_dir, resume=True, runtime=rt2, args={"n": 3}) as wf2:
        assert await wf2.agent("p", label="a") == "one"
    assert rt2.created == []  # identical args: full replay

    rt3 = FakeRuntime([[Turn(text="two")]])
    async with _wf(tmp_path, run_dir=run_dir, resume=True, runtime=rt3, args={"n": 4}) as wf3:
        with pytest.warns(DivergenceWarning):
            assert await wf3.agent("p", label="a") == "two"
    assert len(rt3.created) == 1


@pytest.mark.asyncio
async def test_empty_args_reproduce_legacy_fingerprint(make_wf):
    """Fingerprint-compatibility regression: with no args (and default opts)
    the fingerprint must equal the canned pre-args value byte-for-byte, so
    every existing journal replays unchanged."""
    rt = FakeRuntime([[Turn(text="x")]])
    async with make_wf(runtime=rt) as wf:
        assert wf.args == {}  # wf.args is always readable
        await wf.agent("hello")
    [rec] = wf.journal.records()
    assert rec.fp == "5ecfeb2c2123e57fbbae617975ded266bfb61ecc9530e6306b849f603d7d987a"


# ------------------------------------------- journaled nondeterminism


@pytest.mark.asyncio
async def test_now_random_uuid_replay_identically_on_resume(make_wf, tmp_path):
    run_dir = tmp_path / "values"
    rt1 = FakeRuntime([[Turn(text="agent-out")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        t1 = wf1.now()
        r1 = wf1.random()
        await wf1.agent("mixed in", label="a")  # values and agents interleave
        u1 = rdw.uuid()  # module-level helpers bind to the ambient workflow
        t2 = rdw.now()
    assert isinstance(t1, float) and isinstance(r1, float) and isinstance(u1, str)
    assert t2 >= t1

    rt2 = FakeRuntime()
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
        assert wf2.now() == t1  # exact recorded values, not fresh ones
        assert wf2.random() == r1
        assert await wf2.agent("mixed in", label="a") == "agent-out"
        assert wf2.uuid() == u1
        assert wf2.now() == t2  # second occurrence keeps its own slot
    assert rt2.created == []  # zero live sessions: values did not bust the cache
    assert not wf2.journal.diverged


@pytest.mark.asyncio
async def test_run_boundary_lines_bracket_attempts(make_wf, tmp_path):
    run_dir = tmp_path / "bounds"
    async with make_wf(run_dir=run_dir):
        pass
    async with make_wf(run_dir=run_dir, resume=True):
        pass
    lines = [json.loads(ln) for ln in (run_dir / "journal.jsonl").read_text().splitlines() if ln]
    boundaries = [ln for ln in lines if ln["type"] == "boundary"]
    assert [b["event"] for b in boundaries] == ["start", "resume"]
    info = boundaries[0]["info"]
    assert {"ts", "pid", "budget_total", "model", "effort", "rdw_version",
            "cache_records_loaded"} <= set(info)
    assert boundaries[0]["info"]["cache_records_loaded"] == 0


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
