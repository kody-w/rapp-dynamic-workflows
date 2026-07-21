"""Journal fingerprinting, replay, divergence, and error re-execution."""

from __future__ import annotations

import asyncio
import contextlib
import json
import warnings

import pytest
from pydantic import BaseModel

from rdw.errors import AgentError, DivergenceWarning, JournalError, JournalWarning
from rdw.journal import Journal, fingerprint

from conftest import FakeRuntime, Turn


class Note(BaseModel):
    body: str


@contextlib.contextmanager
def _no_divergence():
    """Fail the test if any DivergenceWarning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DivergenceWarning)
        yield


# ------------------------------------------------------------- fingerprint


def test_fingerprint_deterministic_and_sensitive():
    opts = {"model": "m", "effort": None, "schema": None, "tools": [], "cwd": None}
    a = fingerprint("prompt", opts)
    assert a == fingerprint("prompt", dict(opts))  # content-addressed, stable
    assert a != fingerprint("other prompt", opts)  # prompt matters
    assert a != fingerprint("prompt", {**opts, "model": "m2"})  # opts matter


# ------------------------------------------------------------------ replay


@pytest.mark.asyncio
async def test_resume_replays_cached_results_without_sessions(make_wf, tmp_path):
    run_dir = tmp_path / "r1"
    rt1 = FakeRuntime([[Turn(text="one")], [Turn(submit=[{"body": "two"}])]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        assert await wf1.agent("first", label="a") == "one"
        note = await wf1.agent("second", schema=Note, label="b")
        assert note.body == "two"

    rt2 = FakeRuntime()
    with _no_divergence():
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            assert await wf2.agent("first", label="a") == "one"
            replayed = await wf2.agent("second", schema=Note, label="b")
    assert isinstance(replayed, Note) and replayed.body == "two"  # re-validated
    assert rt2.created == []  # zero live sessions
    assert wf2.journal.cache_hits == 2
    assert not wf2.journal.diverged


@pytest.mark.asyncio
async def test_label_and_timeout_do_not_bust_cache(make_wf, tmp_path):
    run_dir = tmp_path / "r2"
    rt1 = FakeRuntime([[Turn(text="cached")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("stable prompt", label="original", timeout=100)

    rt2 = FakeRuntime()
    with _no_divergence():
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            result = await wf2.agent("stable prompt", label="renamed", timeout=5)
    assert result == "cached"
    assert rt2.created == []


@pytest.mark.asyncio
async def test_model_change_busts_cache(make_wf, tmp_path):
    run_dir = tmp_path / "r3"
    rt1 = FakeRuntime([[Turn(text="old")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("p", label="a")

    rt2 = FakeRuntime([[Turn(text="new")]])
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
        with pytest.warns(DivergenceWarning):
            result = await wf2.agent("p", label="a", model="different-model")
    assert result == "new"
    assert len(rt2.created) == 1


# -------------------------------------------------------------- divergence


@pytest.mark.asyncio
async def test_divergence_goes_live_and_marks_journal(make_wf, tmp_path):
    run_dir = tmp_path / "r4"
    rt1 = FakeRuntime([[Turn(text="one")], [Turn(text="two")], [Turn(text="three")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        for p in ("p1", "p2", "p3"):
            await wf1.agent(p, label=p)

    # resume with a changed second prompt: p1 cached, p2/p3 live
    rt2 = FakeRuntime([[Turn(text="two-B")], [Turn(text="three-B")]])
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
        assert await wf2.agent("p1", label="p1") == "one"
        with pytest.warns(DivergenceWarning):
            assert await wf2.agent("p2-changed", label="p2") == "two-B"
        # p3's prompt matches the original journal, but the run has diverged:
        # everything after the divergence point runs live.
        assert await wf2.agent("p3", label="p3") == "three-B"
    assert wf2.journal.cache_hits == 1
    assert wf2.journal.diverged
    assert len(rt2.created) == 2

    lines = [
        json.loads(ln)
        for ln in (run_dir / "journal.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    markers = [ln for ln in lines if ln["type"] == "divergence"]
    assert len(markers) == 1 and markers[0]["index"] == 1


@pytest.mark.asyncio
async def test_post_divergence_resume_replays_new_tail(make_wf, tmp_path):
    run_dir = tmp_path / "r5"
    rt1 = FakeRuntime([[Turn(text="one")], [Turn(text="two")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("p1", label="p1")
        await wf1.agent("p2", label="p2")

    rt2 = FakeRuntime([[Turn(text="two-B")]])
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
        await wf2.agent("p1", label="p1")
        with pytest.warns(DivergenceWarning):
            await wf2.agent("p2-changed", label="p2")

    # third generation: the *new* tail is now the cache — full replay
    rt3 = FakeRuntime()
    with _no_divergence():
        async with make_wf(runtime=rt3, run_dir=run_dir, resume=True) as wf3:
            assert await wf3.agent("p1", label="p1") == "one"
            assert await wf3.agent("p2-changed", label="p2") == "two-B"
    assert rt3.created == []
    assert wf3.journal.cache_hits == 2


# ---------------------------------------------------------- error records


@pytest.mark.asyncio
async def test_error_record_reexecutes_without_divergence(make_wf, tmp_path):
    run_dir = tmp_path / "r6"
    rt1 = FakeRuntime([[Turn(error=RuntimeError("flaky"))]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        with pytest.raises(AgentError):
            await wf1.agent("fragile", label="f")
    [record] = wf1.journal.records()
    assert record.status == "error" and "flaky" in (record.error or "")

    rt2 = FakeRuntime([[Turn(text="recovered")]])
    with _no_divergence():
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            assert await wf2.agent("fragile", label="f") == "recovered"
    assert len(rt2.created) == 1  # retried live, no divergence, no cache hit
    assert wf2.journal.cache_hits == 0
    assert not wf2.journal.diverged


# ------------------------------------------------------------ append-only


@pytest.mark.asyncio
async def test_journal_is_append_only_across_generations(make_wf, tmp_path):
    run_dir = tmp_path / "r7"
    rt1 = FakeRuntime([[Turn(text="one")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("p1", label="p1")
    first_gen = (run_dir / "journal.jsonl").read_text()

    rt2 = FakeRuntime([[Turn(text="one-B")]])
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
        with pytest.warns(DivergenceWarning):
            await wf2.agent("p1-changed", label="p1")
    second_gen = (run_dir / "journal.jsonl").read_text()
    assert second_gen.startswith(first_gen)  # history preserved, never rewritten


@pytest.mark.asyncio
async def test_session_id_recorded_per_agent(make_wf):
    rt = FakeRuntime([[Turn(text="x")]])
    async with make_wf(runtime=rt) as wf:
        await wf.agent("p", label="a")
    [record] = wf.journal.records()
    assert record.session_id == rt.created[0].session_id


# --------------------------------------------- scheduling-independent resume


@pytest.mark.asyncio
async def test_pipeline_resume_is_scheduling_independent(make_wf, tmp_path):
    """Regression: journal positions must not depend on completion order.

    Run 1 forces item B through both stages before item A even finishes
    stage 1; the resume runs with no delays (and cached replays complete
    synchronously anyway), so the *start order* of stage-2 agents differs.
    An identical script must still replay 100% from the journal.
    """
    run_dir = tmp_path / "pipe"

    def stages(wf, delay_a: float):
        async def stage1(item):
            if item == "A" and delay_a:
                await asyncio.sleep(delay_a)
            return await wf.agent(f"stage1 for {item}", label=f"s1-{item}")

        async def stage2(item):
            return await wf.agent(f"stage2 after {item!r}", label=f"s2-{item}")

        return stage1, stage2

    rt1 = FakeRuntime()  # every live session answers "default"
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        s1, s2 = stages(wf1, delay_a=0.05)
        results1 = await wf1.pipeline(["A", "B"], s1, s2)
    assert results1 == ["default", "default"]
    assert len(rt1.created) == 4

    rt2 = FakeRuntime()
    with _no_divergence():
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            s1, s2 = stages(wf2, delay_a=0.0)  # different scheduling on purpose
            results2 = await wf2.pipeline(["A", "B"], s1, s2)
    assert results2 == ["default", "default"]
    assert rt2.created == []  # zero live sessions: full replay
    assert wf2.journal.cache_hits == 4
    assert not wf2.journal.diverged


@pytest.mark.asyncio
async def test_identical_prompts_replay_by_occurrence(make_wf, tmp_path):
    """Two calls with the same (prompt, opts) each keep their own cache slot."""
    run_dir = tmp_path / "occ"
    rt1 = FakeRuntime([[Turn(text="one")], [Turn(text="two")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        assert await wf1.agent("same prompt", label="a") == "one"
        assert await wf1.agent("same prompt", label="b") == "two"

    rt2 = FakeRuntime()
    with _no_divergence():
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            assert await wf2.agent("same prompt", label="a") == "one"
            assert await wf2.agent("same prompt", label="b") == "two"
    assert rt2.created == []
    assert wf2.journal.cache_hits == 2


# --------------------------------------------------------- crash tolerance


@pytest.mark.asyncio
async def test_torn_final_line_does_not_disable_resume(make_wf, tmp_path):
    """A crash mid-append leaves a truncated last line; resume must skip it
    with a warning instead of raising JournalError forever."""
    run_dir = tmp_path / "torn"
    rt1 = FakeRuntime([[Turn(text="one")], [Turn(text="two")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("p1", label="a")
        await wf1.agent("p2", label="b")

    path = run_dir / "journal.jsonl"
    # Simulate a crash mid-append: torn, unterminated final record.
    path.write_bytes(path.read_bytes() + b'{"type": "agent", "index": 9, "fp": "unterm')

    with pytest.warns(JournalWarning, match="torn final journal line"):
        Journal(run_dir, resume=True)

    rt2 = FakeRuntime()
    with _no_divergence():
        warnings.simplefilter("ignore", JournalWarning)  # already asserted above
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            assert await wf2.agent("p1", label="a") == "one"
            assert await wf2.agent("p2", label="b") == "two"
            wf2.log("resumed")  # append after the torn line: must not merge
    assert rt2.created == []
    assert wf2.journal.cache_hits == 2

    # Appends after the torn line start on a fresh line (no record merging):
    # every line but the torn one must parse as JSON.
    lines = [ln for ln in path.read_text().split("\n") if ln.strip()]
    bad = [ln for ln in lines if not _parses(ln)]
    assert bad == ['{"type": "agent", "index": 9, "fp": "unterm']


def _parses(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def test_mid_file_corruption_still_raises(tmp_path):
    run_dir = tmp_path / "corrupt"
    run_dir.mkdir()
    (run_dir / "journal.jsonl").write_text(
        '{"type": "log", "message": "ok"\n{"type": "log", "message": "fine"}\n'
    )
    with pytest.raises(JournalError, match="invalid JSON"):
        Journal(run_dir, resume=True)


# ------------------------------------------------------- forensic records


def test_agent_record_round_trips_request():
    from rdw.journal import AgentRecord

    request = {
        "model": "gpt-x",
        "effort": "high",
        "session_limits": {"max_ai_credits": 30.0},
        "budget": {"total": 40.0, "spent": 1.5, "outstanding": 19.25},
        "prompt_chars": 512,
    }
    rec = AgentRecord(
        index=3, fp="f" * 64, label="a", phase="p", status="ok", request=request
    )
    back = AgentRecord.from_obj(json.loads(rec.to_line()))
    assert back.request == request
    # absent request stays None (legacy lines have no key at all)
    legacy = AgentRecord(index=0, fp="e" * 64, label="b", phase=None, status="ok")
    assert AgentRecord.from_obj(json.loads(legacy.to_line())).request is None


@pytest.mark.asyncio
async def test_boundary_and_refusal_lines_ignored_by_loader(make_wf, tmp_path):
    """Forensic lines are history, not cache: replay still works around them,
    and a refusal never replays as a result."""
    run_dir = tmp_path / "forensic"
    rt1 = FakeRuntime([[Turn(text="one")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("p1", label="a")
    # synthesize the refusal a capped sibling would have left
    wf1.journal.refusal(
        index=1,
        fp="c" * 64,
        seq=0,
        label="refused-agent",
        phase=None,
        budget={"total": 1.0, "spent": 2.0, "outstanding": 0.0},
    )

    rt2 = FakeRuntime([[Turn(text="live-now")]])
    with _no_divergence():
        async with make_wf(runtime=rt2, run_dir=run_dir, resume=True) as wf2:
            assert await wf2.agent("p1", label="a") == "one"  # cached, not disturbed
            # the refused position runs live (refusals never enter the cache);
            # by now the ok-record was consumed, so this is appended new work.
            assert await wf2.agent("was refused", label="refused-agent") == "live-now"
    assert len(rt2.created) == 1
    assert wf2.journal.cache_hits == 1

    lines = [json.loads(ln) for ln in (run_dir / "journal.jsonl").read_text().splitlines() if ln]
    types = [ln["type"] for ln in lines]
    assert types.count("boundary") == 2  # one start, one resume
    events = [ln["event"] for ln in lines if ln["type"] == "boundary"]
    assert events == ["start", "resume"]
    [refusal] = [ln for ln in lines if ln["type"] == "refusal"]
    assert refusal["label"] == "refused-agent"
    assert refusal["budget"]["spent"] == 2.0


def test_boundary_info_shape(tmp_path):
    journal = Journal(tmp_path / "b")
    journal.run_boundary(
        event="start",
        info={"ts": 1.0, "pid": 42, "budget_total": None, "rdw_version": "0.1.0"},
    )
    [line] = [json.loads(ln) for ln in journal.path.read_text().splitlines() if ln]
    assert line == {
        "type": "boundary",
        "event": "start",
        "info": {"ts": 1.0, "pid": 42, "budget_total": None, "rdw_version": "0.1.0"},
    }


# ------------------------------------------------------------------ values


def test_value_record_and_replay_across_resume(tmp_path):
    run_dir = tmp_path / "values"
    j1 = Journal(run_dir)
    assert j1.value_lookup("now", j1.next_value_occurrence("now")) is None
    j1.value_record("now", 0, 1234.5)
    j1.value_record("uuid", 0, "abc-123")

    j2 = Journal(run_dir, resume=True)
    with _no_divergence():
        assert j2.value_lookup("now", j2.next_value_occurrence("now")) == 1234.5
        assert j2.value_lookup("uuid", j2.next_value_occurrence("uuid")) == "abc-123"
    assert not j2.diverged


def test_value_kind_mismatch_diverges(tmp_path):
    """A resumed script asking for a different value kind than recorded is a
    real divergence: everything downstream of the value shifts."""
    run_dir = tmp_path / "vkind"
    j1 = Journal(run_dir)
    j1.value_record("now", 0, 99.0)

    j2 = Journal(run_dir, resume=True)
    with pytest.warns(DivergenceWarning, match=r"value random\[0\]"):
        assert j2.value_lookup("random", j2.next_value_occurrence("random")) is None
    assert j2.diverged
    lines = [json.loads(ln) for ln in j2.path.read_text().splitlines() if ln]
    marker = [ln for ln in lines if ln["type"] == "divergence"][-1]
    assert marker["kind"] == "random"
