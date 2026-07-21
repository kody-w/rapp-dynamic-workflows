"""Budget accounting (usage events + checkpoints) and hard-ceiling enforcement."""

from __future__ import annotations

import asyncio

import pytest

from rdw.budget import Budget
from rdw.errors import BudgetExceeded

from conftest import FakeRuntime, FakeSession, Turn, checkpoint_event, usage_event

NANO = 1_000_000_000  # 1 AIU


# ------------------------------------------------------------ unit: accounting


def test_tap_accumulates_assistant_usage():
    b = Budget(total=None)
    tap = b.tap("s1")
    tap(usage_event(0.25 * NANO))
    tap(usage_event(0.50 * NANO))
    assert b.spent() == pytest.approx(0.75)
    assert b.session_spent("s1") == pytest.approx(0.75)
    assert b.remaining() is None  # unlimited: accounting only


def test_checkpoint_is_cumulative_max_not_double_counted():
    b = Budget(total=None)
    tap = b.tap("s1")
    tap(usage_event(0.3 * NANO))
    tap(checkpoint_event(0.4 * NANO))  # cumulative >= usage sum: checkpoint wins
    assert b.session_spent("s1") == pytest.approx(0.4)
    tap(checkpoint_event(0.2 * NANO))  # stale checkpoint never lowers spend
    assert b.session_spent("s1") == pytest.approx(0.4)
    tap(usage_event(0.3 * NANO))  # usage sum 0.6 now exceeds checkpoint
    assert b.session_spent("s1") == pytest.approx(0.6)


def test_spend_sums_across_sessions():
    b = Budget(total=10.0)
    b.tap("a")(usage_event(1.0 * NANO))
    b.tap("b")(usage_event(2.5 * NANO))
    assert b.spent() == pytest.approx(3.5)
    assert b.remaining() == pytest.approx(6.5)


def test_malformed_events_never_break_the_tap():
    b = Budget(total=None)
    tap = b.tap("s1")
    tap(object())  # no .type at all
    tap(usage_event(None))  # missing nano value
    assert b.spent() == 0.0


def test_ensure_available_raises_at_ceiling():
    b = Budget(total=1.0)
    b.tap("s1")(usage_event(1.0 * NANO))
    with pytest.raises(BudgetExceeded) as exc_info:
        b.ensure_available(label="next-agent")
    assert exc_info.value.spent == pytest.approx(1.0)
    assert exc_info.value.total == pytest.approx(1.0)
    assert exc_info.value.label == "next-agent"


def test_session_limits_reflect_remaining():
    assert Budget(total=None).session_limits() is None
    b = Budget(total=200.0)
    assert b.session_limits() == {"max_ai_credits": pytest.approx(200.0)}
    b.tap("s1")(usage_event(150.0 * NANO))
    assert b.session_limits() == {"max_ai_credits": pytest.approx(50.0)}
    # Below the provider's 30-credit floor the cap clamps UP to the floor —
    # the API rejects smaller session_limits, and no cap at all lets a
    # running session blow far past the budget (observed live: 267/40).
    b.tap("s1")(usage_event(180.0 * NANO))
    assert b.session_limits() == {"max_ai_credits": pytest.approx(30.0)}


# ------------------------------------------------------- unit: reservations


def test_reservations_divide_remaining_and_never_exceed_total():
    """Concurrent grants shrink geometrically: their sum stays under total."""
    b = Budget(total=1.0)
    r1 = b.reserve(label="a")
    r2 = b.reserve(label="b")
    r3 = b.reserve(label="c")
    assert r1.granted == pytest.approx(0.5)
    assert r2.granted == pytest.approx(0.25)
    assert r3.granted == pytest.approx(0.125)
    assert r1.granted + r2.granted + r3.granted < 1.0
    assert b.outstanding() == pytest.approx(0.875)
    # releasing returns the grant to the pool
    r1.release()
    r1.release()  # idempotent
    assert b.outstanding() == pytest.approx(0.375)
    r4 = b.reserve()
    assert r4.granted == pytest.approx((1.0 - 0.375) / 2)


def test_admission_considers_outstanding_reservations():
    """A wave near the ceiling is not blindly admitted: reserved (in-flight)
    credits count against admission before any usage event lands."""
    b = Budget(total=1.0)
    grants = []
    while True:
        try:
            grants.append(b.reserve())
        except BudgetExceeded:
            break
    assert sum(g.granted for g in grants) < 1.0  # hard ceiling holds
    with pytest.raises(BudgetExceeded):
        b.ensure_available(label="late")
    # once the wave releases (spending nothing), admission reopens
    for g in grants:
        g.release()
    b.ensure_available(label="after-release")


def test_reserve_refused_when_nearly_spent():
    b = Budget(total=1.0)
    b.tap("s1")(usage_event(0.995 * NANO))
    with pytest.raises(BudgetExceeded) as exc_info:
        b.reserve(label="dust")
    assert exc_info.value.spent == pytest.approx(0.995)


def test_unlimited_budget_reservation_is_uncapped():
    b = Budget(total=None)
    r = b.reserve()
    assert r.granted is None
    assert r.limits() is None
    r.release()


# --------------------------------------------------- engine: hard ceiling


@pytest.mark.asyncio
async def test_budget_ceiling_stops_new_agents_mid_workflow(make_wf):
    """Agent 1 spends past the ceiling; agent 2 is refused at admission."""
    rt = FakeRuntime([[Turn(text="done", events=[usage_event(0.6 * NANO)])]])
    budget = Budget(total=0.5)
    async with make_wf(runtime=rt, budget=budget) as wf:
        assert await wf.agent("spendy", label="one") == "done"
        assert budget.spent() == pytest.approx(0.6)
        with pytest.raises(BudgetExceeded):
            await wf.agent("refused", label="two")
    assert len(rt.created) == 1  # the second session was never created
    records = wf.journal.records()
    assert len(records) == 1  # admission refusal is not an agent record
    assert records[0].credits == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_parallel_wave_degrades_on_budget_exhaustion(make_wf):
    """A capped wave yields Nones for refused branches instead of crashing."""
    budget = Budget(total=0.5)
    budget.tap("prior")(usage_event(0.6 * NANO))  # ceiling already blown
    rt = FakeRuntime()
    async with make_wf(runtime=rt, budget=budget) as wf:
        results = await wf.parallel(
            [lambda: wf.agent("a", label="a"), lambda: wf.agent("b", label="b")]
        )
    assert results == [None, None]
    assert rt.created == []


@pytest.mark.asyncio
async def test_session_limits_passed_to_new_sessions(make_wf):
    """Each session's cap is half the uncommitted remainder at admission —
    never the full remainder, so N concurrent caps can't sum past the total."""
    rt = FakeRuntime([[Turn(text="x", events=[usage_event(25.0 * NANO)])], [Turn(text="y")]])
    async with make_wf(runtime=rt, budget=Budget(total=100.0)) as wf:
        await wf.agent("first", label="a")
        await wf.agent("second", label="b")
    assert rt.create_kwargs[0]["session_limits"] == {"max_ai_credits": pytest.approx(50.0)}
    # first agent spent 25 and released its grant: (100 - 25) / 2
    assert rt.create_kwargs[1]["session_limits"] == {"max_ai_credits": pytest.approx(37.5)}


@pytest.mark.asyncio
async def test_sub_floor_grants_clamp_to_provider_minimum(make_wf):
    """Grants under the provider's 30-credit minimum clamp up to it: the API
    rejects smaller caps, and sending none at all lets an already-running
    session ignore the budget entirely."""
    rt = FakeRuntime([[Turn(text="x")]])
    async with make_wf(runtime=rt, budget=Budget(total=40.0)) as wf:
        await wf.agent("go", label="a")  # grant would be 20 < 30
    assert rt.create_kwargs[0]["session_limits"] == {"max_ai_credits": pytest.approx(30.0)}


class _SlowSession(FakeSession):
    """A session whose turn actually suspends, so wave branches overlap."""

    async def send_and_wait(self, prompt, *, timeout: float = 60.0):
        await asyncio.sleep(0.02)
        return await super().send_and_wait(prompt, timeout=timeout)


@pytest.mark.asyncio
async def test_concurrent_wave_caps_sum_under_remaining(make_wf):
    """A truly concurrent wave's session caps together never commit more than
    the remaining budget (regression: each used to get the FULL remainder,
    for a worst case of total + N x remaining)."""
    rt = FakeRuntime([_SlowSession([Turn(text=t)]) for t in ("a", "b", "c")])
    async with make_wf(runtime=rt, budget=Budget(total=2000.0)) as wf:
        await wf.parallel(
            [
                lambda: wf.agent("wave-1", label="w1"),
                lambda: wf.agent("wave-2", label="w2"),
                lambda: wf.agent("wave-3", label="w3"),
            ]
        )
    caps = [kw["session_limits"]["max_ai_credits"] for kw in rt.create_kwargs]
    assert len(caps) == 3
    assert sum(caps) < 2000.0  # grants shrink geometrically: 1000, 500, 250
    assert sorted(caps, reverse=True) == [
        pytest.approx(1000.0),
        pytest.approx(500.0),
        pytest.approx(250.0),
    ]


@pytest.mark.asyncio
async def test_no_ceiling_means_no_session_limits(make_wf):
    rt = FakeRuntime([[Turn(text="x")]])
    async with make_wf(runtime=rt, budget=Budget(total=None)) as wf:
        await wf.agent("free", label="a")
    assert "session_limits" not in rt.create_kwargs[0]


@pytest.mark.asyncio
async def test_replayed_results_bypass_budget(make_wf, tmp_path):
    run_dir = tmp_path / "budget-replay"
    rt1 = FakeRuntime([[Turn(text="cached")]])
    async with make_wf(runtime=rt1, run_dir=run_dir) as wf1:
        await wf1.agent("p", label="a")

    exhausted = Budget(total=0.5)
    exhausted.tap("prior")(usage_event(9.0 * NANO))
    rt2 = FakeRuntime()
    async with make_wf(runtime=rt2, run_dir=run_dir, resume=True, budget=exhausted) as wf2:
        assert await wf2.agent("p", label="a") == "cached"  # free, so allowed
    assert rt2.created == []
