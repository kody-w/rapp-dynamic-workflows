"""CopilotRuntime behavior through the client_factory seam (no CLI spawn)."""

from __future__ import annotations

import pytest

from rdw.runtime import CopilotRuntime, default_concurrency

from conftest import FakeClient, FakeSession, Turn


@pytest.mark.asyncio
async def test_client_starts_lazily_on_first_session():
    client = FakeClient([[Turn(text="hi")]])
    rt = CopilotRuntime(client_factory=lambda: client)
    assert not client.started  # constructing the runtime spawns nothing
    session = await rt.create_session(model="m")
    assert client.started
    assert isinstance(session, FakeSession)
    await rt.close()
    assert client.stopped


@pytest.mark.asyncio
async def test_client_started_exactly_once():
    client = FakeClient()
    starts = []
    orig = client.start

    async def counting_start():
        starts.append(1)
        await orig()

    client.start = counting_start
    rt = CopilotRuntime(client_factory=lambda: client)
    await rt.create_session()
    await rt.create_session()
    assert starts == [1]
    await rt.close()


@pytest.mark.asyncio
async def test_approve_all_permissions_injected_by_default():
    from copilot import PermissionHandler

    client = FakeClient()
    rt = CopilotRuntime(client_factory=lambda: client)
    await rt.create_session()
    assert client.create_kwargs[0]["on_permission_request"] is PermissionHandler.approve_all
    # an explicit handler is respected, not overwritten
    sentinel = object()
    await rt.create_session(on_permission_request=sentinel)
    assert client.create_kwargs[1]["on_permission_request"] is sentinel
    await rt.close()


@pytest.mark.asyncio
async def test_closed_runtime_refuses_new_sessions():
    client = FakeClient()
    rt = CopilotRuntime(client_factory=lambda: client)
    await rt.create_session()
    await rt.close()
    with pytest.raises(RuntimeError):
        await rt.create_session()


@pytest.mark.asyncio
async def test_close_without_start_is_clean():
    client = FakeClient()
    rt = CopilotRuntime(client_factory=lambda: client)
    await rt.close()
    assert not client.started and not client.stopped


def test_default_concurrency_bounds():
    n = default_concurrency()
    assert 1 <= n <= 16


def test_real_client_construction_is_blocked_in_tests():
    """The autouse guard makes any accidental real-client path fail loudly."""
    rt = CopilotRuntime()
    with pytest.raises(AssertionError, match="real CopilotClient"):
        rt._default_factory()
