"""Schema forcing: the submit-tool pattern, validation retry, and the
never-submitted guard."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from rdw.errors import AgentSchemaError
from rdw.schema import (
    NUDGE_PROMPT,
    SUBMIT_INSTRUCTION,
    SUBMIT_TOOL_NAME,
    dump_value,
    load_value,
    schema_fingerprint,
)

from conftest import FakeRuntime, Turn


class Verdict(BaseModel):
    approve: bool
    summary: str
    score: float = Field(ge=0.0, le=10.0, default=5.0)


VALID = {"approve": True, "summary": "looks good", "score": 8.0}
INVALID_TYPE = {"approve": "not-a-bool-at-all", "summary": 42, "score": 99.0}
RAW_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["title", "count"],
}


@pytest.mark.asyncio
async def test_pydantic_schema_returns_validated_instance(make_wf):
    rt = FakeRuntime([[Turn(submit=[VALID])]])
    async with make_wf(runtime=rt) as wf:
        result = await wf.agent("review this", schema=Verdict, label="rev")
    assert isinstance(result, Verdict)
    assert result.approve is True and result.summary == "looks good"


@pytest.mark.asyncio
async def test_schema_session_wiring(make_wf):
    """schema=... installs submit tool + system append + tool-choice pressure."""
    rt = FakeRuntime([[Turn(submit=[VALID])]])
    async with make_wf(runtime=rt) as wf:
        await wf.agent("review", schema=Verdict)
    kwargs = rt.create_kwargs[0]
    assert kwargs["available_tools"] == [SUBMIT_TOOL_NAME]
    assert kwargs["system_message"] == {"mode": "append", "content": SUBMIT_INSTRUCTION}
    tool_names = [t.name for t in kwargs["tools"]]
    assert tool_names == [SUBMIT_TOOL_NAME]


@pytest.mark.asyncio
async def test_text_agent_has_no_submit_machinery(make_wf):
    rt = FakeRuntime([[Turn(text="plain answer")]])
    async with make_wf(runtime=rt) as wf:
        result = await wf.agent("just answer")
    assert result == "plain answer"
    kwargs = rt.create_kwargs[0]
    assert "tools" not in kwargs
    assert "available_tools" not in kwargs
    assert "system_message" not in kwargs


@pytest.mark.asyncio
async def test_extra_tools_disable_allowlist_narrowing(make_wf):
    from copilot.tools import Tool

    extra = Tool(name="extra_tool", description="x", handler=None)
    rt = FakeRuntime([[Turn(submit=[VALID])]])
    async with make_wf(runtime=rt) as wf:
        await wf.agent("go", schema=Verdict, tools=[extra])
    kwargs = rt.create_kwargs[0]
    assert "available_tools" not in kwargs  # agent needs its other tools
    assert {t.name for t in kwargs["tools"]} == {"extra_tool", SUBMIT_TOOL_NAME}


@pytest.mark.asyncio
async def test_explore_keeps_builtin_catalog_with_schema(make_wf):
    """explore=True: schema forcing without narrowing, so the agent can
    still reach bash/view/rg to investigate before submitting."""
    rt = FakeRuntime([[Turn(submit=[VALID])]])
    async with make_wf(runtime=rt) as wf:
        result = await wf.agent("investigate then report", schema=Verdict, explore=True)
    assert isinstance(result, Verdict)
    kwargs = rt.create_kwargs[0]
    assert "available_tools" not in kwargs
    assert {t.name for t in kwargs["tools"]} == {SUBMIT_TOOL_NAME}


@pytest.mark.asyncio
async def test_validation_retry_in_band(make_wf):
    """Invalid args bounce back as failure; the model retries and succeeds."""
    rt = FakeRuntime([[Turn(submit=[INVALID_TYPE, VALID])]])
    async with make_wf(runtime=rt) as wf:
        result = await wf.agent("review", schema=Verdict)
    assert isinstance(result, Verdict) and result.score == 8.0
    session = rt.created[0]
    assert [r.result_type for r in session.submit_results] == ["failure", "success"]
    failure = session.submit_results[0]
    assert "Invalid tool arguments" in failure.text_result_for_llm
    assert "approve" in failure.text_result_for_llm  # names the bad field


@pytest.mark.asyncio
async def test_raw_dict_schema_required_key_retry(make_wf):
    rt = FakeRuntime([[Turn(submit=[{"title": "only-title"}, {"title": "t", "count": 3}])]])
    async with make_wf(runtime=rt) as wf:
        result = await wf.agent("extract", schema=RAW_SCHEMA)
    assert result == {"title": "t", "count": 3}
    session = rt.created[0]
    assert [r.result_type for r in session.submit_results] == ["failure", "success"]
    assert "count" in session.submit_results[0].text_result_for_llm


@pytest.mark.asyncio
async def test_nudge_recovers_missing_submit(make_wf):
    """Turn 1 ends without submitting; the nudge turn submits."""
    rt = FakeRuntime([[Turn(text="oops, prose only"), Turn(submit=[VALID])]])
    async with make_wf(runtime=rt) as wf:
        result = await wf.agent("review", schema=Verdict)
    assert isinstance(result, Verdict)
    session = rt.created[0]
    assert session.prompts == ["review", NUDGE_PROMPT]


@pytest.mark.asyncio
async def test_never_submitted_guard(make_wf):
    """Stonewalling through both nudges raises AgentSchemaError."""
    rt = FakeRuntime([[Turn(text="no"), Turn(text="still no"), Turn(text="never")]])
    async with make_wf(runtime=rt) as wf:
        with pytest.raises(AgentSchemaError):
            await wf.agent("review", schema=Verdict, label="stonewall")
    session = rt.created[0]
    assert session.prompts == ["review", NUDGE_PROMPT, NUDGE_PROMPT]
    assert session.disconnected  # session cleaned up even on failure
    # the failure is journaled as an error record
    [record] = wf.journal.records()
    assert record.status == "error" and "submit_result" in (record.error or "")


@pytest.mark.asyncio
async def test_invalid_schema_type_rejected(make_wf):
    async with make_wf() as wf:
        with pytest.raises(TypeError):
            await wf.agent("x", schema="not-a-schema")  # type: ignore[arg-type]


def test_schema_fingerprint_stability():
    assert schema_fingerprint(None) is None
    assert schema_fingerprint(Verdict) == schema_fingerprint(Verdict)
    assert schema_fingerprint(RAW_SCHEMA) == schema_fingerprint(dict(RAW_SCHEMA))
    assert schema_fingerprint(Verdict) != schema_fingerprint(RAW_SCHEMA)


def test_dump_load_roundtrip():
    v = Verdict(approve=False, summary="s", score=1.0)
    payload = dump_value(v)
    assert payload["kind"] == "model"
    restored = load_value(Verdict, payload)
    assert restored == v
    assert load_value(None, dump_value("hello")) == "hello"
    assert load_value(None, dump_value({"a": 1})) == {"a": 1}
