"""Structured output via the submit-tool pattern.

The Copilot SDK has **no** native ``response_format`` (grep-confirmed against
the installed source), so schema-forced output is built from three real
primitives:

1. A custom ``submit_result`` tool whose JSON schema is compiled from the
   caller's Pydantic model (``define_tool(params_type=Model)``) or taken
   verbatim from a raw JSON-schema dict. Invalid arguments are automatically
   bounced back to the model as ``"Invalid tool arguments"`` failures by the
   SDK (tools.py), which is a free in-band validation-retry loop.
2. A ``system_message`` append instruction telling the model its turn only
   counts once ``submit_result`` has been called.
3. Tool-choice pressure: when the agent needs no other tools, the session's
   ``available_tools`` allowlist is narrowed to just ``submit_result``.

The engine adds the outer guard: if the session goes idle without a submit,
it nudges the model up to N times, then fails the agent with
:class:`~rdw.errors.AgentSchemaError`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from copilot.tools import Tool, ToolInvocation, ToolResult

SUBMIT_TOOL_NAME = "submit_result"

SchemaSpec = type[BaseModel] | dict[str, Any]
"""A structured-output request: a Pydantic model class or a raw JSON schema."""

SUBMIT_INSTRUCTION = (
    "\n\n## Result submission (mandatory)\n"
    f"You MUST call the `{SUBMIT_TOOL_NAME}` tool exactly once with your final "
    "answer before ending your turn. Any prose you write outside that tool "
    "call is discarded; only the arguments you pass to "
    f"`{SUBMIT_TOOL_NAME}` count as your result. Fill every required field. "
    "If the tool reports invalid arguments, correct them and call it again."
)

NUDGE_PROMPT = (
    f"You ended your turn without calling `{SUBMIT_TOOL_NAME}`. Your work is "
    f"not recorded until you do. Call `{SUBMIT_TOOL_NAME}` now with your "
    "final answer, filling every required field of its schema."
)


@dataclass
class SubmitCapture:
    """Mutable cell the submit tool handler writes into.

    ``value`` is a validated Pydantic instance (model schemas) or a plain dict
    (raw JSON-schema dicts). ``called`` distinguishes "submitted None-ish"
    from "never submitted".
    """

    value: Any = None
    called: bool = False
    attempts: int = field(default=0)


def is_model_schema(schema: SchemaSpec | None) -> bool:
    """True when ``schema`` is a Pydantic model class."""
    return isinstance(schema, type) and issubclass(schema, BaseModel)


def build_submit_tool(schema: SchemaSpec, capture: SubmitCapture) -> Tool:
    """Compile ``schema`` into a ``submit_result`` SDK Tool.

    For Pydantic models the SDK's own ``define_tool`` machinery is reused via a
    directly-constructed :class:`Tool` with the model's generated JSON schema;
    the handler validates with ``model_validate`` and returns the SDK-shaped
    ``Invalid tool arguments`` failure on ``ValidationError`` so the model
    retries in-band, exactly like a decorated tool would.

    For raw dict schemas the dict is attached verbatim as the tool's
    ``parameters``; the handler checks top-level ``required`` keys (the only
    validation possible without a full JSON-schema validator dependency).
    """
    if is_model_schema(schema):
        model_cls: type[BaseModel] = schema  # type: ignore[assignment]

        async def model_handler(invocation: ToolInvocation) -> ToolResult:
            from pydantic import ValidationError

            capture.attempts += 1
            args = invocation.arguments or {}
            try:
                instance = model_cls.model_validate(args)
            except ValidationError as exc:
                parts = []
                for err in exc.errors():
                    loc = ".".join(map(str, err["loc"]))
                    parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
                return ToolResult(
                    text_result_for_llm="Invalid tool arguments:\n" + "\n".join(parts),
                    result_type="failure",
                    error=str(exc),
                )
            capture.value = instance
            capture.called = True
            return ToolResult(
                text_result_for_llm="Result recorded. Your task is complete; end your turn.",
                result_type="success",
            )

        return Tool(
            name=SUBMIT_TOOL_NAME,
            description=(
                "Submit your final structured result. Calling this tool with "
                "valid arguments is the only way to complete the task."
            ),
            parameters=model_cls.model_json_schema(),
            handler=model_handler,
            skip_permission=True,
        )

    if isinstance(schema, dict):
        raw_schema: dict[str, Any] = schema

        async def dict_handler(invocation: ToolInvocation) -> ToolResult:
            capture.attempts += 1
            args = invocation.arguments
            if not isinstance(args, dict):
                return ToolResult(
                    text_result_for_llm=(
                        "Invalid tool arguments: expected a JSON object matching the schema."
                    ),
                    result_type="failure",
                )
            missing = [
                key
                for key in raw_schema.get("required", [])
                if isinstance(key, str) and key not in args
            ]
            if missing:
                return ToolResult(
                    text_result_for_llm=(
                        "Invalid tool arguments: missing required field(s): "
                        + ", ".join(missing)
                    ),
                    result_type="failure",
                )
            capture.value = args
            capture.called = True
            return ToolResult(
                text_result_for_llm="Result recorded. Your task is complete; end your turn.",
                result_type="success",
            )

        return Tool(
            name=SUBMIT_TOOL_NAME,
            description=(
                "Submit your final structured result. Calling this tool with "
                "valid arguments is the only way to complete the task."
            ),
            parameters=raw_schema,
            handler=dict_handler,
            skip_permission=True,
        )

    raise TypeError(
        f"schema must be a Pydantic model class or a JSON-schema dict, got {type(schema)!r}"
    )


def schema_fingerprint(schema: SchemaSpec | None) -> str | None:
    """Stable identity of a schema for journal fingerprinting.

    A model class hashes its qualified name plus its generated JSON schema, so
    editing a field invalidates cached results; a raw dict hashes its
    canonical JSON form.
    """
    if schema is None:
        return None
    if is_model_schema(schema):
        model_cls: type[BaseModel] = schema  # type: ignore[assignment]
        body = json.dumps(model_cls.model_json_schema(), sort_keys=True, default=str)
        seed = f"{model_cls.__module__}.{model_cls.__qualname__}:{body}"
    else:
        seed = json.dumps(schema, sort_keys=True, default=str)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def dump_value(value: Any) -> dict[str, Any]:
    """Serialize an agent result into a JSON-safe journal payload."""
    if isinstance(value, BaseModel):
        return {"kind": "model", "value": value.model_dump(mode="json")}
    if isinstance(value, str):
        return {"kind": "text", "value": value}
    return {"kind": "data", "value": value}


def load_value(schema: SchemaSpec | None, payload: dict[str, Any]) -> Any:
    """Rehydrate a journal payload, re-validating against a model schema.

    Re-validation on replay means a schema edit surfaces immediately as a
    validation error (fingerprints would normally miss first, but this is the
    belt to that suspender).
    """
    kind = payload.get("kind")
    value = payload.get("value")
    if kind == "model" and is_model_schema(schema):
        model_cls: type[BaseModel] = schema  # type: ignore[assignment]
        return model_cls.model_validate(value)
    return value
