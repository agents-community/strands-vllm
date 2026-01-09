"""Tool call validation helpers for OpenAI-compatible vLLM tool calling.

This module provides an agent-level guard for tool execution when running against vLLM
or other OpenAI-compatible servers that may post-process model output into `tool_calls`.

The goal is not to replace Strands' tool system, but to make the failure mode explicit:
- Unknown tool names become a deterministic tool error message (useful for RL feedback).
- Basic input-shape checks run before tool execution (missing required keys; unknown keys
  when the schema disallows additional properties).

Credit / reference:
- Inspired by the "parse/validate and feed errors back to the model" approach used in
  `horizon-rl/strands-sglang`:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
from strands.types.tools import ToolSpec


def _schema_from_tool_spec(tool_spec: ToolSpec) -> dict[str, Any]:
    input_schema = tool_spec.get("inputSchema", {})
    if isinstance(input_schema, dict) and "json" in input_schema and isinstance(input_schema["json"], dict):
        return input_schema["json"]
    return input_schema if isinstance(input_schema, dict) else {}


def _validate_tool_input(tool_name: str, tool_input: Any, tool_spec: ToolSpec) -> str | None:
    if tool_input is None:
        tool_input = {}
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            return f"Error: tool_name=<{tool_name}> | tool input is not valid JSON"
    if not isinstance(tool_input, dict):
        return f"Error: tool_name=<{tool_name}> | tool input must be an object"

    schema = _schema_from_tool_spec(tool_spec)
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [k for k in required if isinstance(k, str) and k not in tool_input]
        if missing:
            missing_str = ", ".join(missing)
            return f"Error: tool_name=<{tool_name}> | missing required argument(s): {missing_str}"

    additional = schema.get("additionalProperties", True)
    properties = schema.get("properties", {})
    if additional is False and isinstance(properties, dict):
        unknown = [k for k in tool_input.keys() if k not in properties]
        if unknown:
            unknown_str = ", ".join(sorted(map(str, unknown)))
            return f"Error: tool_name=<{tool_name}> | unknown argument(s): {unknown_str}"

    return None


def _format_allowed_tools(tool_names: Iterable[str], *, max_items: int) -> str:
    names = [n for n in tool_names if isinstance(n, str)]
    if not names:
        return "[]"
    shown = names[:max_items]
    suffix = "" if len(shown) == len(names) else f", ... (+{len(names) - len(shown)} more)"
    return "[" + ", ".join(shown) + suffix + "]"


@dataclass(slots=True)
class VLLMToolValidationHooks(HookProvider):
    """Hook provider that guards tool execution with a client-side allowlist + schema checks."""

    include_allowed_tools_in_errors: bool = True
    max_allowed_tools_in_error: int = 25
    validate_input_shape: bool = True

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before_tool_call)

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        tool_name = str(event.tool_use.get("name", ""))
        allowed_names = event.agent.tool_names

        if event.selected_tool is None:
            allowed = ""
            if self.include_allowed_tools_in_errors:
                allowed = f" | allowed_tools={_format_allowed_tools(allowed_names, max_items=self.max_allowed_tools_in_error)}"
            event.cancel_tool = f"Error: unknown tool: {tool_name}{allowed}"
            return

        if not self.validate_input_shape:
            return

        tool_spec = event.selected_tool.tool_spec
        error = _validate_tool_input(tool_name, event.tool_use.get("input"), tool_spec)
        if error:
            event.cancel_tool = error

