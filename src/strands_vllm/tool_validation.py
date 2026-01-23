"""Optional tool validation hooks for RL training feedback.

Strands SDK already handles unknown tools and malformed JSON gracefully. This module
adds RL-friendly enhancements:

- Unknown tool errors include the list of allowed tools (helps model learn valid tools)
- Schema validation catches missing/extra arguments before tool execution

Usage:
    agent = Agent(model=model, tools=[...], hooks=[VLLMToolValidationHooks()])
"""

from __future__ import annotations

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
    """Validate tool input against schema. Returns error message or None if valid."""
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        return f"Error: tool_name=<{tool_name}> | tool input must be an object"

    schema = _schema_from_tool_spec(tool_spec)

    # Check required arguments
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [k for k in required if isinstance(k, str) and k not in tool_input]
        if missing:
            return f"Error: tool_name=<{tool_name}> | missing required argument(s): {', '.join(missing)}"

    # Check for unknown arguments (only if schema disallows additional properties)
    if schema.get("additionalProperties") is False:
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            unknown = [k for k in tool_input if k not in properties]
            if unknown:
                return f"Error: tool_name=<{tool_name}> | unknown argument(s): {', '.join(sorted(map(str, unknown)))}"

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
    """Hook provider for RL-friendly tool validation.

    Enhances Strands' default tool handling with:
    - Unknown tool errors that list allowed tools (helps RL training)
    - Schema validation (missing required args, unknown args)
    """

    include_allowed_tools_in_errors: bool = True
    max_allowed_tools_in_error: int = 25
    validate_input_shape: bool = True

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.before_tool_call)

    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        tool_name = str(event.tool_use.get("name", ""))

        if event.selected_tool is None:
            allowed = ""
            if self.include_allowed_tools_in_errors:
                allowed = f" | allowed_tools={_format_allowed_tools(event.agent.tool_names, max_items=self.max_allowed_tools_in_error)}"
            event.cancel_tool = f"Error: unknown tool: {tool_name}{allowed}"
            return

        if not self.validate_input_shape:
            return

        error = _validate_tool_input(tool_name, event.tool_use.get("input"), event.selected_tool.tool_spec)
        if error:
            event.cancel_tool = error
