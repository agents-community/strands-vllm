from __future__ import annotations

from dataclasses import dataclass

from strands.hooks import BeforeToolCallEvent, HookRegistry
from strands.tools import tool
from strands.tools.registry import ToolRegistry

from strands_vllm import VLLMToolValidationHooks


@tool
def add(a: int, b: int) -> int:
    return a + b


@dataclass
class _FakeAgent:
    tool_registry: ToolRegistry

    @property
    def tool_names(self) -> list[str]:
        return list(self.tool_registry.get_all_tools_config().keys())


def test_tool_validation_hooks_cancels_unknown_tool():
    tool_registry = ToolRegistry()
    tool_registry.process_tools([add])
    agent = _FakeAgent(tool_registry=tool_registry)

    hooks = VLLMToolValidationHooks()
    registry = HookRegistry()
    hooks.register_hooks(registry)

    event = BeforeToolCallEvent(
        agent=agent,  # type: ignore[arg-type]
        selected_tool=None,
        tool_use={"toolUseId": "t1", "name": "print", "input": {"s": "hi"}},
        invocation_state={},
    )

    event, _ = registry.invoke_callbacks(event)
    assert event.cancel_tool


def test_tool_validation_hooks_cancels_missing_required_args():
    tool_registry = ToolRegistry()
    tool_registry.process_tools([add])
    agent = _FakeAgent(tool_registry=tool_registry)

    selected_tool = tool_registry.registry["add"]
    hooks = VLLMToolValidationHooks()
    registry = HookRegistry()
    hooks.register_hooks(registry)

    event = BeforeToolCallEvent(
        agent=agent,  # type: ignore[arg-type]
        selected_tool=selected_tool,
        tool_use={"toolUseId": "t1", "name": "add", "input": {"a": 1}},
        invocation_state={},
    )

    event, _ = registry.invoke_callbacks(event)
    assert event.cancel_tool


def test_tool_validation_hooks_cancels_invalid_json_string_input():
    tool_registry = ToolRegistry()
    tool_registry.process_tools([add])
    agent = _FakeAgent(tool_registry=tool_registry)

    selected_tool = tool_registry.registry["add"]
    hooks = VLLMToolValidationHooks()
    registry = HookRegistry()
    hooks.register_hooks(registry)

    event = BeforeToolCallEvent(
        agent=agent,  # type: ignore[arg-type]
        selected_tool=selected_tool,
        tool_use={"toolUseId": "t1", "name": "add", "input": "{not json"},
        invocation_state={},
    )

    event, _ = registry.invoke_callbacks(event)
    assert event.cancel_tool

