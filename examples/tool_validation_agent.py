"""Tool-call validation demo for vLLM OpenAI-compatible tool calling.

This example enables `VLLMToolValidationHooks` to guard tool execution when running against
servers that may post-process model outputs into `tool_calls`.

It intentionally prompts the model to call a non-existent tool name ("print") while only
registering a real tool ("calculator"). The hook layer turns unknown/invalid tool calls
into a deterministic tool error result that is fed back to the model.

Credit / reference:
- Inspired by the RL-oriented "tool error feedback" approach used in `horizon-rl/strands-sglang`:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

import os

from strands import Agent
from strands_tools.calculator import calculator

from strands_vllm import VLLMModel, VLLMToolValidationHooks


def main() -> None:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")

    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
    )

    agent = Agent(
        model=model,
        tools=[calculator],
        hooks=[VLLMToolValidationHooks()],
    )

    prompt = (
        "Call the tool named print with JSON arguments {\"s\": \"Hello!\"}.\n"
        "Do not explain.\n"
        "If that tool is unavailable, then use the calculator tool to compute 17 * 19 and return only the number."
    )

    result = agent(prompt)
    print(result)


if __name__ == "__main__":
    main()

