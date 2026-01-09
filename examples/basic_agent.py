"""Basic vLLM hello-world example.

Credit / reference:
- Inspired by `horizon-rl/strands-sglang` community-provider style:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

import os

from strands import Agent
from strands.handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler

from strands_vllm import VLLMModel, VLLMTokenRecorder


def _summ(values: list[int] | None, head: int = 16, tail: int = 16) -> str:
    if not values:
        return "None"
    if len(values) <= head + tail:
        return str(values)
    return f"len={len(values)} head={values[:head]} tail={values[-tail:]}"


def main() -> None:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")

    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
        disable_tools=True,
        params={"temperature": 0, "max_tokens": 128},
    )

    recorder = VLLMTokenRecorder()
    printer = PrintingCallbackHandler(verbose_tool_use=True)
    callback = CompositeCallbackHandler(printer, recorder)
    agent = Agent(model=model, callback_handler=callback)

    result = agent('Reply with exactly this text and nothing else: "Hello!"')
    print("\nFinal result:\n")
    print(result)

    if recorder.prompt_token_ids and recorder.token_ids:
        print("\nCaptured token IDs:")
        print("  prompt_token_ids:", _summ(recorder.prompt_token_ids))
        print("  token_ids:", _summ(recorder.token_ids))


if __name__ == "__main__":
    main()

