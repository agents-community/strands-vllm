#!/usr/bin/env python3
"""RL-oriented token-in/token-out rollout demo for vLLM.

This example builds an RL-friendly trajectory from vLLM-provided token IDs:
- prompt tokens use loss_mask=0
- model output tokens use loss_mask=1

It also demonstrates retokenization drift: encoding decoded text can differ from the original token IDs,
so for RL training you should use the server-provided token IDs captured during rollout.

Install:
  pip install "strands-vllm[drift]" strands-agents-tools

Run:
  export VLLM_BASE_URL="http://localhost:8000/v1"
  export VLLM_MODEL_ID="AMead10/Llama-3.2-3B-Instruct-AWQ"
  python examples/rl_rollout_tito.py

Credit / reference:
- Inspired by the TITO/RL design in `horizon-rl/strands-sglang`:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

import asyncio
import os

from strands import Agent, tool
from strands.handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler
from strands_tools.calculator import calculator as _calculator_impl

from strands_vllm import TokenManager, VLLMModel, VLLMTokenRecorder, VLLMToolValidationHooks


@tool
def calculator(expression: str) -> dict:
    return _calculator_impl(expression=expression)


def find_drift_index(original: list[int], re_encoded: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(original, re_encoded)):
        if a != b:
            return i
    if len(original) != len(re_encoded):
        return min(len(original), len(re_encoded))
    return None


async def main() -> None:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")

    print(f"Model:  {model_id}")
    print(f"Server: {base_url}\n")

    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
        params={"temperature": 0},
    )

    recorder = VLLMTokenRecorder()
    printer = PrintingCallbackHandler(verbose_tool_use=True)
    callback = CompositeCallbackHandler(printer, recorder)

    agent = Agent(
        model=model,
        tools=[calculator],
        callback_handler=callback,
        hooks=[VLLMToolValidationHooks()],
        system_prompt=(
            "You are a careful assistant. Use tools when explicitly asked. "
            "If a tool is requested, call exactly that tool with a valid JSON object. "
            "Never request more than one tool call in a single assistant message."
        ),
    )

    prompt_1 = 'Call the tool named "print" with JSON arguments {"s": "Hello!"}.'
    await agent.invoke_async(prompt_1)

    tool_error_text: str | None = None
    for message in agent.messages:
        if message.get("role") != "user":
            continue
        for content in message.get("content", []):
            if not isinstance(content, dict) or "toolResult" not in content:
                continue
            tool_result = content["toolResult"]
            if not isinstance(tool_result, dict) or tool_result.get("status") != "error":
                continue
            for block in tool_result.get("content", []):
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    tool_error_text = block["text"]
                    break
        if tool_error_text:
            break

    if tool_error_text:
        print("\n--- Tool error feedback (hook) ---")
        print(tool_error_text)

    prompt_2 = (
        "Compute: 1-2+3-4+5-...+99-100. Simplify first.\n"
        "Then verify by calling the calculator tool with a closed-form expression.\n"
        "Return only the final integer."
    )
    await agent.invoke_async(prompt_2)

    tm = TokenManager()
    for entry in recorder.history:
        pti = entry.get("prompt_token_ids")
        ti = entry.get("token_ids")
        if isinstance(pti, list) and all(isinstance(x, int) for x in pti):
            tm.add_prompt(pti)
        if isinstance(ti, list) and all(isinstance(x, int) for x in ti):
            tm.add_response(ti)

    if len(tm) == 0:
        print("\nNo token IDs captured from vLLM streaming.")
        print("Ensure vLLM supports return_token_ids in streaming mode and that strands-vllm is enabled.")
        return

    n_prompt = sum(1 for m in tm.loss_mask if m == 0)
    n_output = sum(1 for m in tm.loss_mask if m == 1)
    print("\n--- RL Trajectory (from vLLM token IDs) ---")
    print(f"Total tokens:   {len(tm)}")
    print(f"Prompt tokens:  {n_prompt} (loss_mask=0)")
    print(f"Output tokens:  {n_output} (loss_mask=1)")
    print(f"Segments:       {len(tm.segments)}")

    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except Exception:
        print("\nRetokenization check skipped (install transformers):")
        print('  pip install "strands-vllm[drift]"')
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tito_tokens = tm.token_ids
    decoded = tokenizer.decode(tito_tokens)
    re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

    print("\n--- Retokenization check ---")
    print(f"TITO tokens:  {len(tito_tokens)}")
    print(f"Re-encoded:   {len(re_encoded)}")

    drift_idx = find_drift_index(tito_tokens, re_encoded)
    if drift_idx is None:
        print("No drift detected (drift is rare). Use TITO anyway for RL correctness.")
    else:
        print(f"DRIFT at index {drift_idx}/{len(tito_tokens)}")
        print("Use the original vLLM token IDs for RL training.")


if __name__ == "__main__":
    asyncio.run(main())

