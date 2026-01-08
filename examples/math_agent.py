#!/usr/bin/env python3
"""Math agent example with vLLM + streaming token IDs (TITO).

This mirrors the structure of `strands-sglang/examples/math_agent.py`, but uses vLLM's
OpenAI-compatible chat API and captures token IDs from the streaming response.

Credit / reference:
- Adapted from `horizon-rl/strands-sglang` examples:
  https://github.com/horizon-rl/strands-sglang

Requirements:
  - vLLM server running with token IDs enabled:
      vllm serve <MODEL_ID> ... --enable-auto-tool-choice --tool-call-parser llama3_json \
        --chat-template examples/tool_chat_template_llama3.2_json.jinja
  - pip install strands-agents-tools

Run:
  python examples/math_agent.py
"""

from __future__ import annotations

import asyncio
import json
import os

from strands import Agent, tool
from strands_tools.calculator import calculator as _calculator_impl

from strands_vllm import TokenManager, VLLMModel, VLLMTokenRecorder


@tool
def calculator(expression: str) -> dict:
    """Evaluate a math expression (thin wrapper around strands-agents-tools calculator).

    This wrapper intentionally exposes a *small schema* (expression only) so models
    don't send invalid placeholder values for optional parameters.
    """
    return _calculator_impl(expression=expression)


async def main() -> None:
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    print(f"Model:  {model_id}")
    print(f"Server: {base_url}")

    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
        params={"temperature": 0, "max_tokens": 256},
    )

    recorder = VLLMTokenRecorder()

    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=(
            "You are a helpful math assistant.\n"
            "You MUST use the calculator tool to verify arithmetic results.\n"
            "Calculator tool rule: provide ONLY one field: expression (a single string).\n"
            "Calculator tool rule: NEVER include '...' or '\\\\dots' or any ellipsis-like shorthand in expression.\n"
            "If the user gives a series with '...'/'\\\\dots', simplify it to a closed form first, then call calculator\n"
            "only on the simplified expression (e.g. '-50', '17*19', '123+456').\n"
            "After the tool returns, provide a short final answer."
        ),
        callback_handler=recorder,
    )

    print("\n" + "=" * 60)
    print("Math 500 Example")
    print("=" * 60)

    math_500_problem = (
        r"Compute: $1-2+3-4+5- \dots +99-100$." "\n"
        "First simplify it to a single integer.\n"
        "Then verify by calling calculator with expression \"-50\" (do NOT pass the original series).\n"
        "Finally, print the final integer answer."
    )
    print(f"\n[Input Problem]: {math_500_problem}")

    recorder.reset()
    await agent.invoke_async(math_500_problem)
    print(f"\n[Output Trajectory]: {json.dumps(agent.messages, indent=2)}")

    print("\n" + "-" * 40)
    print("TITO Data (from vLLM token IDs)")
    print("-" * 40)

    if not recorder.prompt_token_ids or not recorder.token_ids:
        print("No token IDs captured from vLLM streaming.")
        print(
            "Make sure your vLLM server supports `return_token_ids` in streaming mode and that "
            "`return_token_ids=True` is set on VLLMModel."
        )
        return

    tm = TokenManager()
    tm.add_prompt(recorder.prompt_token_ids)
    tm.add_response(recorder.token_ids)

    token_ids = tm.token_ids
    output_mask = tm.loss_mask
    n_output = sum(output_mask)
    n_prompt = len(output_mask) - n_output

    print(f"Total tokens: {len(token_ids)}")
    print(f"Prompt tokens: {n_prompt} (loss_mask=False)")
    print(f"Response tokens: {n_output} (loss_mask=True)")

    print(f"Segments: {len(tm.segment_info)} (Segment 0 = prompt, Segment 1 = response)")
    for i, (is_output, length) in enumerate(tm.segment_info):
        seg_type = "Response" if is_output else "Prompt"
        print(f"  Segment {i}: {seg_type} ({length} tokens)")

    print("\n" + "=" * 60)
    print("Multi-turn (token lengths per turn)")
    print("=" * 60)
    print(
        "Note: For OpenAI-style chat, vLLM's `prompt_token_ids` is the tokenization of the *entire* prompt\n"
        "at that turn (including history + tool results). For RL, you typically capture per-turn rollouts.\n"
    )

    questions = [
        "Compute 17 * 19. Call calculator with expression \"17*19\" and return the result.",
        "Compute 123 + 456. Call calculator with expression \"123+456\" and return the result.",
    ]
    for q in questions:
        before = len(agent.messages)
        recorder.reset()
        await agent.invoke_async(q)
        new_messages = agent.messages[before:]
        pti_len = len(recorder.prompt_token_ids or [])
        ti_len = len(recorder.token_ids or [])
        print(f"- prompt_token_ids: {pti_len:5d} | token_ids: {ti_len:4d} | question: {q}")

        final_text = None
        for msg in reversed(new_messages):
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []):
                if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
                    final_text = block["text"].strip()
                    break
            if final_text:
                break
        if final_text:
            print(f"  answer: {final_text}")


if __name__ == "__main__":
    asyncio.run(main())

