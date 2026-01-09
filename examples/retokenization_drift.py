#!/usr/bin/env python3
"""Retokenization drift demo for vLLM TITO.

Demonstrates: encode(decode(tokens)) != tokens

Why:
- For RL training you must use the exact token IDs produced during rollout.
- Re-tokenizing decoded text can produce a different tokenization (drift).

Credit / reference:
- Adapted from `horizon-rl/strands-sglang` drift example:
  https://github.com/horizon-rl/strands-sglang

This is an educational demo. It requires a local HuggingFace tokenizer to
decode/re-encode tokens, but the rollout token IDs come from vLLM (server-side).

Install:
  pip install "strands-vllm[drift]" strands-agents-tools

Run:
  python examples/retokenization_drift.py
"""

from __future__ import annotations

import asyncio
import os

from strands import Agent, tool
from strands.handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler
from strands_tools.calculator import calculator as _calculator_impl
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from strands_vllm import TokenManager, VLLMModel, VLLMTokenRecorder


@tool
def calculator(expression: str) -> dict:
    """Evaluate a math expression (thin wrapper around strands-agents-tools calculator)."""
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
        params={"temperature": 0, "max_tokens": 4096},
    )

    recorder = VLLMTokenRecorder()
    printer = PrintingCallbackHandler(verbose_tool_use=True)
    callback = CompositeCallbackHandler(printer, recorder)
    agent = Agent(
        model=model,
        tools=[calculator],
        callback_handler=callback,
        system_prompt=(
            "You are a math expert. Think as long and thoroughly as possible, exploring multiple solution paths "
            "and verifying each step. Use the calculator for all arithmetic."
        ),
    )

    problem = """
    A farmer has 3 fields. Field A is 2.5 acres, Field B is 3.75 acres, Field C is 1.8 acres.

    Crop yields per acre: Wheat=$450, Corn=$380, Soybeans=$520.
    Costs per acre: Wheat=$120, Corn=$95, Soybeans=$150.

    The farmer plants wheat in Field A, corn in Field B, and soybeans in Field C.
    There's also a 15% tax on total profit.

    Calculate: (1) Revenue per field, (2) Cost per field, (3) Profit per field,
    (4) Total profit before tax, (5) Tax amount, (6) Final profit after tax.

    Think through each step very carefully, exploring multiple approaches.
    """
    await agent.invoke_async(problem)

    output_token_ids: list[int] = []
    first_prompt_token_ids: list[int] | None = None
    for entry in recorder.history:
        pti = entry.get("prompt_token_ids")
        if first_prompt_token_ids is None and isinstance(pti, list) and all(isinstance(x, int) for x in pti):
            first_prompt_token_ids = pti
        token_ids = entry.get("token_ids")
        if isinstance(token_ids, list) and all(isinstance(x, int) for x in token_ids):
            output_token_ids.extend(token_ids)

    if not output_token_ids:
        print("\nNo token IDs captured from vLLM streaming.")
        print("Ensure vLLM supports return_token_ids in streaming mode and that strands-vllm is enabled.")
        return

    tm = TokenManager()
    tm.add_response(output_token_ids)

    tito_tokens = tm.token_ids
    decoded = tokenizer.decode(tito_tokens)
    re_encoded = tokenizer.encode(decoded, add_special_tokens=False)

    print(f"\nTITO tokens:  {len(tito_tokens)}")
    print(f"Re-encoded:   {len(re_encoded)}")

    drift_idx = find_drift_index(tito_tokens, re_encoded)
    if drift_idx is None:
        print("\nNo drift detected (drift is rare). TITO still captures exact tokens.")
    else:
        print(f"\n>>> DRIFT at index {drift_idx}/{len(tito_tokens)} <<<")
        ctx = 5
        start = max(0, drift_idx - ctx)
        end = min(len(tito_tokens), drift_idx + ctx + 1)

        print(f"\nContext (indices {start}-{end - 1}):")
        print("  Original tokens:")
        for i in range(start, end):
            marker = " -->" if i == drift_idx else "    "
            print(f"  {marker} [{i}] {tito_tokens[i]:6d} -> {repr(tokenizer.decode([tito_tokens[i]]))}")

        print("  Re-encoded tokens:")
        for i in range(start, min(end, len(re_encoded))):
            marker = " -->" if i == drift_idx else "    "
            print(f"  {marker} [{i}] {re_encoded[i]:6d} -> {repr(tokenizer.decode([re_encoded[i]]))}")

        print("\nTITO captures exact tokens - use token_ids directly for RL training.")

    print("\n--- TITO Data ---")
    prompt_len = len(first_prompt_token_ids or [])
    output_len = len(output_token_ids)
    print(f"Prompt tokens (first call): {prompt_len}")
    print(f"Output tokens (all calls):  {output_len}")
    print("Segments: 2 (Prompt + Response)")
    print(f"  Seg 0: {prompt_len:5d} Prompt")
    print(f"  Seg 1: {output_len:5d} Response")


if __name__ == "__main__":
    asyncio.run(main())

