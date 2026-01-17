#!/usr/bin/env python3
"""Test script for Slime integration pattern with vLLM.

This demonstrates the TITO extraction pattern used in Slime training.

Install THUDM/slime (not the pip 'slime' package):
  pip install git+https://github.com/THUDM/slime.git

Run as pytest:
  pytest tests/integration/test_slime_integration.py -v --vllm-base-url=http://localhost:8000/v1

Run as standalone:
  export VLLM_BASE_URL="http://localhost:8000/v1"
  export VLLM_MODEL_ID="AMead10/Llama-3.2-3B-Instruct-AWQ"
  python tests/integration/test_slime_integration.py

Credit / reference:
- Inspired by the "parse/validate and feed errors back to the model" approach used in
  `horizon-rl/strands-sglang`:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

import asyncio
import os

import pytest
from slime.utils.types import Sample  # type: ignore[import-untyped]
from strands import Agent

from strands_vllm import TokenManager, VLLMModel, VLLMTokenRecorder


async def generate_sample(
    prompt: str,
    base_url: str,
    model_id: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Sample:
    """Generate a sample with TITO: tokens captured during generation, no retokenization.

    This follows the Slime generate function pattern from the README.
    """
    # Set up Agent with VLLMModel and VLLMTokenRecorder
    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
        params={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    recorder = VLLMTokenRecorder()
    agent = Agent(
        model=model,
        callback_handler=recorder,
    )

    # Run Agent Loop
    sample = Sample(prompt=prompt)
    try:
        await agent.invoke_async(prompt)
        sample.status = Sample.Status.COMPLETED
    except Exception as e:
        # Always use TRUNCATED instead of ABORTED because Slime doesn't properly
        # handle ABORTED samples in reward processing. See: https://github.com/THUDM/slime/issues/200
        sample.status = Sample.Status.TRUNCATED
        print(f"TRUNCATED: {type(e).__name__}: {e}")

    # TITO: extract trajectory from recorder and TokenManager
    tm = TokenManager()
    for entry in recorder.history:
        pti = entry.get("prompt_token_ids")
        ti = entry.get("token_ids")
        if pti:
            tm.add_prompt(pti)
        if ti:
            tm.add_response(ti)

    if not tm.segments:
        print("⚠️  No token segments captured from vLLM streaming.")
        print("   Make sure your vLLM server supports `return_token_ids` in streaming mode.")
        return sample

    prompt_len = len(tm.segments[0]) if tm.segments else 0
    sample.tokens = tm.token_ids
    sample.loss_mask = tm.loss_mask[prompt_len:]
    sample.rollout_log_probs = tm.logprobs[prompt_len:]
    sample.response_length = len(sample.tokens) - prompt_len

    # Extract response from agent messages (vLLM returns text directly, no tokenizer needed)
    response_text = ""
    for msg in reversed(agent.messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        response_text = block["text"]
                        break
            if response_text:
                break
    sample.response = response_text

    # Cleanup
    recorder.reset()
    agent.cleanup()

    return sample


@pytest.mark.asyncio
async def test_slime_tito_pattern(vllm_base_url: str, vllm_model_id: str) -> None:
    """Test TITO extraction pattern for Slime training."""
    prompt = "What is 17 * 19? Return only the number."

    sample = await generate_sample(
        prompt=prompt,
        base_url=vllm_base_url,
        model_id=vllm_model_id,
        max_tokens=128,
        temperature=0,
    )

    # Verify sample fields are populated
    assert sample.status == Sample.Status.COMPLETED
    assert sample.tokens is not None, "Token IDs not captured from vLLM"
    assert len(sample.tokens) > 0
    assert sample.response_length > 0
    assert sample.loss_mask is not None
    assert len(sample.loss_mask) == sample.response_length
    assert sample.response != ""

    # Verify loss_mask contains only 1s (response tokens)
    assert all(m == 1 for m in sample.loss_mask), "loss_mask should be all 1s for response"


async def main() -> None:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")

    print("=" * 70)
    print("Slime Integration Test (TITO Pattern)")
    print("=" * 70)
    print(f"\nModel:  {model_id}")
    print(f"Server: {base_url}\n")

    prompt = "What is 17 * 19? Return only the number."
    print(f"Prompt: {prompt}\n")

    sample = await generate_sample(
        prompt=prompt,
        base_url=base_url,
        model_id=model_id,
        max_tokens=128,
        temperature=0,
    )

    print("=" * 70)
    print("TITO Data (for Slime training)")
    print("=" * 70)

    if sample.tokens is None:
        print("\n❌ No tokens captured. Check vLLM server configuration.")
        return

    print(f"\nStatus: {sample.status.value}")
    print(f"Total tokens: {len(sample.tokens)}")
    print(f"Prompt tokens: {len(sample.tokens) - sample.response_length}")
    print(f"Response tokens: {sample.response_length}")
    print(f"Response: {sample.response}")

    if sample.loss_mask:
        n_prompt = sum(1 for m in sample.loss_mask if m == 0)
        n_output = sum(1 for m in sample.loss_mask if m == 1)
        print(f"\nLoss mask (response tokens only): {len(sample.loss_mask)} tokens")
        print(f"  Prompt (mask=0): {n_prompt}")
        print(f"  Output (mask=1): {n_output}")

    if sample.rollout_log_probs:
        logprob_count = sum(1 for lp in sample.rollout_log_probs if lp is not None)
        print(f"\nLogprobs: {len(sample.rollout_log_probs)} (non-None: {logprob_count})")

    print("\n" + "=" * 70)
    print("Slime Sample Fields")
    print("=" * 70)
    print(f"  sample.tokens: {len(sample.tokens)} tokens")
    print(f"  sample.loss_mask: {len(sample.loss_mask) if sample.loss_mask else 0} values")
    print(f"  sample.rollout_log_probs: {len(sample.rollout_log_probs) if sample.rollout_log_probs else 0} values")
    print(f"  sample.response_length: {sample.response_length}")
    print(f"  sample.response: {sample.response[:100]}{'...' if len(sample.response) > 100 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
