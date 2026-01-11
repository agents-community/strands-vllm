#!/usr/bin/env python3
"""Demonstrate token IDs in OpenTelemetry spans for Agent Lightning.

This example shows how VLLMTokenRecorder automatically adds token IDs to OpenTelemetry
spans, making them available for Agent Lightning to extract during training.

The token IDs are added as span attributes:
- llm.token_count.prompt: Token count for the prompt (standard OpenTelemetry attribute)
- llm.token_count.completion: Token count for the completion (standard OpenTelemetry attribute)
- llm.hosted_vllm.prompt_token_ids: Token ID array for the prompt
- llm.hosted_vllm.response_token_ids: Token ID array for the response

Install:
  pip install "strands-vllm" strands-agents-tools

Run:
  python examples/span_token_ids.py
"""

from __future__ import annotations

import os
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    print("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
    print("This example will still run but won't demonstrate span attributes.\n")

from strands import Agent
from strands.handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler

from strands_vllm import VLLMModel, VLLMTokenRecorder


class SpanAttributeInspector:
    """Inspect span attributes to demonstrate token ID capture."""

    def __init__(self) -> None:
        self.captured_attributes: list[dict[str, Any]] = []

    def inspect_current_span(self) -> None:
        """Inspect the current span and capture its attributes."""
        if not _HAS_OTEL:
            return

        try:
            span = trace.get_current_span()
            if not span.is_recording():
                return

            # Get span attributes (note: OpenTelemetry doesn't expose attributes directly,
            # but we can check if they're set by looking at the span context)
            # In practice, Agent Lightning would read these from the exported spans
            span_context = span.get_span_context()
            if span_context.is_valid:
                self.captured_attributes.append(
                    {
                        "trace_id": format(span_context.trace_id, "032x"),
                        "span_id": format(span_context.span_id, "016x"),
                    }
                )
        except Exception:
            pass


def main() -> None:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model_id = os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ")

    # Setup OpenTelemetry tracer (optional - Strands Agent sets this up automatically)
    if _HAS_OTEL:
        # Create a simple tracer provider for demonstration
        # In practice, Agent Lightning would configure its own tracer
        provider = TracerProvider()
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        print("OpenTelemetry tracer configured for demonstration\n")

    model = VLLMModel(
        base_url=base_url,
        model_id=model_id,
        return_token_ids=True,
        disable_tools=True,
        params={"temperature": 0, "max_tokens": 128},
    )

    # Create recorder - by default, it adds token IDs to spans (add_to_span=True)
    recorder = VLLMTokenRecorder(add_to_span=True)
    printer = PrintingCallbackHandler(verbose_tool_use=False)
    callback = CompositeCallbackHandler(printer, recorder)

    inspector = SpanAttributeInspector()

    # Create agent with tracing enabled
    agent = Agent(model=model, callback_handler=callback)

    print("=" * 70)
    print("Token IDs in OpenTelemetry Spans for Agent Lightning")
    print("=" * 70)
    print(
        "\nWhen VLLMTokenRecorder captures token IDs from vLLM, it automatically\n"
        "adds them as attributes to the current OpenTelemetry span. Agent Lightning\n"
        "can then extract these token IDs from spans during training.\n"
    )

    # Start a trace context
    if _HAS_OTEL:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("agent_invocation") as span:
            result = agent('Reply with exactly this text: "The answer is 42"')
            inspector.inspect_current_span()
    else:
        result = agent('Reply with exactly this text: "The answer is 42"')

    print("\n" + "=" * 70)
    print("Captured Token IDs (via VLLMTokenRecorder)")
    print("=" * 70)

    if recorder.prompt_token_ids and recorder.token_ids:
        print(f"\nPrompt token IDs: {len(recorder.prompt_token_ids)} tokens")
        print(f"  First 10: {recorder.prompt_token_ids[:10]}")
        print(f"  Last 10:  {recorder.prompt_token_ids[-10:]}")

        print(f"\nResponse token IDs: {len(recorder.token_ids)} tokens")
        print(f"  First 10: {recorder.token_ids[:10]}")
        print(f"  Last 10:  {recorder.token_ids[-10:]}")

    else:
        print("\n⚠️  No token IDs captured.")
        print("   Make sure your vLLM server supports `return_token_ids` in streaming mode.")

    print("\n" + "=" * 70)
    print("Span Attributes (for Agent Lightning)")
    print("=" * 70)
    print(
        "\nThe following span attributes are automatically added when token IDs are captured:\n"
    )
    print("Standard OpenTelemetry Semantic Convention Attributes:")
    print("  • llm.token_count.prompt")
    print("    - Token count for the prompt (per OpenTelemetry LLM semantic conventions)")
    print("\n  • llm.token_count.completion")
    print("    - Token count for the completion (per OpenTelemetry LLM semantic conventions)")
    print("\nToken ID Arrays:")
    print("  • llm.hosted_vllm.prompt_token_ids")
    print("    - Token ID array for the prompt")
    print("\n  • llm.hosted_vllm.response_token_ids")
    print("    - Token ID array for the response")

    print("\n" + "=" * 70)
    print("Agent Lightning Usage")
    print("=" * 70)
    print(
        "\nAgent Lightning can extract token IDs from spans using:\n"
        "  - OpenTelemetry span exporters (e.g., OTLP, console, custom exporters)\n"
        "  - Standard attributes: `llm.token_count.prompt`, `llm.token_count.completion`\n"
        "  - Token ID arrays: `llm.hosted_vllm.prompt_token_ids`, `llm.hosted_vllm.response_token_ids`\n"
        "  - Custom span processors that capture these attributes for training data\n"
        "\nReference: https://blog.vllm.ai/2025/10/22/agent-lightning.html\n"
    )

    if _HAS_OTEL and inspector.captured_attributes:
        print("Span context captured:")
        for attr in inspector.captured_attributes:
            print(f"  Trace ID: {attr['trace_id']}")
            print(f"  Span ID:  {attr['span_id']}")
            print("  (Token IDs are stored as attributes on this span)")

    print("\n" + "=" * 70)
    print("Result")
    print("=" * 70)
    print(f"\n{result}\n")


if __name__ == "__main__":
    main()

