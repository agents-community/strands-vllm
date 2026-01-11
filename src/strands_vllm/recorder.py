"""Streaming token-id capture helpers for vLLM.

Strands core streaming events include `ModelStreamChunkEvent` which is emitted to the
Agent callback handler as `{"event": <StreamEvent>}`.

This recorder watches for a `messageStop` chunk containing vLLM token fields and
stores them for later inspection (e.g., RL rollouts).

Credit / reference:
- Inspired by `horizon-rl/strands-sglang` TITO patterns:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from opentelemetry import trace as trace_api

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    trace_api = None  # type: ignore[assignment, misc]


class VLLMTokenRecorder:
    """Capture vLLM token fields from streaming events.

    Usage:
        recorder = VLLMTokenRecorder()
        agent = Agent(model=model, callback_handler=recorder)
        agent("hi")
        print(recorder.prompt_token_ids, recorder.token_ids)
    
    If OpenTelemetry is available, token IDs are automatically added as span attributes
    (`llm.hosted_vllm.prompt_token_ids`, `llm.hosted_vllm.response_token_ids`) for Agent Lightning compatibility.
    """

    def __init__(self, inner: Any | None = None, add_to_span: bool = True) -> None:
        self.inner = inner
        self.add_to_span = add_to_span and _HAS_OTEL
        self.prompt_token_ids: list[int] | None = None
        self.token_ids: list[int] | None = None
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.prompt_token_ids = None
        self.token_ids = None
        self.history = []

    @staticmethod
    def _coerce_int_list(value: Any) -> Optional[list[int]]:
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            return value
        return None

    def __call__(self, **kwargs: Any) -> None:
        if self.inner is not None:
            self.inner(**kwargs)

        evt = kwargs.get("event")
        if not isinstance(evt, dict):
            return
        message_stop = evt.get("messageStop")
        if not isinstance(message_stop, dict):
            return

        additional = message_stop.get("additionalModelResponseFields")
        if not isinstance(additional, dict):
            return

        pti = self._coerce_int_list(additional.get("prompt_token_ids"))
        ti = self._coerce_int_list(additional.get("token_ids"))

        if pti is not None:
            self.prompt_token_ids = pti
        if ti is not None:
            self.token_ids = ti
        
        if pti is not None or ti is not None:
            self.history.append({"prompt_token_ids": pti, "token_ids": ti})
            if self.add_to_span:
                self._add_token_ids_to_span(pti, ti)
    
    def _add_token_ids_to_span(self, prompt_token_ids: list[int] | None, token_ids: list[int] | None) -> None:
        """Add token IDs as attributes to the current OpenTelemetry span for Agent Lightning compatibility.
        
        Sets the following span attributes:
        - llm.token_count.prompt, llm.token_count.completion (standard OpenTelemetry attributes)
        - llm.hosted_vllm.prompt_token_ids, llm.hosted_vllm.response_token_ids (token ID arrays)
        
        Reference: https://blog.vllm.ai/2025/10/22/agent-lightning.html
        """
        if not _HAS_OTEL or trace_api is None:
            return
        
        try:
            span = trace_api.get_current_span()
            if not span.is_recording():
                return
            
            if prompt_token_ids is not None:
                span.set_attribute("llm.token_count.prompt", len(prompt_token_ids))
                span.set_attribute("llm.hosted_vllm.prompt_token_ids", prompt_token_ids)
            if token_ids is not None:
                span.set_attribute("llm.token_count.completion", len(token_ids))
                span.set_attribute("llm.hosted_vllm.response_token_ids", token_ids)
        except Exception:
            pass

