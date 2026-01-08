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


class VLLMTokenRecorder:
    """Capture vLLM token fields from streaming events.

    Usage:
        recorder = VLLMTokenRecorder()
        agent = Agent(model=model, callback_handler=recorder)
        agent("hi")
        print(recorder.prompt_token_ids, recorder.token_ids)
    """

    def __init__(self, inner: Any | None = None) -> None:
        self.inner = inner
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

