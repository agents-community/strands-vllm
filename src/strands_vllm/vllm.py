"""vLLM convenience model provider for Strands Agents.

vLLM serves an OpenAI-compatible API, so this is intentionally a thin wrapper around
`strands.models.openai.OpenAIModel` that:
- sets `base_url` + default `api_key`
- optionally enables vLLM token IDs via `params["extra_body"] = {"return_token_ids": True}`

This package is designed as a community add-on, similar to how `strands-sglang` keeps
provider-specific logic outside the core SDK.

Implementation notes:
- We keep the streaming structure aligned with core Strands `OpenAIModel` so the Agent loop
  behaves the same (tool calling, stop reasons, usage extraction).
- vLLM often returns `usage` only in a trailing chunk, so after emitting the final `messageStop`
  we continue iterating the stream to capture the final usage payload (when present) and emit
  it as a Strands `metadata` event.

Credit / reference:
- Inspired by `horizon-rl/strands-sglang` patterns for TITO-focused providers:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Optional, TypedDict, cast

import openai
from strands.models.openai import OpenAIModel
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


class VLLMModelConfig(TypedDict, total=False):
    """Configuration for vLLM (OpenAI-compatible) endpoints."""

    base_url: str
    api_key: str
    model_id: str
    params: dict[str, Any] | None
    return_token_ids: bool
    disable_tools: bool


def make_vllm_openai_model(
    *,
    base_url: str,
    model_id: str,
    api_key: str = "EMPTY",
    params: Optional[dict[str, Any]] = None,
    return_token_ids: bool = False,
    disable_tools: bool = False,
) -> OpenAIModel:
    """Create an OpenAIModel configured to talk to a vLLM server."""
    merged_params: dict[str, Any] = dict(params or {})
    if return_token_ids:
        extra_body = dict(cast(dict[str, Any], merged_params.get("extra_body", {})))
        extra_body.setdefault("return_token_ids", True)
        merged_params["extra_body"] = extra_body
    if disable_tools:
        merged_params.setdefault("tool_choice", "none")

    return OpenAIModel(
        client_args={"api_key": api_key, "base_url": base_url},
        model_id=model_id,
        params=merged_params,
    )


class VLLMModel(OpenAIModel):
    """Thin convenience wrapper for using vLLM via the core OpenAIModel."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        params: Optional[dict[str, Any]] = None,
        return_token_ids: bool = False,
        disable_tools: bool = False,
        client: Any | None = None,
    ) -> None:
        self._disable_tools = disable_tools
        merged_params: dict[str, Any] = dict(params or {})
        if return_token_ids:
            extra_body = dict(cast(dict[str, Any], merged_params.get("extra_body", {})))
            extra_body.setdefault("return_token_ids", True)
            merged_params["extra_body"] = extra_body
        if disable_tools:
            merged_params.setdefault("tool_choice", "none")

        if client is not None:
            super().__init__(client=client, model_id=model_id, params=merged_params)
        else:
            super().__init__(client_args={"api_key": api_key, "base_url": base_url}, model_id=model_id, params=merged_params)

    @staticmethod
    def _coerce_int_list(value: Any) -> list[int] | None:
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            return cast(list[int], value)
        return None

    @classmethod
    def _extract_token_fields_from_openai_chunk(cls, chunk_dict: dict[str, Any]) -> tuple[list[int] | None, list[int] | None]:
        pti = cls._coerce_int_list(chunk_dict.get("prompt_token_ids"))
        ti = cls._coerce_int_list(chunk_dict.get("token_ids"))

        choices = chunk_dict.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0] if isinstance(choices[0], dict) else {}
            pti = pti or cls._coerce_int_list(c0.get("prompt_token_ids"))
            ti = ti or cls._coerce_int_list(c0.get("token_ids"))

            delta = c0.get("delta")
            if isinstance(delta, dict):
                pti = pti or cls._coerce_int_list(delta.get("prompt_token_ids"))
                ti = ti or cls._coerce_int_list(delta.get("token_ids"))

        return pti, ti

    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation and preserve vLLM token-id fields when available.

        This mirrors core `OpenAIModel.stream`, but adds `messageStop.additionalModelResponseFields`
        when vLLM returns token IDs (e.g., prompt_token_ids, token_ids).
        """
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        if getattr(self, "_disable_tools", False) and not tool_specs:
            request.pop("tools", None)
            request.pop("tool_choice", None)

        requested_prompt_token_ids: list[int] | None = None
        req_extra_body = cast(dict[str, Any], request.get("extra_body", {}) or {})
        requested_prompt_token_ids = self._coerce_int_list(req_extra_body.get("prompt_token_ids"))

        async with self._get_client() as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except openai.RateLimitError as e:
                raise ModelThrottledException(str(e)) from e

            yield self.format_chunk({"chunk_type": "message_start"})
            tool_calls: dict[int, list[Any]] = {}
            data_type = None
            finish_reason: str | None = None
            last_chunk_dict: dict[str, Any] | None = None
            last_usage_obj: Any | None = None
            streamed_prompt_token_ids: list[int] | None = None
            streamed_token_ids: list[int] = []

            async for event in response:
                try:
                    last_chunk_dict = cast(dict[str, Any], event.model_dump())
                except Exception:
                    last_chunk_dict = None

                if last_chunk_dict is not None:
                    pti, ti = self._extract_token_fields_from_openai_chunk(last_chunk_dict)
                    if pti is not None and streamed_prompt_token_ids is None:
                        streamed_prompt_token_ids = pti
                    if ti is not None:
                        streamed_token_ids.extend(ti)

                if hasattr(event, "usage") and event.usage:
                    last_usage_obj = event.usage

                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]

                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": data_type,
                            "data": choice.delta.reasoning_content,
                        }
                    )

                if choice.delta.content:
                    chunks, data_type = self._stream_switch_content("text", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk(
                        {"chunk_type": "content_delta", "data_type": data_type, "data": choice.delta.content}
                    )

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    if data_type:
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                    break

            for tool_deltas in tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})
                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            additional: dict[str, Any] = {}
            if streamed_prompt_token_ids is not None:
                additional["prompt_token_ids"] = streamed_prompt_token_ids
            if streamed_token_ids:
                additional["token_ids"] = streamed_token_ids
            if requested_prompt_token_ids is not None:
                additional["prompt_token_ids"] = requested_prompt_token_ids

            stop_reason_data = finish_reason or "end_turn"
            if stop_reason_data == "tool_calls" and not tool_calls:
                stop_reason_data = "end_turn"
            stop_chunk = self.format_chunk({"chunk_type": "message_stop", "data": stop_reason_data})
            if additional and "messageStop" in stop_chunk:
                stop_chunk["messageStop"]["additionalModelResponseFields"] = additional
            yield stop_chunk

            async for event in response:
                if hasattr(event, "usage") and event.usage:
                    last_usage_obj = event.usage

            if last_usage_obj is not None:
                yield self.format_chunk({"chunk_type": "metadata", "data": last_usage_obj})
