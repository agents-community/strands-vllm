"""Token tracking helpers for token-in/token-out (TITO) workflows.

This is intentionally lightweight and provider-agnostic: it stores token IDs and
an RL-friendly loss mask (0 = prompt/tool, 1 = model output).

vLLM can return server-side token IDs (avoids retokenization drift), so a local
tokenizer is optional and not required for training.

Credit / reference:
- This module is inspired by the segment-based TITO design in `strands-sglang`
  (see `TokenManager` in `horizon-rl/strands-sglang`).
  We keep the same core idea (prompt vs response segments + loss_mask), but do
  not require a tokenizer because vLLM can return token IDs directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    token_id: int
    logprob: float | None = None
    loss_mask: bool = True


class TokenManager:
    """Segment-based token accumulator (prompt vs response)."""

    def __init__(self) -> None:
        self._segments: list[list[Token]] = []

    def reset(self) -> None:
        self._segments = []

    def add_prompt(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        if not token_ids:
            return
        self._segments.append(
            [
                Token(token_id=tid, logprob=(logprobs[i] if logprobs and i < len(logprobs) else None), loss_mask=False)
                for i, tid in enumerate(token_ids)
            ]
        )

    def add_response(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        if not token_ids:
            return
        self._segments.append(
            [
                Token(token_id=tid, logprob=(logprobs[i] if logprobs and i < len(logprobs) else None), loss_mask=True)
                for i, tid in enumerate(token_ids)
            ]
        )

    @property
    def tokens(self) -> list[Token]:
        return [t for seg in self._segments for t in seg]

    @property
    def token_ids(self) -> list[int]:
        return [t.token_id for t in self.tokens]

    @property
    def loss_mask(self) -> list[int]:
        return [int(t.loss_mask) for t in self.tokens]

    @property
    def logprobs(self) -> list[float | None]:
        return [t.logprob for t in self.tokens]

    @property
    def segments(self) -> list[list[Token]]:
        return [list(seg) for seg in self._segments]

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        """List of (is_output, length) per segment."""
        return [(seg[0].loss_mask if seg else False, len(seg)) for seg in self._segments]

    def __len__(self) -> int:
        return sum(len(seg) for seg in self._segments)

    def __repr__(self) -> str:
        n_segments = len(self._segments)
        n_tokens = len(self)
        n_output = sum(1 for t in self.tokens if t.loss_mask)
        return f"TokenManager(segments={n_segments}, tokens={n_tokens}, output_tokens={n_output})"

