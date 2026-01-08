"""Community vLLM utilities for Strands Agents.

Credit / reference:
- Inspired by `horizon-rl/strands-sglang` community provider packaging patterns:
  https://github.com/horizon-rl/strands-sglang
"""

from .token import Token, TokenManager
from .vllm import VLLMModel, VLLMModelConfig, make_vllm_openai_model
from .recorder import VLLMTokenRecorder

__all__ = [
    "VLLMModel",
    "VLLMModelConfig",
    "make_vllm_openai_model",
    "Token",
    "TokenManager",
    "VLLMTokenRecorder",
]
