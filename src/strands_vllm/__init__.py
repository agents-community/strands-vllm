"""Community vLLM utilities for Strands Agents.

Credit / reference:
- Inspired by `horizon-rl/strands-sglang` community provider packaging patterns:
  https://github.com/horizon-rl/strands-sglang
"""

from .recorder import VLLMTokenRecorder
from .token import Token, TokenManager
from .tool_validation import VLLMToolValidationHooks
from .vllm import VLLMModel, VLLMModelConfig, make_vllm_openai_model

__all__ = [
    "VLLMModel",
    "VLLMModelConfig",
    "make_vllm_openai_model",
    "Token",
    "TokenManager",
    "VLLMTokenRecorder",
    "VLLMToolValidationHooks",
]
