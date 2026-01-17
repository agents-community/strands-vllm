"""Pytest configuration for integration tests.

Usage:
  pytest tests/integration/ -v --vllm-base-url=http://localhost:8000/v1
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options for integration tests."""
    parser.addoption(
        "--vllm-base-url",
        action="store",
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM server base URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)",
    )
    parser.addoption(
        "--vllm-model-id",
        action="store",
        default=os.getenv("VLLM_MODEL_ID", "AMead10/Llama-3.2-3B-Instruct-AWQ"),
        help="vLLM model ID (default: $VLLM_MODEL_ID or AMead10/Llama-3.2-3B-Instruct-AWQ)",
    )


@pytest.fixture
def vllm_base_url(request: pytest.FixtureRequest) -> str:
    """Get vLLM server base URL from CLI or env."""
    return request.config.getoption("--vllm-base-url")


@pytest.fixture
def vllm_model_id(request: pytest.FixtureRequest) -> str:
    """Get vLLM model ID from CLI or env."""
    return request.config.getoption("--vllm-model-id")
