"""Integration tests for strands-vllm.

These tests require a running vLLM server and can be run with:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_ID="AMead10/Llama-3.2-3B-Instruct-AWQ"
pytest tests/integration/ -v
```
"""
