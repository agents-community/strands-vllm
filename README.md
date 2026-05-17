# Strands-vLLM

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

[![PyPI](https://img.shields.io/pypi/v/strands-vllm.svg)](https://pypi.org/project/strands-vllm/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Community vLLM provider for [Strands Agents SDK](https://github.com/strands-agents/sdk-python) with Token-In/Token-Out (TITO) support and Agent Lightning integration.

## Features

This package provides convenient utilities for using vLLM with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python), designed for training-ready agent rollouts:

- **Token-In/Token-Out (TITO)**: capture token IDs directly from vLLM streaming responses (no retokenization drift)
- **Agent Lightning integration**: automatic OpenTelemetry span attributes for token IDs
- **Tool calling support**: validation hooks for vLLM's server-side tool call post-processing
- **OpenAI-compatible API**: works with vLLM's OpenAI-compatible endpoint

## Requirements

- Python 3.10+
- Strands Agents SDK
- vLLM server running with your model

## Installation

```bash
pip install strands-vllm
```

Or install from source with development dependencies:

```bash
git clone https://github.com/agents-community/strands-vllm.git
cd strands-vllm
pip install -e ".[dev]"
```

## Quick Start

### 1. Start vLLM Server

```bash
vllm serve <MODEL_ID> \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

### 2. Basic Agent

```python
from strands import Agent
from strands_vllm import VLLMModel

model = VLLMModel(
    base_url="http://localhost:8000/v1",
    model_id="AMead10/Llama-3.2-3B-Instruct-AWQ",
    return_token_ids=True,
)

agent = Agent(model=model)
result = agent("Say hello")
print(result)
```

### 3. Token IDs for RL Training

```python
from strands import Agent
from strands.handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler
from strands_vllm import VLLMModel, VLLMTokenRecorder

model = VLLMModel(
    base_url="http://localhost:8000/v1",
    model_id="AMead10/Llama-3.2-3B-Instruct-AWQ",
    return_token_ids=True,
)

recorder = VLLMTokenRecorder()
printer = PrintingCallbackHandler(verbose_tool_use=False)
callback = CompositeCallbackHandler(printer, recorder)

agent = Agent(model=model, callback_handler=callback)
agent("What is 17 * 19?")

# Access TITO data for RL training
print(f"Prompt token IDs: {recorder.prompt_token_ids}")
print(f"Response token IDs: {recorder.token_ids}")
```

**Note**: `VLLMTokenRecorder` automatically adds token IDs as OpenTelemetry span attributes (`llm.hosted_vllm.prompt_token_ids`, `llm.hosted_vllm.response_token_ids`) for [Agent Lightning](https://blog.vllm.ai/2025/10/22/agent-lightning.html) compatibility.

## Slime Training

For RL training with [Slime](https://github.com/THUDM/slime/), `VLLMModel` with `VLLMTokenRecorder` eliminates the retokenization step by capturing token IDs directly from vLLM streaming responses.

**Note**: This requires THUDM/slime to be installed (not the pip `slime` package):
```bash
pip install git+https://github.com/THUDM/slime.git
```

```python
from strands import Agent, tool
from strands_vllm import VLLMModel, VLLMTokenRecorder, TokenManager
from slime.utils.types import Sample

SYSTEM_PROMPT = "..."
MAX_TOOL_ITERATIONS = ...  # e.g., 5

@tool
def execute_python_code(code: str):
    """Execute Python code and return the output."""
    ...

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate with TITO: tokens captured during generation, no retokenization."""
    assert not args.partial_rollout, "Partial rollout not supported."

    # Set up Agent with VLLMModel and VLLMTokenRecorder
    model = VLLMModel(
        base_url=args.vllm_base_url,
        model_id=args.hf_checkpoint.split("/")[-1],
        return_token_ids=True,
        params={k: sampling_params[k] for k in ["max_new_tokens", "temperature", "top_p"]},
    )
    recorder = VLLMTokenRecorder()
    agent = Agent(
        model=model,
        tools=[execute_python_code],
        callback_handler=recorder,
        system_prompt=SYSTEM_PROMPT,
    )

    # Run Agent Loop
    prompt = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
    try:
        await agent.invoke_async(prompt)
        sample.status = Sample.Status.COMPLETED
    except Exception as e:
        # Always use TRUNCATED instead of ABORTED because Slime doesn't properly
        # handle ABORTED samples in reward processing. See: https://github.com/THUDM/slime/issues/200
        sample.status = Sample.Status.TRUNCATED
        logger.warning(f"TRUNCATED: {type(e).__name__}: {e}")

    # TITO: extract trajectory from recorder and TokenManager
    tm = TokenManager()
    for entry in recorder.history:
        pti = entry.get("prompt_token_ids")
        ti = entry.get("token_ids")
        if pti:
            tm.add_prompt(pti)
        if ti:
            tm.add_response(ti)

    prompt_len = len(tm.segments[0])  # system + user are first segment
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

    # Cleanup and return
    recorder.reset()
    agent.cleanup()
    return sample
```

## Examples

All examples can be configured with environment variables:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_ID="AMead10/Llama-3.2-3B-Instruct-AWQ"
```

### Math agent with tools

```bash
pip install strands-agents-tools
python examples/math_agent.py
```

### Agent Lightning integration

Demonstrates token IDs in OpenTelemetry spans for Agent Lightning compatibility:

```bash
python examples/agent_lightning.py
```

### Tool-call validation

vLLM tool calling can involve server-side post-processing. Use validation hooks to guard tool execution:

```python
from strands import Agent
from strands_tools.calculator import calculator
from strands_vllm import VLLMModel, VLLMToolValidationHooks

model = VLLMModel(base_url="http://localhost:8000/v1", model_id="...", return_token_ids=True)
agent = Agent(model=model, tools=[calculator], hooks=[VLLMToolValidationHooks()])
print(agent("Compute 17 * 19 using the calculator tool."))
```

### Retokenization drift (educational)

This demo shows why TITO matters: `encode(decode(tokens)) != tokens` can happen.

```bash
pip install "strands-vllm[drift]" strands-agents-tools
python examples/retokenization_drift.py
```

## Testing

```bash
# Unit tests
uv run pytest tests/unit/ -v

# Integration tests (requires vLLM server)
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_ID="AMead10/Llama-3.2-3B-Instruct-AWQ"
uv run pytest tests/integration/ -v
```

Integration tests include:
- `test_agent_math500.py` - Agent tests with real MATH-500 problems and TITO consistency checks
- `test_slime_integration.py` - Slime training pattern using Slime's `Sample` type (requires `pip install git+https://github.com/THUDM/slime.git`)

## Contributing

Contributions welcome! Install pre-commit hooks for code style and commit message validation:

```bash
pip install -e ".[dev]"
pre-commit install -t pre-commit -t commit-msg
```

This project uses [Conventional Commits](https://www.conventionalcommits.org/). Commit messages must follow the format:

```
<type>(<scope>): <description>

# Examples:
feat(recorder): add Agent Lightning span attributes
fix(vllm): handle empty response from server
docs: update TITO usage examples
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## Related Projects

- [strands-sglang](https://github.com/horizon-rl/strands-sglang) - SGLang provider for Strands Agents SDK

## License

Apache License 2.0 - see [LICENSE](LICENSE).