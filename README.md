# strands-vllm

Community vLLM utilities for the [Strands Agents SDK](https://github.com/strands-agents/sdk-python).

vLLM serves an OpenAI-compatible API, so most users can simply use `OpenAIModel` with `base_url`.
This package provides small convenience helpers and (optionally) token-id/TITO-friendly defaults.

## Credit / reference

This community package is inspired by the structure and example style of
[`horizon-rl/strands-sglang`](https://github.com/horizon-rl/strands-sglang).

## Install

```bash
pip install strands-vllm
```

## vLLM server notes (tools + token IDs)

- **Tools**: if you want tool calling, your vLLM server must be started with tool-calling enabled and an appropriate
  chat template for your model (e.g., Llama 3.2 tool template).
- **Token IDs (TITO)**: `return_token_ids=True` requests vLLM token IDs; vLLM will include `prompt_token_ids` and
  streamed `token_ids` when supported.

## Usage

### Minimal: OpenAIModel pointed at vLLM

```python
from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(
    client_args={"api_key": "EMPTY", "base_url": "http://localhost:8000/v1"},
    model_id="AMead10/Llama-3.2-3B-Instruct-AWQ",
)

agent = Agent(model=model)
print(agent("Hi"))
```

### Convenience: `VLLMModel`

```python
from strands import Agent
from strands_vllm import VLLMModel

model = VLLMModel(
    base_url="http://localhost:8000/v1",
    model_id="AMead10/Llama-3.2-3B-Instruct-AWQ",
    return_token_ids=True,
)

agent = Agent(model=model)
print(agent("Say hello"))
```

Tip: if you want to print only the final result (without streaming output being printed along the way),
pass `callback_handler=None`:

```python
agent = Agent(model=model, callback_handler=None)
print(agent("Say hello"))
```

### Examples

All examples can be pointed at your server with:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_ID="AMead10/Llama-3.2-3B-Instruct-AWQ"
```

### Tools (strands-agents-tools)

Install the optional tools package and run the example:

```bash
pip install strands-agents-tools
python examples/math_agent.py
```

### Tool-call validation example

```bash
pip install strands-agents-tools
python examples/tool_validation_agent.py
```

### RL rollout (TITO + loss_mask + retokenization check)

```bash
pip install "strands-vllm[drift]" strands-agents-tools
python examples/rl_rollout_tito.py
```

### Tool-call validation (recommended with vLLM tool parsers)

vLLM tool calling can involve server-side post-processing, so it can be useful to guard tool execution:

```python
from strands import Agent
from strands_tools.calculator import calculator
from strands_vllm import VLLMModel, VLLMToolValidationHooks

model = VLLMModel(base_url="http://localhost:8000/v1", model_id="...", return_token_ids=True)
agent = Agent(model=model, tools=[calculator], hooks=[VLLMToolValidationHooks()])
print(agent("Compute 17 * 19 using the calculator tool."))
```

### Retokenization drift (educational)

This demo mirrors the idea from `strands-sglang` and shows why TITO matters:
`encode(decode(tokens)) != tokens` can happen.

```bash
pip install "strands-vllm[drift]" strands-agents-tools
python examples/retokenization_drift.py
```

### Token-in / token-out (TITO)

If your vLLM server includes token IDs in streaming responses, you can capture them
using `VLLMTokenRecorder` (see `examples/basic_agent.py`).

## Development

Install from source:

```bash
git clone <your-fork-url>
cd strands-vllm
pip install -e ".[dev]"
```

## License

Apache-2.0
