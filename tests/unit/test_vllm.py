from __future__ import annotations

from strands_vllm import VLLMModel, make_vllm_openai_model


def test_make_vllm_openai_model_injects_base_url_and_token_ids():
    model = make_vllm_openai_model(
        base_url="http://localhost:8000/v1",
        model_id="m",
        return_token_ids=True,
        params={"temperature": 0},
    )

    cfg = model.get_config()
    assert cfg["model_id"] == "m"
    assert cfg["params"]["temperature"] == 0
    assert cfg["params"]["extra_body"]["return_token_ids"] is True


def test_vllm_model_merges_extra_body():
    model = VLLMModel(
        base_url="http://localhost:8000/v1",
        model_id="m",
        return_token_ids=True,
        params={"extra_body": {"foo": "bar"}},
    )

    cfg = model.get_config()
    assert cfg["params"]["extra_body"]["foo"] == "bar"
    assert cfg["params"]["extra_body"]["return_token_ids"] is True

