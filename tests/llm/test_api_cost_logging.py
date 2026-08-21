"""API-level test for token and cost logging without making a real request."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.pricing import load_pricing


@pytest.mark.asyncio
async def test_generate_logs_tokens_rates_and_total_cost(monkeypatch, caplog):
    """A successful API response should immediately emit its complete cost."""
    raw_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="mock answer"))],
        usage=SimpleNamespace(
            # OpenAI includes cached reads in prompt_tokens.
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=100),
        ),
    )
    create = Mock(return_value=raw_response)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        "skydiscover.llm.openai.openai.OpenAI", lambda **kwargs: fake_client
    )
    monkeypatch.delenv("SKYDISCOVER_MODELS_YAML", raising=False)
    load_pricing(refresh=True)

    llm = OpenAILLM(
        LLMModelConfig(
            name="gpt-5.6-luna",
            api_base="https://api.openai.com/v1",
            api_key="test-key",
            max_tokens=100,
            retries=0,
        )
    )

    with caplog.at_level(logging.INFO, logger="skydiscover.llm"):
        response = await llm.generate("system", [{"role": "user", "content": "hello"}])

    assert response.text == "mock answer"
    assert response.usage is not None
    assert response.usage.input_tokens == 600
    assert response.usage.cache_read_tokens == 400
    assert response.usage.output_tokens == 500
    create.assert_called_once()

    cost_log = next(message for message in caplog.messages if message.startswith("LLM call "))
    assert "input=600 (@ $0.20/1M = $0.000120)" in cost_log
    assert "cache_read=400 (@ $0.02/1M = $0.000008)" in cost_log
    assert "cache_write=0 (@ $0.25/1M = $0.0000)" in cost_log
    assert "output=500 (@ $1.20/1M = $0.000600)" in cost_log
    assert "reasoning=100 (included in output)" in cost_log
    assert "total_tokens=1500, total_cost=$0.000728" in cost_log
