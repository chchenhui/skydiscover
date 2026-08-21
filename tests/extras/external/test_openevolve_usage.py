"""Token accounting bridge for OpenEvolve's private LLM client."""

from types import SimpleNamespace

import pytest

pytest.importorskip("openevolve")

from skydiscover.extras.external.openevolve_backend import _init_tracked_openevolve_llm


def test_raw_response_is_recorded_before_openevolve_discards_usage(monkeypatch):
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
    )
    calls = []

    class FakeOpenAILLM:
        def __init__(self, model_cfg):
            self.model = model_cfg.name
            self.api_base = model_cfg.api_base
            self.client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=lambda **kwargs: response)
                )
            )

    monkeypatch.setattr("openevolve.llm.openai.OpenAILLM", FakeOpenAILLM)
    monkeypatch.setattr(
        "skydiscover.llm.pricing.record_response",
        lambda model, raw, api_base=None: calls.append((model, raw, api_base)),
    )

    model = _init_tracked_openevolve_llm(
        SimpleNamespace(name="gpt-5.6-luna", api_base="https://api.openai.com/v1")
    )
    assert model.client.chat.completions.create(model="gpt-5.6-luna") is response
    assert calls == [("gpt-5.6-luna", response, "https://api.openai.com/v1")]
