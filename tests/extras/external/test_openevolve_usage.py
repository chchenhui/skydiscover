"""Token accounting bridge for OpenEvolve's private LLM client."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openevolve")

from skydiscover.extras.external.openevolve_backend import (
    _USAGE_EVENT_PATH_ENV,
    _init_tracked_openevolve_llm,
    _report_usage_events,
)
from skydiscover.llm.pricing import Usage


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


def test_worker_usage_events_are_aggregated_in_main_process(tmp_path, monkeypatch):
    event_path = tmp_path / "llm_usage_events.jsonl"
    monkeypatch.setenv(_USAGE_EVENT_PATH_ENV, str(event_path))
    usage = Usage(input_tokens=10, cache_read_tokens=5, output_tokens=2, calls=1)

    from skydiscover.extras.external.openevolve_backend import _append_usage_event

    _append_usage_event("gpt-5.6-luna", "https://api.openai.com/v1", usage)
    _append_usage_event("gpt-5.6-luna", "https://api.openai.com/v1", usage)

    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert len(events) == 2
    assert all(event["usage"]["input_tokens"] == 10 for event in events)

    tracker = _report_usage_events(str(event_path), str(tmp_path))
    assert tracker.total_usage.calls == 2
    assert tracker.total_usage.input_tokens == 20
    assert tracker.total_usage.cache_read_tokens == 10
    assert tracker.total_usage.output_tokens == 4

    summary = json.loads((tmp_path / "llm_usage.json").read_text())
    assert summary["total"]["calls"] == 2
    assert summary["total"]["total_tokens"] == 34
    assert summary["total"]["cost_usd"] is not None
