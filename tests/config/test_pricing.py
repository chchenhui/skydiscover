"""Tests for token accounting and cost estimation (skydiscover/llm/pricing.py)."""

from types import SimpleNamespace

import pytest

from skydiscover.llm.pricing import (
    CostTracker,
    PricingTable,
    Usage,
    load_pricing,
    provider_from_api_base,
)

# Mirrors the shape of skydiscover/llm/models.yaml, including a model that is
# priced differently by two providers.
TABLE = PricingTable(
    {
        "openai": [
            {
                "name": "gpt-5.6-luna",
                "input_per_million": 0.20,
                "output_per_million": 1.20,
                "cache_read_per_million": 0.02,
                "cache_write_per_million": 0.25,
            }
        ],
        "anthropic": [
            {
                "name": "claude-opus-5",
                "input_per_million": 5.00,
                "output_per_million": 25.00,
                "cache_read_per_million": 0.50,
                "cache_write_per_million": 6.25,
                "cache_write_1h_per_million": 10.00,
            }
        ],
        "openrouter": [
            {
                "name": "openai/gpt-5.6-luna",
                "input_per_million": 0.10,
                "output_per_million": 0.60,
                "cache_read_per_million": 0.01,
            },
            {
                "name": "moonshotai/kimi-k3",
                "input_per_million": 3.00,
                "output_per_million": 15.00,
            },
        ],
    }
)


def _chat_usage(prompt, completion, cached=0, reasoning=0):
    """A Chat Completions usage block (prompt_tokens includes cached)."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        )
    )


class TestUsageExtraction:
    def test_chat_completions_splits_cached_from_input(self):
        usage = Usage.from_response(_chat_usage(1000, 500, cached=400, reasoning=300))
        # prompt_tokens includes the cached ones, so input is the remainder.
        assert usage.input_tokens == 600
        assert usage.cache_read_tokens == 400
        assert usage.output_tokens == 500
        assert usage.reasoning_tokens == 300
        assert usage.prompt_tokens == 1000
        assert usage.total_tokens == 1500
        assert usage.calls == 1

    def test_responses_api_shape(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=800,
                output_tokens=200,
                input_tokens_details=SimpleNamespace(cached_tokens=300),
                output_tokens_details=SimpleNamespace(reasoning_tokens=150),
            )
        )
        usage = Usage.from_response(resp)
        assert (usage.input_tokens, usage.cache_read_tokens) == (500, 300)
        assert usage.output_tokens == 200

    def test_anthropic_input_tokens_already_exclude_cache(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=900,
                cache_creation_input_tokens=200,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=120,
                    ephemeral_1h_input_tokens=80,
                ),
            )
        )
        usage = Usage.from_response(resp)
        assert usage.input_tokens == 100  # not reduced by the cached tokens
        assert usage.cache_read_tokens == 900
        assert usage.cache_write_tokens == 120
        assert usage.cache_write_1h_tokens == 80
        assert usage.prompt_tokens == 1200

    def test_missing_or_empty_usage_returns_none(self):
        assert Usage.from_response(SimpleNamespace()) is None
        assert Usage.from_response(_chat_usage(0, 0)) is None

    def test_dict_response_is_accepted(self):
        usage = Usage.from_response({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        assert usage.input_tokens == 10 and usage.output_tokens == 5

    def test_addition_accumulates(self):
        total = Usage(input_tokens=10, output_tokens=5, calls=1) + Usage(
            input_tokens=1, cache_read_tokens=2, calls=1
        )
        assert total.input_tokens == 11
        assert total.cache_read_tokens == 2
        assert total.calls == 2


class TestResolution:
    def test_provider_disambiguates_same_model(self):
        direct = TABLE.resolve("gpt-5.6-luna", provider="openai")
        assert direct.input_per_million == 0.20

        via_router = TABLE.resolve("openai/gpt-5.6-luna", provider="openrouter")
        assert via_router.input_per_million == 0.10

    def test_vendor_prefix_falls_back_to_bare_name(self):
        # Configured as "openai/gpt-5.6-luna" but served by OpenAI directly.
        pricing = TABLE.resolve("openai/gpt-5.6-luna", provider="openai")
        assert pricing.input_per_million == 0.20

    def test_dated_snapshot_bills_as_base_model(self):
        pricing = TABLE.resolve("claude-opus-5-2026-01-15", provider="anthropic")
        assert pricing is not None and pricing.name == "claude-opus-5"

    def test_unambiguous_match_without_provider(self):
        pricing = TABLE.resolve("moonshotai/kimi-k3")
        assert pricing.provider == "openrouter"

    def test_unknown_model_is_unpriced(self):
        assert TABLE.resolve("some-unlisted-model") is None

    def test_provider_from_api_base(self):
        assert provider_from_api_base("https://api.openai.com/v1") == "openai"
        assert provider_from_api_base("https://openrouter.ai/api/v1") == "openrouter"
        assert provider_from_api_base("https://api.anthropic.com/v1/") == "anthropic"
        assert provider_from_api_base(None) is None


class TestCost:
    def test_cache_tiers_are_billed_at_their_own_rates(self):
        pricing = TABLE.resolve("claude-opus-5", provider="anthropic")
        usage = Usage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
            cache_write_1h_tokens=1_000_000,
            calls=1,
        )
        # 5.00 input + 25.00 output + 0.50 read + 6.25 write + 10.00 write-1h
        assert pricing.cost(usage) == pytest.approx(46.75)

    def test_cache_read_discount_applies(self):
        pricing = TABLE.resolve("gpt-5.6-luna", provider="openai")
        usage = Usage(input_tokens=600, cache_read_tokens=400, output_tokens=500, calls=1)
        expected = (600 * 0.20 + 400 * 0.02 + 500 * 1.20) / 1_000_000
        assert pricing.cost(usage) == pytest.approx(expected)

    def test_missing_cache_rate_falls_back_to_input_rate(self):
        pricing = TABLE.resolve("moonshotai/kimi-k3", provider="openrouter")
        usage = Usage(cache_read_tokens=1_000_000, calls=1)
        assert pricing.cost(usage) == pytest.approx(3.00)

    def test_unpriced_entry_yields_none_not_zero(self):
        table = PricingTable({"openai": [{"name": "mystery", "input_per_million": None}]})
        pricing = table.resolve("mystery", provider="openai")
        assert pricing is not None
        assert pricing.is_priced is False
        assert pricing.cost(Usage(input_tokens=1000, calls=1)) is None


class TestCostTracker:
    def test_accumulates_per_model(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(1000, 500), api_base="https://api.openai.com/v1"
        )
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(2000, 100), api_base="https://api.openai.com/v1"
        )

        total = tracker.total_usage
        assert total.calls == 2
        assert total.input_tokens == 3000
        assert total.output_tokens == 600
        expected = (3000 * 0.20 + 600 * 1.20) / 1_000_000
        assert tracker.total_cost_usd == pytest.approx(expected)

    def test_unpriced_model_counts_tokens_but_not_cost(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response("not-in-table", _chat_usage(100, 50))

        assert tracker.total_usage.total_tokens == 150
        assert tracker.total_cost_usd is None  # unknown, never 0.0
        assert tracker.unpriced_models == ["not-in-table"]
        assert "cost unknown" in tracker.format_summary()

    def test_empty_tracker_summary(self):
        assert CostTracker().format_summary() == "LLM usage: no tracked API calls"

    def test_bad_response_does_not_raise(self):
        tracker = CostTracker()
        assert tracker.record_response("m", object()) is None
        assert tracker.total_usage.calls == 0

    def test_each_call_logs_all_billable_tokens_rates_and_cost(self, monkeypatch, caplog):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()

        with caplog.at_level("INFO", logger="skydiscover.llm"):
            tracker.record_response(
                "gpt-5.6-luna",
                _chat_usage(1000, 500, cached=400, reasoning=300),
                api_base="https://api.openai.com/v1",
            )

        message = caplog.messages[-1]
        assert "input=600 (@ $0.20/1M" in message
        assert "cache_read=400 (@ $0.02/1M" in message
        assert "cache_write=0 (@ $0.25/1M" in message
        assert "cache_write_1h=0 (@ $0.25/1M" in message
        assert "output=500 (@ $1.20/1M" in message
        assert "reasoning=300 (included in output)" in message
        assert "total_tokens=1500, total_cost=$0.000728" in message


class TestShippedTable:
    def test_models_yaml_parses_and_prices_a_known_model(self):
        table = load_pricing(refresh=True)
        assert len(table) > 0
        pricing = table.resolve("gpt-5.6-luna", provider="openai")
        assert pricing is not None and pricing.is_priced


class TestUsageCollection:
    """`generate()` must see usage recorded inside a child asyncio Task.

    `asyncio.wait_for` wraps its coroutine in a Task on Python < 3.12, and a
    child Task gets a *copy* of the context -- so a ContextVar holding the
    usage directly would be lost. Only the mutable-sink form survives.
    """

    def test_usage_crosses_task_boundary(self, monkeypatch):
        import asyncio

        from skydiscover.llm.pricing import collect_usage, record_response, sum_usage

        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)

        async def api_call():
            record_response("gpt-5.6-luna", _chat_usage(100, 20))

        async def main():
            with collect_usage() as recorded:
                # Explicit Task: the pre-3.12 asyncio.wait_for shape.
                await asyncio.ensure_future(api_call())
            return sum_usage(recorded)

        usage = asyncio.run(main())
        assert usage is not None, "usage recorded in a child task was lost"
        assert usage.input_tokens == 100 and usage.output_tokens == 20

    def test_retries_and_fallbacks_are_summed(self, monkeypatch):
        from skydiscover.llm.pricing import collect_usage, record_response, sum_usage

        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        with collect_usage() as recorded:
            record_response("gpt-5.6-luna", _chat_usage(100, 20))
            record_response("gpt-5.6-luna", _chat_usage(50, 10))
        usage = sum_usage(recorded)
        assert usage.calls == 2 and usage.input_tokens == 150

    def test_nested_collection_is_isolated(self, monkeypatch):
        from skydiscover.llm.pricing import collect_usage, record_response, sum_usage

        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        with collect_usage() as outer:
            record_response("gpt-5.6-luna", _chat_usage(10, 1))
            with collect_usage() as inner:
                record_response("gpt-5.6-luna", _chat_usage(99, 9))
            record_response("gpt-5.6-luna", _chat_usage(20, 2))

        assert sum_usage(inner).input_tokens == 99
        assert sum_usage(outer).input_tokens == 30  # inner call not double-counted

    def test_no_calls_yields_none(self):
        from skydiscover.llm.pricing import collect_usage, sum_usage

        with collect_usage() as recorded:
            pass
        assert sum_usage(recorded) is None


class TestPerRunScoping:
    """A second run in the same process must not inherit the first run's cost."""

    def test_since_reports_only_the_delta(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(1000, 100), api_base="https://api.openai.com/v1"
        )

        baseline = tracker.snapshot()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(200, 50), api_base="https://api.openai.com/v1"
        )

        run2 = tracker.since(baseline)
        assert run2.total_usage.calls == 1
        assert run2.total_usage.input_tokens == 200
        expected = (200 * 0.20 + 50 * 1.20) / 1_000_000
        assert run2.total_cost_usd == pytest.approx(expected)
        # The parent tracker keeps the cumulative view.
        assert tracker.total_usage.calls == 2

    def test_since_skips_models_untouched_in_the_window(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response("moonshotai/kimi-k3", _chat_usage(10, 5))
        baseline = tracker.snapshot()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(10, 5), api_base="https://api.openai.com/v1"
        )

        assert list(tracker.since(baseline).to_dict()["models"]) == ["openai/gpt-5.6-luna"]

    def test_empty_window_reports_nothing(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response("moonshotai/kimi-k3", _chat_usage(10, 5))
        assert tracker.since(tracker.snapshot()).total_usage.calls == 0

    def test_usage_subtraction_clamps_at_zero(self):
        assert (Usage(input_tokens=5) - Usage(input_tokens=9)).input_tokens == 0


class TestPricingAtReportTime:
    """Tokens are counted as they arrive; prices are applied only when asked."""

    def test_totals_follow_an_edited_price_table(self, monkeypatch):
        cheap = PricingTable(
            {"openai": [{"name": "m", "input_per_million": 1.0, "output_per_million": 2.0}]}
        )
        dear = PricingTable(
            {"openai": [{"name": "m", "input_per_million": 10.0, "output_per_million": 20.0}]}
        )

        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: cheap)
        tracker = CostTracker()
        tracker.record_response("m", _chat_usage(1_000_000, 1_000_000))
        assert tracker.total_cost_usd == pytest.approx(3.0)

        # Correct the rate in models.yaml -- the same counted tokens reprice,
        # with no re-run and no drift from per-call accumulation.
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: dear)
        assert tracker.total_cost_usd == pytest.approx(30.0)

    def test_price_added_after_the_fact_resolves_unknown_cost(self, monkeypatch):
        empty = PricingTable({})
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: empty)
        tracker = CostTracker()
        tracker.record_response("m", _chat_usage(1_000_000, 0))
        assert tracker.total_cost_usd is None
        assert tracker.unpriced_models == ["m"]

        filled = PricingTable(
            {"openai": [{"name": "m", "input_per_million": 4.0, "output_per_million": 8.0}]}
        )
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: filled)
        assert tracker.total_cost_usd == pytest.approx(4.0)
        assert tracker.unpriced_models == []

    def test_same_model_two_providers_stays_separate(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(1_000_000, 0), api_base="https://api.openai.com/v1"
        )
        tracker.record_response(
            "openai/gpt-5.6-luna",
            _chat_usage(1_000_000, 0),
            api_base="https://openrouter.ai/api/v1",
        )

        labels = list(tracker.to_dict()["models"])
        assert labels == ["gpt-5.6-luna", "openrouter/openai/gpt-5.6-luna"] or labels == [
            "openai/gpt-5.6-luna",
            "openrouter/openai/gpt-5.6-luna",
        ]
        # 0.20 direct + 0.10 via OpenRouter, not 2x either rate.
        assert tracker.total_cost_usd == pytest.approx(0.30)

    def test_reported_rates_come_from_the_table(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(100, 10), api_base="https://api.openai.com/v1"
        )
        entry = tracker.to_dict()["models"]["openai/gpt-5.6-luna"]
        assert entry["rates_per_million"]["input_per_million"] == 0.20
        assert entry["rates_per_million"]["output_per_million"] == 1.20
        assert entry["cost_usd"] == pytest.approx((100 * 0.20 + 10 * 1.20) / 1_000_000)

    def test_summary_shows_the_applied_rate(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(100, 10), api_base="https://api.openai.com/v1"
        )
        assert "per 1M" in tracker.format_summary()


class TestSummaryBreakdown:
    """Every bucket that costs money is reported, not just a prompt/output pair."""

    def _tracker(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "claude-opus-5",
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=900,
                    cache_creation_input_tokens=200,
                    cache_creation=SimpleNamespace(
                        ephemeral_5m_input_tokens=120,
                        ephemeral_1h_input_tokens=80,
                    ),
                )
            ),
            api_base="https://api.anthropic.com",
        )
        return tracker

    def test_every_billable_bucket_is_printed(self, monkeypatch):
        summary = self._tracker(monkeypatch).format_summary()
        for label, tokens in [
            ("input", 100),
            ("cache read", 900),
            ("cache write", 120),
            ("cache write 1h", 80),
            ("output", 50),
        ]:
            assert f"{label}: {tokens} tokens" in summary, f"{label} missing from:\n{summary}"

    def test_each_bucket_shows_its_own_rate(self, monkeypatch):
        summary = self._tracker(monkeypatch).format_summary()
        # Cache writes cost more than input, reads far less -- the whole point
        # of breaking them out.
        assert "input: 100 tokens @ $5.00 per 1M" in summary
        assert "cache read: 900 tokens @ $0.50 per 1M" in summary
        assert "cache write: 120 tokens @ $6.25 per 1M" in summary
        assert "cache write 1h: 80 tokens @ $10.00 per 1M" in summary
        assert "output: 50 tokens @ $25.00 per 1M" in summary

    def test_bucket_costs_sum_to_the_model_total(self, monkeypatch):
        tracker = self._tracker(monkeypatch)
        row = tracker.costs()[0]
        parts = sum(
            getattr(row.usage, field) * row.pricing.rate_for(field)
            for field in (
                "input_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cache_write_1h_tokens",
                "output_tokens",
            )
        )
        assert row.cost_usd == pytest.approx(parts / 1_000_000)

    def test_zero_buckets_are_omitted(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna", _chat_usage(100, 10), api_base="https://api.openai.com/v1"
        )
        summary = tracker.format_summary()
        assert "cache read" not in summary
        assert "cache write" not in summary
        assert "input: 100 tokens" in summary

    def test_reasoning_is_flagged_as_part_of_output_not_extra(self, monkeypatch):
        monkeypatch.setattr("skydiscover.llm.pricing.load_pricing", lambda *a, **k: TABLE)
        tracker = CostTracker()
        tracker.record_response(
            "gpt-5.6-luna",
            _chat_usage(100, 80, reasoning=60),
            api_base="https://api.openai.com/v1",
        )
        summary = tracker.format_summary()
        assert "reasoning: 60 tokens" in summary
        # Billed once, as output -- never added on top.
        assert tracker.total_cost_usd == pytest.approx((100 * 0.20 + 80 * 1.20) / 1_000_000)

    def test_total_line_lists_the_buckets(self, monkeypatch):
        summary = self._tracker(monkeypatch).format_summary()
        total = [ln for ln in summary.splitlines() if "TOTAL" in ln][0]
        assert "input 100" in total
        assert "cache read 900" in total
        assert "cache write 120" in total
        assert "output 50" in total

    def test_unpriced_model_still_shows_its_buckets(self, monkeypatch):
        monkeypatch.setattr(
            "skydiscover.llm.pricing.load_pricing", lambda *a, **k: PricingTable({})
        )
        tracker = CostTracker()
        tracker.record_response("mystery", _chat_usage(100, 10, cached=40))
        summary = tracker.format_summary()
        assert "input: 60 tokens (rate unknown)" in summary
        assert "cache read: 40 tokens (rate unknown)" in summary
        assert "cost unknown" in summary
