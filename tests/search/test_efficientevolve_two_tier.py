"""Tests for EfficientEvolve's two-tier generation.

An expensive model proposes a strategy in prose; a cheap model turns it into
several programs. The economics only work if one expensive call feeds many
cheap ones, so that ratio is what these tests pin down.
"""

import asyncio
import json
import re
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from skydiscover.api import run_discovery
from skydiscover.config import Config, DatabaseConfig, EfficientEvolveDatabaseConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.runner import Runner
from skydiscover.search.base_database import Program
from skydiscover.search.efficientevolve.controller import (
    _DB_DEFAULTS,
    EfficientEvolveController,
)
from skydiscover.search.efficientevolve.database import EfficientEvolveDatabase
from skydiscover.search.efficientevolve.strategy import (
    Strategy,
    StrategyLedger,
    parse_reference_implementation,
    parse_strategies,
    strategy_request,
)
from skydiscover.search.registry import _CONTROLLER_REGISTRY, _DATABASE_REGISTRY
from skydiscover.search.route import get_discovery_controller  # noqa: F401 - registers routes
from skydiscover.search.utils.budget_curve import BudgetCurve


def test_config_defaults_to_one_plan_feeding_several_implementations():
    config = Config.from_dict({"search": {"type": "efficientevolve"}})
    db = config.search.database

    assert isinstance(db, EfficientEvolveDatabaseConfig)
    assert db.implementations_per_strategy == 3
    assert db.strategies_per_guide_call == 3
    assert db.initial_strategies_per_guide_call == 1
    assert db.adaptive_implementation_racing is True
    assert db.adaptive_unguided_racing is True
    assert db.pilot_candidates == 2
    assert db.pilot_score_ratio == 1.0
    assert db.stop_racing_on_improvement is True
    assert db.adaptive_pilot_sizing is True
    assert db.repair_failed_pilots is False
    assert db.opening_cascade is True
    assert db.opening_cheap_candidates == 1
    assert db.opening_guide_reasoning_effort == "low"
    assert db.opening_guide_defer_target_ratio == pytest.approx(0.9)
    assert db.target_score is None
    assert db.target_score_tolerance == pytest.approx(1e-9)
    assert db.strategy_when == "reuse_on_improvement"
    assert db.strategy_reference_candidate is True
    assert db.strategy_patience == 1
    assert db.improvement_epsilon == 1e-9
    assert db.initial_strategy_reasoning_effort is None
    assert db.strategy_reasoning_effort == "low"
    assert db.strategy_escalation_reasoning_effort == "high"
    assert db.strategy_escalation_patience == 32
    assert db.strategy_escalation_interval == 32
    assert db.strategy_escalation_backoff == 2.0
    assert db.guide_implementation_on_escalation is True
    assert db.guide_implementation_reasoning_effort == "medium"
    assert db.guide_implementation_after_productive_stall is False
    assert db.target_aware_guide_promotion_ratio == pytest.approx(0.98)
    assert db.unguided_guide_hedge_interval == 3
    assert db.unguided_guide_hedge_warmup_calls == 2
    assert db.min_guide_amortization_iterations == 2
    assert db.long_horizon_scheduler is True
    assert db.strategy_replays_per_portfolio == 1
    assert db.strategy_replay_min_productive_rounds == 2
    assert db.strategy_replay_cooldown == 2
    assert db.strategy_replay_ucb_exploration == pytest.approx(0.15)
    assert db.strategy_history_size == 8
    assert db.long_horizon_exploration_interval == 12
    assert db.long_horizon_exploration_candidates == 1
    assert db.reasoning_effort == "low"
    assert _DATABASE_REGISTRY["efficientevolve"] is EfficientEvolveDatabase
    assert _CONTROLLER_REGISTRY["efficientevolve"] is EfficientEvolveController


def test_a_plain_database_config_still_gets_the_documented_defaults():
    """Setting search.type after the config is built leaves a plain DatabaseConfig.

    The controller must then still use EfficientEvolve's documented defaults
    rather than a second set hardcoded at the read sites.
    """
    documented = {f.name: f.default for f in fields(EfficientEvolveDatabaseConfig)}
    assert _DB_DEFAULTS == documented

    controller = object.__new__(EfficientEvolveController)
    controller.config = SimpleNamespace(search=SimpleNamespace(database=DatabaseConfig()))

    assert controller._setting("implementations_per_strategy") == 3
    assert controller._setting("strategy_reasoning_effort") == "low"
    assert controller._setting("reasoning_effort") == "low"


class TwoTierFakePool:
    """Fake pool that answers strategy and implementation prompts differently."""

    strategy_calls = 0
    impl_calls = 0
    impl_prompts: list = []
    strategy_prompts: list = []
    impl_kwargs: list = []
    strategy_kwargs: list = []
    impl_models: list = []
    strategy_models: list = []
    unusable_strategy_calls: set = set()
    unusable_impl_calls: set = set()
    cheap_impl_value = None
    strong_impl_value = None
    reference_impl_value = 8

    def __init__(self, models_cfg):
        self.name = models_cfg[0].name

    async def generate(self, system_message, messages, **kwargs):
        text = messages[0]["content"]
        if "A strategy is a **plan, not code**" in text:
            type(self).strategy_calls += 1
            type(self).strategy_prompts.append(text)
            type(self).strategy_kwargs.append(kwargs)
            type(self).strategy_models.append(self.name)
            index = type(self).strategy_calls
            if index in type(self).unusable_strategy_calls:
                return LLMResponse(text="")
            match = re.search(r"exactly (\d+) block", text)
            count = int(match.group(1)) if match else 1
            first = (index - 1) * count + 1
            response = "".join(
                    f"<STRATEGY_TITLE>plan {number}</STRATEGY_TITLE>"
                    f"<STRATEGY>Apply idea {number} thoroughly.</STRATEGY>"
                    for number in range(first, first + count)
                )
            if "<REFERENCE_IMPLEMENTATION>" in text:
                response += (
                    "<REFERENCE_IMPLEMENTATION>```python\n"
                    f"def solve():\n    return {type(self).reference_impl_value}\n"
                    "```</REFERENCE_IMPLEMENTATION>"
                )
            return LLMResponse(text=response)
        type(self).impl_calls += 1
        type(self).impl_prompts.append(text)
        type(self).impl_kwargs.append(kwargs)
        type(self).impl_models.append(self.name)
        if type(self).impl_calls in type(self).unusable_impl_calls:
            return LLMResponse(text="")
        value = type(self).impl_calls
        if self.name == "cheap" and type(self).cheap_impl_value is not None:
            value = type(self).cheap_impl_value
        if self.name == "expensive" and type(self).strong_impl_value is not None:
            value = type(self).strong_impl_value
        return LLMResponse(text=f"```python\ndef solve():\n    return {value}\n```")

    @classmethod
    def reset(cls):
        cls.strategy_calls = cls.impl_calls = 0
        cls.impl_prompts = []
        cls.strategy_prompts = []
        cls.impl_kwargs = []
        cls.strategy_kwargs = []
        cls.impl_models = []
        cls.strategy_models = []
        cls.unusable_strategy_calls = set()
        cls.unusable_impl_calls = set()
        cls.cheap_impl_value = None
        cls.strong_impl_value = None
        cls.reference_impl_value = 8


def _run(
    tmp_path,
    iterations=4,
    implementations=3,
    strategy_patience=1,
    strategy_when="reuse_on_improvement",
    strategies_per_guide=3,
    initial_strategies_per_guide=3,
    racing=True,
    pilot_candidates=2,
    repair_failed_pilots=True,
    unusable_strategy_calls=(),
    unusable_impl_calls=(),
    cheap_impl_value=None,
    strong_impl_value=None,
    reference_impl_value=None,
    strategy_reference_candidate=False,
    long_horizon_scheduler=False,
    stop_racing_on_improvement=False,
    adaptive_pilot_sizing=False,
    long_horizon_exploration_interval=0,
    opening_cascade=False,
    adaptive_unguided_racing=False,
):
    """Run a short two-tier discovery against the fake pool.

    The shipped policy reuses a strategy while it improves the global best.
    """
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "def evaluate(program_path):\n"
        "    body = open(program_path).read()\n"
        "    n = int(body.strip().rsplit(' ', 1)[-1])\n"
        "    return {'combined_score': min(0.9, 0.1 * n), 'validity': 1.0}\n"
    )
    seed = tmp_path / "seed.py"
    seed.write_text("def solve():\n    return 0\n")

    config = Config.from_dict(
        {
            "max_iterations": iterations,
            "diff_based_generation": False,
            "monitor": {"enabled": False},
            "search": {
                "type": "efficientevolve",
                "database": {
                    "implementations_per_strategy": implementations,
                    "strategy_patience": strategy_patience,
                    "strategy_when": strategy_when,
                    "strategies_per_guide_call": strategies_per_guide,
                    "initial_strategies_per_guide_call": initial_strategies_per_guide,
                    "adaptive_implementation_racing": racing,
                    "adaptive_unguided_racing": adaptive_unguided_racing,
                    "pilot_candidates": pilot_candidates,
                    "repair_failed_pilots": repair_failed_pilots,
                    "strategy_reference_candidate": strategy_reference_candidate,
                    # Existing integration tests isolate the post-opening
                    # strategy state machine. The cascade has focused tests.
                    "opening_cascade": opening_cascade,
                    # Most tests below pin the pre-long-horizon state machine.
                    # Dedicated tests enable each new allocation mechanism.
                    "long_horizon_scheduler": long_horizon_scheduler,
                    "stop_racing_on_improvement": stop_racing_on_improvement,
                    "adaptive_pilot_sizing": adaptive_pilot_sizing,
                    "long_horizon_exploration_interval": (
                        long_horizon_exploration_interval
                    ),
                },
            },
            "evaluator": {"evaluation_file": str(evaluator)},
            "llm": {
                "models": [{"name": "cheap", "api_key": "k", "api_base": "http://x"}],
                "guide_models": [{"name": "expensive", "api_key": "k", "api_base": "http://x"}],
            },
        }
    )
    TwoTierFakePool.reset()
    TwoTierFakePool.unusable_strategy_calls = set(unusable_strategy_calls)
    TwoTierFakePool.unusable_impl_calls = set(unusable_impl_calls)
    TwoTierFakePool.cheap_impl_value = cheap_impl_value
    TwoTierFakePool.strong_impl_value = strong_impl_value
    if reference_impl_value is not None:
        TwoTierFakePool.reference_impl_value = reference_impl_value
    with patch("skydiscover.search.default_discovery_controller.LLMPool", TwoTierFakePool):
        return run_discovery(
            evaluator=str(evaluator),
            initial_program=str(seed),
            config=config,
            output_dir=str(tmp_path / "out"),
            cleanup=False,
        )


def test_one_strategy_feeds_many_implementations(tmp_path):
    """The whole point: the expensive rate is charged per plan, not per program."""
    _run(tmp_path, iterations=4, implementations=3)

    assert TwoTierFakePool.strategy_calls == 1
    assert TwoTierFakePool.impl_calls == 12  # 4 rounds x 3 implementations


def test_opening_cascade_records_cheap_probe_then_low_effort_direct_guide(tmp_path):
    result = _run(
        tmp_path,
        iterations=2,
        implementations=3,
        opening_cascade=True,
        cheap_impl_value=4,
        strong_impl_value=8,
    )

    assert result.best_score == pytest.approx(0.8)
    assert TwoTierFakePool.strategy_calls == 0
    assert TwoTierFakePool.impl_models == ["cheap", "expensive"]
    assert [kwargs.get("reasoning_effort") for kwargs in TwoTierFakePool.impl_kwargs] == [
        "low",
        "low",
    ]


def test_direct_cheap_racing_stops_after_an_improving_pilot():
    controller = object.__new__(EfficientEvolveController)
    controller._guide_implementation_due = False
    controller._guide_implementation_unguided = False
    controller._last_guide_implementation_spend = 0.0
    controller._last_guide_implementation_count = 0
    controller._last_unguided_guide_count = 0
    controller.adaptive_implementation_racing = True
    controller.adaptive_unguided_racing = True
    controller.improvement_epsilon = 1e-9
    controller.curve = BudgetCurve()
    controller.curve.observe(0.5, iteration=0, candidates=0)
    calls = []

    async def generate_one(**kwargs):
        calls.append(kwargs["variant"])
        return SimpleNamespace(
            error=None,
            child_program_dict={"metrics": {"combined_score": 0.8}},
        )

    controller._generate_one = generate_one
    sample = (None, None, None, {}, [], [])
    outcomes = asyncio.run(controller._implement_strategy(3, None, 3, sample=sample))

    assert calls == [1]
    assert len(outcomes) == 1


def test_non_improving_round_uses_the_next_prepaid_strategy(tmp_path):
    # The fake score reaches its 0.9 ceiling in round 3. Round 4 therefore
    # schedules one guide implementation of the productive strategy; after
    # that also stalls in round 5, round 6 activates prepaid strategy 2.
    _run(tmp_path, iterations=6, implementations=3)

    assert TwoTierFakePool.strategy_calls == 1
    assert TwoTierFakePool.impl_calls == 18
    assert any("Apply idea 2 thoroughly" in prompt for prompt in TwoTierFakePool.impl_prompts)


def test_guide_is_called_again_only_after_the_portfolio_is_exhausted(tmp_path):
    _run(tmp_path, iterations=9, implementations=3, strategies_per_guide=3)

    assert TwoTierFakePool.strategy_calls == 2


def test_successful_unguided_hedge_runs_before_and_skips_unused_portfolio(tmp_path):
    result = _run(
        tmp_path,
        iterations=7,
        implementations=3,
        strategy_when="stalled",
        strategies_per_guide=3,
        initial_strategies_per_guide=3,
        cheap_impl_value=1,
        strong_impl_value=9,
    )

    assert result.best_score == pytest.approx(0.9)
    # The first portfolio supplies plans 1..3.  At the next refresh, the
    # scheduled unguided guide implementation improves directly, so no second
    # prose portfolio is bought.
    assert TwoTierFakePool.strategy_calls == 1
    strong_calls = [
        prompt
        for model, prompt in zip(TwoTierFakePool.impl_models, TwoTierFakePool.impl_prompts)
        if model == "expensive"
    ]
    assert len(strong_calls) == 1
    assert "# Strong-model baseline hedge" in strong_calls[0]
    # Improvement under no strategy keeps that same state for the following
    # Luna round, anchored to the just-improved incumbent.
    assert "Apply idea" not in TwoTierFakePool.impl_prompts[-1]
    assert "return 9" in TwoTierFakePool.impl_prompts[-1]


def test_hundred_iteration_plateau_amortizes_guide_calls(tmp_path):
    result = _run(
        tmp_path,
        iterations=100,
        implementations=1,
        strategies_per_guide=3,
        initial_strategies_per_guide=1,
        racing=False,
        repair_failed_pilots=False,
    )

    assert result.best_score == 0.9
    assert TwoTierFakePool.impl_calls == 100
    # One productive plan covers iterations 1..10. Thereafter most guide calls
    # supply three one-round plans. A failed pre-portfolio hedge consumes that
    # iteration's single implementation slot, so the newly bought plan starts
    # next iteration and amortizes the portfolio even further.
    assert TwoTierFakePool.strategy_calls == 28
    efforts = [kwargs.get("reasoning_effort") for kwargs in TwoTierFakePool.strategy_kwargs]
    high_effort_calls = efforts.count("high")
    assert high_effort_calls > 0
    assert efforts.count("high") < efforts.count("low")
    # Sparse periodic unguided hedges preserve a baseline-like Terra code lane;
    # exponentially backed-off high-effort portfolios also get a guided Terra
    # implementation. Most code calls remain cheap.
    expensive_implementations = TwoTierFakePool.impl_models.count("expensive")
    assert expensive_implementations > high_effort_calls
    assert expensive_implementations < TwoTierFakePool.impl_models.count("cheap")
    assert any(
        model == "expensive" and "# Strong-model baseline hedge" in prompt
        for model, prompt in zip(TwoTierFakePool.impl_models, TwoTierFakePool.impl_prompts)
    )
    strong_kwargs = [
        kwargs
        for model, kwargs in zip(TwoTierFakePool.impl_models, TwoTierFakePool.impl_kwargs)
        if model == "expensive"
    ]
    assert all(kwargs.get("reasoning_effort") == "medium" for kwargs in strong_kwargs)


def test_long_horizon_replay_exploits_once_then_forces_fresh_guide(tmp_path):
    _run(
        tmp_path,
        iterations=15,
        implementations=1,
        strategies_per_guide=3,
        initial_strategies_per_guide=3,
        long_horizon_scheduler=True,
    )

    assert any(
        "# Strategy to implement: replay plan 1" in prompt
        for prompt in TwoTierFakePool.impl_prompts
    )
    # One replay is allowed for the first paid portfolio. Once it also stalls,
    # the quota is exhausted and a second guide portfolio is necessarily paid.
    assert TwoTierFakePool.strategy_calls == 2


def test_unguided_guide_hedge_schedule_is_low_density():
    controller = object.__new__(EfficientEvolveController)
    controller.unguided_guide_hedge_interval = 3
    controller.unguided_guide_hedge_warmup_calls = 2

    due = [controller._unguided_guide_hedge_due(call) for call in range(1, 10)]

    assert due == [False, True, False, False, True, False, False, True, False]


def test_runner_resume_keeps_curve_and_does_not_rebuy_active_strategy(tmp_path):
    evaluator = tmp_path / "resume_evaluator.py"
    evaluator.write_text(
        "def evaluate(program_path):\n"
        "    body = open(program_path).read()\n"
        "    n = int(body.strip().rsplit(' ', 1)[-1])\n"
        "    return {'combined_score': min(0.9, 0.1 * n), 'validity': 1.0}\n"
    )
    seed = tmp_path / "resume_seed.py"
    seed.write_text("def solve():\n    return 0\n")
    config = Config.from_dict(
        {
            "max_iterations": 2,
            "checkpoint_interval": 1,
            "diff_based_generation": False,
            "monitor": {"enabled": False},
            "search": {
                "type": "efficientevolve",
                "database": {
                    "strategy_when": "reuse_on_improvement",
                        # This test isolates persistence of an active Luna plan;
                        # reference-candidate behavior is covered separately.
                        "opening_cascade": False,
                        "strategy_reference_candidate": False,
                    "implementations_per_strategy": 3,
                    "initial_strategies_per_guide_call": 1,
                    "strategies_per_guide_call": 3,
                    "adaptive_implementation_racing": False,
                },
            },
            "evaluator": {"evaluation_file": str(evaluator)},
            "llm": {
                "models": [{"name": "cheap", "api_key": "k", "api_base": "http://x"}],
                "guide_models": [{"name": "expensive", "api_key": "k", "api_base": "http://x"}],
            },
        }
    )
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    TwoTierFakePool.reset()

    with patch("skydiscover.search.default_discovery_controller.LLMPool", TwoTierFakePool):
        first = Runner(
            evaluation_file=str(evaluator),
            initial_program_path=str(seed),
            config=config,
            output_dir=str(first_output),
        )
        asyncio.run(first.run(iterations=2))
        checkpoint = first_output / "checkpoints" / "checkpoint_2"
        assert (checkpoint / "efficientevolve_state.json").exists()
        assert TwoTierFakePool.strategy_calls == 1

        resumed = Runner(
            evaluation_file=str(evaluator),
            initial_program_path=str(seed),
            config=config,
            output_dir=str(second_output),
        )
        asyncio.run(resumed.run(iterations=2, checkpoint_path=str(checkpoint)))

    # The active productive plan crosses the process-style resume boundary;
    # buying a second guide plan here would prove the controller state was lost.
    assert TwoTierFakePool.strategy_calls == 1
    curve = [
        json.loads(line) for line in (second_output / "budget_curve.jsonl").read_text().splitlines()
    ]
    assert len(curve) == 5  # one seed + four logical iterations across both runs
    assert [point["iteration"] for point in curve] == [0, 1, 2, 3, 4]


def test_failed_pilots_use_one_feedback_guided_repair(tmp_path):
    _run(
        tmp_path,
        iterations=1,
        implementations=3,
        unusable_impl_calls=(1, 2, 3),
    )

    assert TwoTierFakePool.strategy_calls == 1
    assert TwoTierFakePool.impl_calls == 3
    repair_prompt = TwoTierFakePool.impl_prompts[-1]
    assert "Previous Failed Attempts" in repair_prompt
    assert "Empty implementation response" in repair_prompt


def test_one_successful_pilot_launches_the_rest_of_the_batch(tmp_path):
    _run(tmp_path, iterations=1, implementations=3, unusable_impl_calls=(1,))

    assert TwoTierFakePool.strategy_calls == 1
    assert TwoTierFakePool.impl_calls == 3


def test_improving_pilot_stops_before_stale_parent_expansion(tmp_path):
    result = _run(
        tmp_path,
        iterations=1,
        implementations=3,
        stop_racing_on_improvement=True,
    )

    assert result.best_score == pytest.approx(0.2)
    assert TwoTierFakePool.impl_calls == 2


def test_reliable_productive_operator_uses_one_pilot():
    controller = object.__new__(EfficientEvolveController)
    controller.pilot_candidates = 2
    controller.adaptive_pilot_sizing = True
    controller.reliable_pilot_min_attempts = 4
    controller.reliable_pilot_usable_ratio = 0.75
    strategy = Strategy(
        index=1,
        title="reliable",
        plan="plan",
        rounds=2,
        attempts=4,
        implementations=4,
        cumulative_relative_gain=0.1,
    )
    controller.ledger = StrategyLedger(entries=[strategy])

    assert controller._adaptive_pilot_count(strategy, available=3) == 1

    strategy.implementations = 2
    assert controller._adaptive_pilot_count(strategy, available=3) == 2


def test_pilot_gate_treats_float_jitter_as_equal():
    controller = object.__new__(EfficientEvolveController)
    controller.curve = SimpleNamespace(incumbent=0.9644862832020152)
    controller.pilot_score_ratio = 1.0
    strategy = Strategy(index=1, title="test", plan="test")
    outcome = SimpleNamespace(
        error=None,
        child_program_dict={"metrics": {"combined_score": 0.9644862832020151}},
    )

    assert controller._pilot_is_promising(outcome, strategy) is True


def test_the_strategy_reaches_the_implementer(tmp_path):
    _run(tmp_path, iterations=2, implementations=2)

    joined = "\n".join(TwoTierFakePool.impl_prompts)
    assert "copy every SEARCH block byte-for-byte" in joined
    assert "An unmatched SEARCH block produces no candidate" in joined
    assert "# Strategy to implement" in joined
    assert "Apply idea 1 thoroughly" in joined
    assert "Implement exactly this strategy" in joined


def test_each_round_uses_the_guide_then_implementation_model_pool(tmp_path):
    _run(tmp_path, iterations=2, implementations=2)

    assert TwoTierFakePool.strategy_models == ["expensive"]
    assert TwoTierFakePool.impl_models == ["cheap"] * 4
    assert all(call["reasoning_effort"] == "low" for call in TwoTierFakePool.strategy_kwargs)
    assert all(call["reasoning_effort"] == "low" for call in TwoTierFakePool.impl_kwargs)


def test_strategy_and_implementations_share_the_same_sampled_parent(tmp_path):
    _run(tmp_path, iterations=1, implementations=3)

    seed = "def solve():\n    return 0"
    assert seed in TwoTierFakePool.strategy_prompts[0]
    assert all(seed in prompt for prompt in TwoTierFakePool.impl_prompts)


def test_an_unusable_strategy_records_the_round_and_next_iteration_recovers(tmp_path):
    result = _run(
        tmp_path,
        iterations=2,
        implementations=2,
        unusable_strategy_calls=(1,),
    )

    assert result.best_score > 0
    assert TwoTierFakePool.strategy_calls == 2
    assert TwoTierFakePool.impl_calls == 2


def test_implementations_are_told_which_variant_they_are(tmp_path):
    _run(tmp_path, iterations=1, implementations=3)

    joined = "\n".join(TwoTierFakePool.impl_prompts)
    assert "variant 1 of 3" in joined
    assert "variant 3 of 3" in joined


def test_strategy_portfolio_parser_keeps_ranked_plans():
    response = (
        "<STRATEGY_TITLE>first</STRATEGY_TITLE><STRATEGY>Plan A.</STRATEGY>"
        "<STRATEGY_TITLE>second</STRATEGY_TITLE><STRATEGY>Plan B.</STRATEGY>"
    )
    strategies = parse_strategies(response, start_index=4, limit=3)

    assert [(strategy.index, strategy.title, strategy.plan) for strategy in strategies] == [
        (4, "first", "Plan A."),
        (5, "second", "Plan B."),
    ]


def test_reference_implementation_is_parsed_outside_strategy_blocks():
    response = (
        "<STRATEGY_TITLE>first</STRATEGY_TITLE><STRATEGY>Plan A.</STRATEGY>"
        "<REFERENCE_IMPLEMENTATION>```python\ndef solve():\n    return 8\n```"
        "</REFERENCE_IMPLEMENTATION>"
    )
    assert "return 8" in parse_reference_implementation(response)
    assert parse_reference_implementation("<STRATEGY>plan only</STRATEGY>") is None


def test_optional_guide_reference_is_evaluated_without_an_extra_api_call(tmp_path):
    result = _run(
        tmp_path,
        iterations=1,
        implementations=1,
        cheap_impl_value=1,
        strategy_reference_candidate=True,
    )

    assert result.best_score == pytest.approx(0.8)
    assert TwoTierFakePool.strategy_calls == 1
    assert TwoTierFakePool.impl_calls == 0
    assert "<REFERENCE_IMPLEMENTATION>" in TwoTierFakePool.strategy_prompts[0]


def test_non_improving_guide_reference_still_launches_luna(tmp_path):
    result = _run(
        tmp_path,
        iterations=1,
        implementations=1,
        cheap_impl_value=7,
        reference_impl_value=0,
        strategy_reference_candidate=True,
    )

    assert result.best_score == pytest.approx(0.7)
    assert TwoTierFakePool.strategy_calls == 1
    assert TwoTierFakePool.impl_calls == 1


def test_strategy_outcome_reports_usable_yield():
    strategy = Strategy(
        index=1,
        title="candidate",
        plan="plan",
        baseline_score=0.8,
        best_score=0.7,
        attempts=3,
        implementations=1,
    )

    assert "1/3 implementations usable" in strategy.outcome()

    strategy.record_failure("  invalid   geometry  ")
    assert "failures: invalid geometry" in strategy.outcome()


def test_long_guide_history_keeps_early_high_value_operators():
    productive = Strategy(
        index=1,
        title="early breakthrough",
        plan="plan",
        rounds=1,
        attempts=2,
        cumulative_relative_gain=0.4,
    )
    recent = [
        Strategy(index=index, title=f"recent {index}", plan="plan", rounds=1, attempts=1)
        for index in range(2, 12)
    ]
    ledger = StrategyLedger(entries=[productive, *recent])

    text = ledger.history_text(limit=4)

    assert "early breakthrough" in text
    assert "high-value operator" in text
    assert "recent 11" in text
    assert "recent 2" not in text


def test_strategy_effort_escalates_with_backoff_during_a_sustained_plateau():
    controller = object.__new__(EfficientEvolveController)
    controller.ledger = StrategyLedger(entries=[Strategy(index=1, title="tried", plan="plan")])
    controller.initial_strategy_reasoning_effort = None
    controller.strategy_reasoning_effort = "medium"
    controller.strategy_escalation_reasoning_effort = "high"
    controller.strategy_escalation_patience = 8
    controller.strategy_escalation_interval = 8
    controller.strategy_escalation_backoff = 2.0
    controller._next_strategy_escalation = 8
    controller._strategy_escalation_gap = 8

    controller._stagnation = 7
    assert controller._current_strategy_reasoning_effort() == "medium"
    controller._stagnation = 8
    assert controller._current_strategy_reasoning_effort() == "high"
    controller._stagnation = 12
    assert controller._current_strategy_reasoning_effort() == "medium"
    controller._stagnation = 16
    assert controller._current_strategy_reasoning_effort() == "high"
    controller._stagnation = 24
    assert controller._current_strategy_reasoning_effort() == "medium"
    controller._stagnation = 32
    assert controller._current_strategy_reasoning_effort() == "high"


def test_hundred_iteration_plateau_does_not_permanently_escalate_guide():
    controller = object.__new__(EfficientEvolveController)
    controller.ledger = StrategyLedger(entries=[Strategy(index=1, title="tried", plan="plan")])
    controller.initial_strategy_reasoning_effort = None
    controller.strategy_reasoning_effort = "medium"
    controller.strategy_escalation_reasoning_effort = "high"
    controller.strategy_escalation_patience = 8
    controller.strategy_escalation_interval = 8
    controller.strategy_escalation_backoff = 2.0
    controller._next_strategy_escalation = 8
    controller._strategy_escalation_gap = 8

    efforts = []
    # A full three-plan portfolio makes a new guide call roughly every three
    # rounds on a complete plateau. Model that 100-iteration cadence directly.
    for stagnation in range(0, 100, 3):
        controller._stagnation = stagnation
        efforts.append(controller._current_strategy_reasoning_effort())

    assert efforts.count("high") == 4
    assert efforts.count("medium") == 30
    assert efforts[-1] == "medium"
    assert not any(effort == "high" for effort in efforts[1:3])


def test_improvement_resets_the_periodic_strategy_escalation():
    controller = object.__new__(EfficientEvolveController)
    controller._stagnation = 12
    controller.strategy_escalation_patience = 8
    controller.strategy_escalation_interval = 8
    controller._next_strategy_escalation = 16
    controller._strategy_escalation_gap = 16
    controller._restore_exploration_ratio = lambda: None

    controller._update_adaptation(iteration=4, productive=True)

    assert controller._stagnation == 0
    assert controller._next_strategy_escalation == 8
    assert controller._strategy_escalation_gap == 8


def test_first_strategy_call_can_use_a_distinct_reasoning_effort():
    controller = object.__new__(EfficientEvolveController)
    controller.ledger = StrategyLedger()
    controller.initial_strategy_reasoning_effort = "medium"
    controller.strategy_reasoning_effort = "low"
    controller.strategy_escalation_reasoning_effort = "high"
    controller.strategy_escalation_patience = 32
    controller.strategy_escalation_interval = 32
    controller.strategy_escalation_backoff = 2.0
    controller._stagnation = 0
    controller._next_strategy_escalation = 32
    controller._strategy_escalation_gap = 32

    assert controller._current_strategy_reasoning_effort() == "medium"

    controller.ledger.add(Strategy(index=1, title="opening", plan="plan"))
    assert controller._current_strategy_reasoning_effort() == "low"


def test_final_unamortized_guide_call_replays_the_best_strategy():
    controller = object.__new__(EfficientEvolveController)
    controller.strategy_when = "reuse_on_improvement"
    controller.min_guide_amortization_iterations = 2
    controller._progress = (9, 10)
    controller.curve = SimpleNamespace(incumbent=0.9)
    controller.ledger = SimpleNamespace(
        entries=[
            Strategy(
                index=1,
                title="weak",
                plan="weak plan",
                rounds=1,
                attempts=2,
                implementations=2,
                best_score=0.7,
            ),
            Strategy(
                index=2,
                title="best",
                plan="best plan",
                rounds=1,
                attempts=2,
                implementations=1,
                best_score=0.9,
            ),
        ],
        add=lambda strategy: controller.ledger.entries.append(strategy),
    )

    replay = controller._strategy_replay_when_guide_cannot_amortize(iteration=10)

    assert replay is not None
    assert replay.plan == "best plan"
    assert replay.cost == 0.0
    assert replay.baseline_score == 0.9
    assert replay in controller.ledger.entries


def test_portfolio_starts_narrow_and_does_not_prebuy_past_the_run():
    controller = object.__new__(EfficientEvolveController)
    controller.strategy_when = "reuse_on_improvement"
    controller.initial_strategies_per_guide_call = 1
    controller.strategies_per_guide_call = 3
    controller.ledger = SimpleNamespace(entries=[])
    controller._progress = (0, 10)

    assert controller._guide_portfolio_size() == 1

    # If cheap bootstrap rounds already moved the curve before the first guide
    # call, this is a plateau portfolio and should be amortized immediately.
    controller.ledger.entries.clear()
    controller.curve = SimpleNamespace(points=[object(), object(), object()])
    assert controller._guide_portfolio_size() == 3

    controller.ledger.entries.append(object())
    controller._progress = (8, 10)
    assert controller._guide_portfolio_size() == 2
    controller._progress = (9, 10)
    assert controller._guide_portfolio_size() == 1

    controller.strategy_when = "always"
    controller._progress = (0, 10)
    assert controller._guide_portfolio_size() == 1


def test_new_strategy_is_anchored_to_global_incumbent():
    weak = Program(id="weak-parent", solution="weak", metrics={"combined_score": 0.2})
    best = Program(id="global-best", solution="best", metrics={"combined_score": 0.9})
    diverse = Program(id="diverse", solution="other", metrics={"combined_score": 0.5})
    controller = object.__new__(EfficientEvolveController)
    controller.database = SimpleNamespace(
        programs={program.id: program for program in (weak, best, diverse)},
        get_best_program=lambda: best,
    )
    sample = (
        weak,
        weak,
        ("", weak.id),
        {"": [best, diverse]},
        [best.id, diverse.id],
        [("", best.id), ("", diverse.id)],
    )

    raw_parent, parent, parent_info, context, context_ids, _ = (
        controller._anchor_sample_to_incumbent(sample)
    )

    assert raw_parent is best
    assert parent is best
    assert parent_info == ("", best.id)
    assert best.id not in context_ids
    assert set(context_ids) == {weak.id, diverse.id}
    assert weak in context["exploration_parent"]


def test_productive_reused_strategy_stays_anchored_to_incumbent():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller._anchor_next_cheap_to_incumbent = False
    controller._strategy = Strategy(index=1, title="productive", plan="continue")

    assert controller._should_anchor_strategy_parent(wants_new_strategy=False) is True

    controller._strategy = None
    assert controller._should_anchor_strategy_parent(wants_new_strategy=False) is False
    controller._anchor_next_cheap_to_incumbent = True
    assert controller._should_anchor_strategy_parent(wants_new_strategy=False) is True
    controller._anchor_next_cheap_to_incumbent = False
    assert controller._should_anchor_strategy_parent(wants_new_strategy=True) is True

    controller.two_tier = False
    assert controller._should_anchor_strategy_parent(wants_new_strategy=True) is False


def test_post_hedge_cheap_anchor_overrides_opening_strategy_policy():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.strategy_when = "reuse_on_improvement"
    controller._strategy = None
    controller._barren_rounds = 0
    controller._guide_implementation_due = False
    controller._anchor_next_cheap_to_incumbent = True

    assert controller._wants_new_strategy() is False

    controller._anchor_next_cheap_to_incumbent = False
    assert controller._wants_new_strategy() is True


def test_the_strategist_is_asked_for_a_plan_not_code(tmp_path):
    _run(tmp_path, iterations=1, implementations=1)

    prompt = TwoTierFakePool.strategy_prompts[0]
    assert "**plan, not code**" in prompt
    assert "Do not write the\nprogram" in prompt
    assert "Do not depend on an offline solve" in prompt
    assert "preserves the problem's\nhard validity constraints" in prompt
    assert "local, verifiable refinement path" in prompt


def test_strategy_prompt_compares_metrics_in_the_same_units():
    prompt = strategy_request(
        "def solve(): pass",
        0.9579749786383034,
        StrategyLedger(),
        metrics={
            "validity": 1.0,
            "sum_radii": 2.524264068711929,
            "target_ratio": 0.9579749786383034,
            "combined_score": 0.9579749786383034,
            "ignored": [1, 2, 3],
        },
    )

    assert "# Current evaluated metrics" in prompt
    assert "- sum_radii: 2.52426406871" in prompt
    assert "- combined_score: 0.957974978638" in prompt
    assert "- ignored:" not in prompt
    assert "Compare like with like" in prompt
    assert "never with\n`combined_score`" in prompt
    assert "predicted\n`combined_score`" in prompt
    assert 'An "intended", "designed", "about", or' in prompt


def test_three_strategy_portfolio_covers_local_and_structural_scales():
    prompt = strategy_request(
        "def solve(): pass",
        0.9,
        StrategyLedger(),
        metrics={"combined_score": 0.9},
        count=3,
    )

    assert "highest predicted combined score FIRST" in prompt
    assert "break near-ties" in prompt
    assert "an explicit local exploitation" in prompt
    assert "a mesoscopic subsystem replacement" in prompt
    assert "a high-upside structural exploration" in prompt
    assert "only tiny perturbations" in prompt
    assert "auxiliary\nobjective variable" in prompt
    assert "analytic constraint derivatives" in prompt


def test_reference_candidate_prioritizes_strongest_executable_strategy():
    prompt = strategy_request(
        "def solve():\n    return 1\n",
        None,
        StrategyLedger(),
        count=3,
        include_reference=True,
    )

    assert "strategy 1 and its reference" in prompt
    assert "highest predicted\ncombined score" in prompt
    assert "conservative local proposal" in prompt


def test_live_strategy_prompt_contains_the_incumbent_metric_breakdown(tmp_path):
    _run(tmp_path, iterations=1, implementations=1)

    prompt = TwoTierFakePool.strategy_prompts[0]
    assert "# Current evaluated metrics" in prompt
    assert "- combined_score: 0" in prompt
    assert "- validity: 1" in prompt


def test_children_record_the_strategy_they_came_from(tmp_path):
    result = _run(tmp_path, iterations=2, implementations=2)
    assert result.best_score > 0

    import glob
    import json

    records = [
        json.load(open(path)) for path in glob.glob(f"{tmp_path}/out/checkpoints/*/programs/*.json")
    ]
    children = [r for r in records if r.get("parent_id")]
    assert children
    assert all(c["metadata"].get("two_tier_generation") for c in children)
    assert any(c["metadata"].get("strategy_title") == "plan 1" for c in children)


def test_single_tier_mode_skips_the_strategy_call(tmp_path):
    """two_tier: false is the ablation that isolates the plan's contribution.

    Without it there is no way to tell whether a strategy helps or whether the
    gain is just from drawing more samples per round.
    """
    config = Config.from_dict(
        {"search": {"type": "efficientevolve", "database": {"two_tier": False}}}
    )
    assert config.search.database.two_tier is False


def test_two_tier_is_on_by_default():
    config = Config.from_dict({"search": {"type": "efficientevolve"}})
    assert config.search.database.two_tier is True


def test_the_plan_text_reaches_the_implementation_instruction():
    from skydiscover.search.efficientevolve.strategy import (
        Strategy,
        implementation_instruction,
    )

    strategy = Strategy(index=1, title="grid packing", plan="Use a 5x5 lattice.")
    with_plan = implementation_instruction(strategy, 1, 3)
    assert "grid packing" in with_plan
    assert "Use a 5x5 lattice." in with_plan


# ----------------------------------------------------------------------
# When the plan is bought
# ----------------------------------------------------------------------


def test_default_buys_an_opening_plan_then_reuses_only_an_improving_plan():
    config = Config.from_dict({"search": {"type": "efficientevolve"}})
    assert config.search.database.strategy_when == "reuse_on_improvement"

    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.strategy_when = "reuse_on_improvement"
    controller.strategy_patience = 1
    controller._strategy = None
    controller._barren_rounds = 0
    controller._stagnation = 0
    controller._guide_implementation_due = False
    controller._anchor_next_cheap_to_incumbent = False
    # This test isolates the post-cascade cadence policy.
    controller.opening_cascade = False

    assert controller._wants_new_strategy() is True
    controller._strategy = Strategy(index=1, title="productive", plan="keep it")
    controller._barren_rounds = 0
    assert controller._wants_new_strategy() is False
    controller._barren_rounds = 1
    assert controller._wants_new_strategy() is True


def test_opening_cascade_is_sequential_cheap_then_direct_guide():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.opening_cascade = True
    controller.opening_cheap_candidates = 1
    controller._opening_cheap_probe_done = False
    controller._opening_guide_hedge_done = False
    controller._anchor_next_cheap_to_incumbent = False
    controller._strategy = None
    controller._strategy_queue = []
    controller._strategy_call_count = 0
    controller._strategy_replays_by_guide_call = {}
    controller.long_horizon_scheduler = False
    controller.strategy_when = "reuse_on_improvement"
    controller._barren_rounds = 0
    controller._stagnation = 0
    controller.strategy_patience = 1
    controller._guide_implementation_due = False

    assert controller._opening_cheap_probe_due() is True
    assert controller._wants_new_strategy() is False

    controller._opening_cheap_probe_done = True
    assert controller._opening_guide_hedge_due() is True
    assert controller._wants_new_strategy() is True
    assert controller._pre_strategy_hedge_due() is True

    controller._opening_guide_hedge_done = True
    assert controller._opening_guide_hedge_due() is False
    assert controller._wants_new_strategy() is True


def test_near_target_cheap_opening_defers_guide_only_until_first_stall():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.opening_cascade = True
    controller._opening_cheap_probe_done = True
    controller._opening_guide_hedge_done = False
    controller.opening_guide_defer_target_ratio = 0.9
    controller.target_score = 1.0
    controller.target_score_tolerance = 1e-9
    controller.curve = BudgetCurve()
    controller.curve.observe(0.91, iteration=1, candidates=1)
    controller._stagnation = 0

    assert controller._defer_opening_guide_hedge() is True
    assert controller._opening_guide_hedge_due() is False

    controller._stagnation = 1
    assert controller._defer_opening_guide_hedge() is False
    assert controller._opening_guide_hedge_due() is True


def test_stalled_mode_withholds_the_plan_while_the_cheap_tier_improves():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.strategy_when = "stalled"
    controller.strategy_patience = 2
    controller._strategy = None
    controller._barren_rounds = 0

    controller._stagnation = 0
    assert controller._wants_new_strategy() is False
    controller._stagnation = 1
    assert controller._wants_new_strategy() is False
    controller._stagnation = 2
    assert controller._wants_new_strategy() is True


def test_always_mode_buys_a_new_plan_even_with_an_active_strategy():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.strategy_when = "always"
    controller.strategy_patience = 2
    controller._strategy = Strategy(index=1, title="previous", plan="old")
    controller._barren_rounds = 0
    controller._stagnation = 0

    assert controller._wants_new_strategy() is True


def test_a_spent_plan_is_replaced_regardless_of_mode():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = True
    controller.strategy_when = "stalled"
    controller.strategy_patience = 2
    controller._strategy = Strategy(index=1, title="t", plan="p")
    controller._stagnation = 0

    controller._barren_rounds = 1
    assert controller._wants_new_strategy() is False
    controller._barren_rounds = 2
    assert controller._wants_new_strategy() is True


def test_single_tier_never_buys_a_plan():
    controller = object.__new__(EfficientEvolveController)
    controller.two_tier = False
    controller.strategy_when = "always"
    controller.strategy_patience = 1
    controller._strategy = None
    controller._barren_rounds = 99
    controller._stagnation = 99

    assert controller._wants_new_strategy() is False
