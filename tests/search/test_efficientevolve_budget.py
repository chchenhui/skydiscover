"""Tests for EfficientEvolve's budget-aware control loop (BA-AUC optimization)."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from skydiscover.config import Config
from skydiscover.search.base_database import Program
from skydiscover.search.efficientevolve.controller import EfficientEvolveController
from skydiscover.search.efficientevolve.database import EfficientEvolveDatabase
from skydiscover.search.efficientevolve.strategy import Strategy, StrategyLedger
from skydiscover.search.utils.budget_curve import BudgetCurve


class FakeSpend:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


def make_controller(
    *,
    budget=None,
    spend=None,
    schedule="fixed",
    implementations=3,
    patience=2,
    novelty_memory=24,
    stalled_exploration_ratio=0.6,
    base_exploration_ratio=0.2,
    reasoning_effort="low",
    strategy_when="reuse_on_improvement",
    two_tier=True,
):
    """A controller with only the budget-aware state wired up.

    The batch/LLM machinery is covered in ``test_efficientevolve_batch.py``;
    these tests exercise the control loop's decisions in isolation.
    """
    controller = object.__new__(EfficientEvolveController)
    controller.schedule = schedule
    controller.implementations_per_strategy = implementations
    controller.candidates_first = implementations
    controller.candidates_last = 1
    controller.candidates_per_iteration = implementations
    controller.stagnation_patience = patience
    controller.strategy_patience = patience
    controller.improvement_epsilon = 1e-9
    controller.strategy_when = strategy_when
    controller.two_tier = two_tier
    controller.ledger = StrategyLedger()
    controller._strategy = None
    controller._strategy_queue = []
    controller._strategy_costs = []
    controller._impl_costs = []
    controller._guide_impl_costs = []
    controller._guide_implementation_due = False
    controller._guide_implementation_due_reason = None
    controller.guide_implementation_after_productive_stall = True
    controller._last_guide_implementation_spend = 0.0
    controller._last_guide_implementation_count = 0
    controller._anchor_next_cheap_to_incumbent = False
    controller.long_horizon_scheduler = False
    controller.strategy_replays_per_portfolio = 1
    controller.strategy_replay_min_productive_rounds = 2
    controller.strategy_replay_cooldown = 2
    controller.strategy_replay_ucb_exploration = 0.15
    controller._strategy_call_count = 0
    controller._strategy_replays_by_guide_call = {}
    controller.long_horizon_exploration_interval = 0
    controller.long_horizon_exploration_candidates = 1
    controller._next_long_horizon_exploration = 0
    controller._barren_rounds = 0
    controller.budget = budget
    controller.curve = BudgetCurve(budget=budget, cost_fn=spend or FakeSpend())
    controller.reasoning_effort = reasoning_effort
    controller.novelty_memory = novelty_memory
    controller.stalled_exploration_ratio = stalled_exploration_ratio
    controller._base_exploration_ratio = base_exploration_ratio
    controller._exploring = False
    controller._seen_solutions = set()
    controller._score_ledger = OrderedDict()
    controller._stagnation = 0
    controller._cost_samples = []
    controller._progress = (0, 10)
    controller.database = SimpleNamespace(exploration_ratio=base_exploration_ratio)
    return controller


def outcome(score, solution="def solve():\n    return 1\n"):
    """A successful SerializableResult-shaped stand-in."""
    return SimpleNamespace(
        error=None,
        child_program_dict={"solution": solution, "metrics": {"combined_score": score}},
    )


def unguided_guide_outcome(score, solution="def solve():\n    return 2\n"):
    result = outcome(score, solution)
    result.child_program_dict["metadata"] = {
        "implementation_model_tier": "guide",
        "strategy_index": None,
    }
    return result


def test_known_sufficient_target_stops_only_when_configured():
    controller = make_controller()
    controller.curve.observe(0.9999999995, iteration=1, candidates=1)
    controller.target_score = None
    controller.target_score_tolerance = 1e-9
    assert controller._target_reached() is False

    controller.target_score = 1.0
    assert controller._target_reached() is True

    controller.target_score_tolerance = 0.0
    assert controller._target_reached() is False


def test_target_aware_guide_promotion_is_confined_to_the_last_mile():
    controller = make_controller()
    controller.target_score = 1.0
    controller.target_score_tolerance = 1e-9
    controller.target_aware_guide_promotion_ratio = 0.98
    controller.curve.observe(0.979, iteration=1, candidates=1)
    assert controller._target_aware_guide_promotion_due() is False

    controller.curve.observe(0.981, iteration=2, candidates=1)
    assert controller._target_aware_guide_promotion_due() is True

    controller.target_score = None
    assert controller._target_aware_guide_promotion_due() is False


# ----------------------------------------------------------------------


def test_a_batch_landing_on_a_known_score_is_not_productive():
    """34 novel-looking programs at one score is the plateau, not progress."""
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1)
    controller.curve.observe(0.9)

    spend.value = 0.01
    controller._record_iteration(
        iteration=1,
        candidates=1,
        successful=[(1, outcome(0.9, solution="def solve():\n    return 2\n"))],
        cost_before=0.0,
    )

    assert controller._stagnation == 1
    assert controller.curve.incumbent == pytest.approx(0.9)


def test_a_raised_incumbent_is_productive():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1)
    controller.curve.observe(0.5)
    controller._stagnation = 3

    spend.value = 0.01
    controller._record_iteration(
        iteration=2,
        candidates=2,
        successful=[(1, outcome(0.5)), (2, outcome(0.95, solution="other"))],
        cost_before=0.0,
    )

    assert controller._stagnation == 0
    assert controller.curve.incumbent == pytest.approx(0.95)
    assert controller._anchor_next_cheap_to_incumbent is True


def test_unguided_stall_does_not_extend_incumbent_anchor():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1)
    controller.curve.observe(0.9)

    spend.value = 0.01
    controller._record_iteration(
        iteration=2,
        candidates=1,
        successful=[(1, outcome(0.9))],
        cost_before=0.0,
    )

    assert controller._anchor_next_cheap_to_incumbent is False


def test_guide_candidate_stops_early_only_for_a_real_gain():
    controller = make_controller()
    controller.curve.observe(0.8)

    assert controller._outcome_improves_incumbent(outcome(0.81)) is True
    assert controller._outcome_improves_incumbent(outcome(0.8 + 5e-10)) is False


def test_float_jitter_does_not_reuse_a_strategy_or_reset_stagnation():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1)
    controller.improvement_epsilon = 1e-9
    controller.curve.observe(0.9)
    controller._strategy = Strategy(
        index=1,
        title="jitter",
        plan="plan",
        baseline_score=0.9,
        improvement_epsilon=controller.improvement_epsilon,
    )

    spend.value = 0.01
    controller._record_iteration(
        iteration=1,
        candidates=1,
        successful=[(1, outcome(0.9 + 2.5e-10, solution="tiny jitter"))],
        cost_before=0.0,
    )

    assert controller.curve.incumbent > 0.9
    assert controller._strategy.improved is False
    assert controller._barren_rounds == 1
    assert controller._stagnation == 1


def test_an_improving_unguided_hedge_retains_but_does_not_credit_strategy():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1)
    controller.curve.observe(0.8)
    controller._strategy = Strategy(
        index=1,
        title="queued plan",
        plan="try this with Luna next",
        baseline_score=0.8,
    )
    # The main loop reserves the strategy round before the hedge preempts it.
    controller._strategy.rounds = 1
    controller._last_unguided_guide_count = 1

    spend.value = 0.02
    controller._record_iteration(
        iteration=1,
        candidates=1,
        successful=[(1, unguided_guide_outcome(0.9))],
        cost_before=0.0,
    )

    assert controller.curve.incumbent == pytest.approx(0.9)
    assert controller._strategy.best_score is None
    assert controller._strategy.implementations == 0
    assert controller._strategy.attempts == 0
    assert controller._strategy.baseline_score == pytest.approx(0.9)
    assert controller._strategy.rounds == 0
    assert controller._barren_rounds == 0
    assert controller._wants_new_strategy() is False


def test_an_empty_batch_still_lands_on_the_curve():
    """A call that produced nothing usable cost money; the curve must show it."""
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1)
    controller.curve.observe(0.4)

    spend.value = 0.02
    controller._record_iteration(iteration=1, candidates=3, successful=[], cost_before=0.0)

    points = controller.curve.points
    assert points[-1].cost == pytest.approx(0.02)
    assert points[-1].incumbent == pytest.approx(0.4)
    assert controller._stagnation == 1


# ----------------------------------------------------------------------
# Budget admission control
# ----------------------------------------------------------------------


def test_unbounded_runs_always_admit():
    controller = make_controller(budget=None)
    assert controller._plan_iteration(1) == 3


def test_first_call_is_never_blocked_for_lack_of_history():
    controller = make_controller(budget=0.001)
    assert controller._plan_iteration(1) == 3


def test_exhausted_budget_stops_the_run():
    spend = FakeSpend(value=0.05)
    controller = make_controller(budget=0.05, spend=spend)
    assert controller._plan_iteration(7) is None


def test_a_call_that_would_not_fit_stops_the_run():
    spend = FakeSpend(value=0.049)
    controller = make_controller(budget=0.05, spend=spend)
    controller._impl_costs = [0.01]
    controller._strategy_costs = [0.02]
    assert controller._plan_iteration(9) is None


def test_cost_samples_accumulate_from_observed_spend():
    spend = FakeSpend()
    controller = make_controller(spend=spend)
    controller.curve.observe(0.1)

    spend.value = 0.007
    controller._record_iteration(1, 2, [(1, outcome(0.2))], cost_before=0.0)

    assert controller._impl_costs == [pytest.approx(0.0035)]


def test_implementation_cost_history_excludes_the_strategy_call():
    spend = FakeSpend()
    controller = make_controller(spend=spend, implementations=3)
    controller.curve.observe(0.1)

    # The guide spent 0.010 before the three implementations spent 0.006.
    spend.value = 0.016
    controller._record_iteration(
        1,
        3,
        [(1, outcome(0.2))],
        cost_before=0.0,
        implementation_cost_before=0.010,
    )

    assert controller._impl_costs == [pytest.approx(0.002)]


def test_cheap_cost_history_excludes_a_guide_implementation():
    spend = FakeSpend()
    controller = make_controller(spend=spend, implementations=3)
    controller._last_guide_implementation_count = 1
    controller._last_guide_implementation_spend = 0.03

    spend.value = 0.05
    controller._record_iteration(
        iteration=1,
        candidates=3,
        successful=[],
        cost_before=0.0,
        implementation_cost_before=0.0,
    )

    # The other two implementations cost $0.02 in total.
    assert controller._impl_costs == [pytest.approx(0.01)]


# ----------------------------------------------------------------------
# Novelty ledger
# ----------------------------------------------------------------------


def test_ledger_is_withheld_while_the_search_is_still_improving():
    controller = make_controller()
    controller._register_solution("a", 0.5)
    assert controller._stagnation == 0
    assert controller._novelty_ledger_text() == ""


def test_ledger_reports_repeat_counts_once_stalled():
    controller = make_controller()
    for solution in ("a", "b", "c"):
        controller._register_solution(solution, 0.964486)
    controller._register_solution("d", 0.5)
    controller._stagnation = 2

    text = controller._novelty_ledger_text()

    assert "Already explored" in text
    assert "score 0.964486 (reached 3 times)" in text
    assert "score 0.500000" in text
    assert "reached 1 times" not in text


def test_ledger_can_be_disabled():
    controller = make_controller(novelty_memory=0)
    controller._register_solution("a", 0.5)
    controller._stagnation = 3
    assert controller._novelty_ledger_text() == ""


def test_ledger_is_bounded_by_novelty_memory():
    controller = make_controller(novelty_memory=2)
    for index in range(10):
        controller._register_solution(f"solution-{index}", index / 10)
    controller._stagnation = 1

    text = controller._novelty_ledger_text()

    assert text.count("- score") == 2
    # The most recent scores survive, not the oldest.
    assert "score 0.900000" in text
    assert "score 0.000000" not in text


def test_repeated_solutions_are_not_counted_as_novel():
    controller = make_controller()

    assert controller._register_solution("def f():\n  return 1\n", 0.5) is True
    # Same code, different whitespace -> same solution.
    assert controller._register_solution("def f():\n      return 1", 0.5) is False
    assert controller._register_solution("def g(): return 2", 0.5) is True


def test_empty_solution_is_never_novel():
    controller = make_controller()
    assert controller._register_solution("", 0.5) is False


# ----------------------------------------------------------------------
# Stagnation-driven exploration shift
# ----------------------------------------------------------------------


def test_exploration_ratio_rises_when_stalled_and_is_restored_on_improvement():
    controller = make_controller(patience=2, base_exploration_ratio=0.2)

    controller._update_adaptation(iteration=1, productive=False)
    assert controller.database.exploration_ratio == pytest.approx(0.2)

    controller._update_adaptation(iteration=2, productive=False)
    assert controller.database.exploration_ratio == pytest.approx(0.6)

    controller._update_adaptation(iteration=3, productive=True)
    assert controller.database.exploration_ratio == pytest.approx(0.2)


def test_exploration_shift_is_skipped_when_it_would_lower_the_ratio():
    controller = make_controller(patience=1, base_exploration_ratio=0.8)
    controller._update_adaptation(iteration=1, productive=False)
    assert controller.database.exploration_ratio == pytest.approx(0.8)


def test_exploration_shift_is_skipped_when_unset():
    controller = make_controller(patience=1, stalled_exploration_ratio=None)
    controller._update_adaptation(iteration=1, productive=False)
    assert controller.database.exploration_ratio == pytest.approx(0.2)


def test_long_horizon_exploration_recurs_and_resets_after_progress():
    controller = make_controller()
    controller.long_horizon_scheduler = True
    controller.long_horizon_exploration_interval = 3
    controller._next_long_horizon_exploration = 3

    controller._stagnation = 2
    assert controller._long_horizon_exploration_due() is False
    controller._stagnation = 3
    assert controller._long_horizon_exploration_due() is True
    controller._consume_long_horizon_exploration()
    assert controller._next_long_horizon_exploration == 6

    controller._update_adaptation(iteration=4, productive=True)
    assert controller._next_long_horizon_exploration == 3


def test_diversity_lane_budget_forecast_does_not_charge_for_a_guide_call():
    controller = make_controller(implementations=3)
    controller._impl_costs = [0.002]
    controller._strategy_costs = [0.02]

    assert controller._forecast_cost(1, unguided_only=True) == pytest.approx(0.002)


def test_productive_operator_is_replayed_once_before_the_next_portfolio():
    controller = make_controller()
    controller.long_horizon_scheduler = True
    controller.strategy_replays_per_portfolio = 1
    controller.strategy_replay_cooldown = 0
    controller.strategy_replay_ucb_exploration = 0.0
    controller._strategy_call_count = 2
    controller.curve.observe(0.9)
    productive = Strategy(
        index=1,
        title="productive operator",
        plan="reuse this mechanism",
        baseline_score=0.5,
        best_score=0.8,
        rounds=2,
        attempts=4,
        implementations=4,
        cumulative_relative_gain=0.4,
        productive_rounds=2,
        last_used_iteration=3,
    )
    spent = Strategy(
        index=2,
        title="spent operator",
        plan="do not immediately retry this",
        rounds=1,
        attempts=2,
        last_used_iteration=9,
    )
    controller.ledger = StrategyLedger(entries=[productive, spent])
    controller._strategy = spent

    replay = controller._strategy_replay_for_long_horizon(iteration=10)

    assert replay is not None
    assert replay.plan == productive.plan
    assert replay.source_strategy_index == productive.index
    assert replay.baseline_score == pytest.approx(0.9)
    assert controller._strategy_replay_for_long_horizon(iteration=11) is None


def test_failed_replays_reduce_the_root_operators_gain_rate():
    controller = make_controller()
    root = Strategy(
        index=1,
        title="root",
        plan="plan",
        rounds=1,
        attempts=2,
        cumulative_relative_gain=0.2,
    )
    failed_replay = Strategy(
        index=2,
        title="replay root",
        plan="plan",
        rounds=2,
        attempts=8,
        source_strategy_index=1,
    )

    entries = [root, failed_replay]
    gain = sum(controller._entry_relative_gain(entry) for entry in entries)
    attempts = sum(entry.attempts for entry in entries)

    assert gain / attempts == pytest.approx(0.02)


# ----------------------------------------------------------------------
# Config wiring
# ----------------------------------------------------------------------


def test_budget_config_reaches_the_curve():
    config = Config.from_dict(
        {
            "search": {
                "type": "efficientevolve",
                "database": {"budget_usd": 0.05, "budget_unit": "tokens"},
            }
        }
    )
    db = config.search.database

    assert db.budget_usd == pytest.approx(0.05)
    assert db.budget_unit == "tokens"


def test_legacy_width_alias_is_preserved_for_controller_compatibility():
    config = Config.from_dict(
        {
            "search": {
                "type": "efficientevolve",
                "database": {"candidates_per_iteration": 4},
            }
        }
    )
    assert config.search.database.candidates_per_iteration == 4


# ----------------------------------------------------------------------
# Migration bookkeeping
# ----------------------------------------------------------------------


def test_exploitation_samples_only_from_the_top_archive_fraction():
    config = Config.from_dict(
        {
            "search": {
                "type": "efficientevolve",
                "database": {
                    "num_islands": 1,
                    "elite_selection_ratio": 0.5,
                    "random_seed": 42,
                },
            }
        }
    ).search.database
    database = EfficientEvolveDatabase("efficientevolve", config)
    for index, score in enumerate((0.1, 0.2, 0.8, 0.9), start=1):
        program = Program(
            id=f"p{index}",
            solution=f"def solve():\n    return {index}\n",
            metrics={"combined_score": score},
            metadata={"island": 0},
        )
        database.programs[program.id] = program
        database.archive.add(program.id)
        database.islands[0].add(program.id)

    sampled = {database._sample_exploitation_parent().id for _ in range(30)}

    assert sampled == {"p3", "p4"}


def test_long_horizon_lane_uses_a_least_used_non_incumbent_parent():
    config = Config.from_dict(
        {"search": {"type": "efficientevolve", "database": {"num_islands": 1}}}
    ).search.database
    database = EfficientEvolveDatabase("efficientevolve", config)
    for index, score, uses in ((1, 0.9, 0), (2, 0.7, 5), (3, 0.6, 1)):
        program = Program(
            id=f"p{index}",
            solution=f"def solve():\n    return {index}\n",
            metrics={"combined_score": score},
            metadata={"island": 0, "_efficientevolve_parent_uses": uses},
        )
        database.programs[program.id] = program
        database.islands[0].add(program.id)
    database.best_program_id = "p1"

    parent, _context = database.sample(parent_mode="diversity")

    assert parent.id == "p3"
    assert parent.metadata["_efficientevolve_parent_uses"] == 2


def test_migrated_clones_keep_the_iteration_they_were_found_at():
    """Clones are not discoveries: iteration_found must survive migration."""
    config = Config.from_dict(
        {"search": {"type": "efficientevolve", "database": {"num_islands": 2}}}
    ).search.database
    database = EfficientEvolveDatabase("efficientevolve", config)

    database.add(
        Program(
            id="origin",
            solution="def solve():\n    return 1\n",
            metrics={"combined_score": 0.9},
        ),
        iteration=7,
        target_island=0,
    )
    database._migrate_programs()

    clones = [p for p in database.programs.values() if p.metadata.get("migrant")]
    assert clones, "expected the island program to be migrated"
    assert all(clone.iteration_found == 7 for clone in clones)


# ----------------------------------------------------------------------
# End-to-end: the run writes its curve
# ----------------------------------------------------------------------


def test_run_writes_the_budget_curve_and_seeds_it_from_the_initial_program(tmp_path):
    """The curve must start at (cost 0, seed score) and land in the output dir."""
    from unittest.mock import patch

    from skydiscover.api import run_discovery
    from skydiscover.llm.base import LLMResponse

    evaluator_file = tmp_path / "evaluator.py"
    evaluator_file.write_text(
        "def evaluate(program_path):\n"
        "    with open(program_path) as handle:\n"
        "        body = handle.read()\n"
        "    score = 0.9 if 'return 1' in body else 0.4\n"
        "    return {'combined_score': score, 'validity': 1.0}\n"
    )
    seed_file = tmp_path / "seed.py"
    seed_file.write_text("def solve():\n    return 0\n")

    class FakeLLMPool:
        def __init__(self, models_cfg):
            pass

        async def generate(self, system_message, messages, **kwargs):
            return LLMResponse(
                text="<CANDIDATE_1>\n```python\ndef solve():\n    return 1\n```\n</CANDIDATE_1>"
            )

    config = Config.from_dict(
        {
            "max_iterations": 2,
            "diff_based_generation": False,
            "monitor": {"enabled": False},
            "search": {"type": "efficientevolve", "database": {"budget_usd": 1.0}},
            "evaluator": {"evaluation_file": str(evaluator_file)},
            "llm": {
                "models": [
                    {"name": "fake-model", "api_key": "fake", "api_base": "http://localhost:1"}
                ]
            },
        }
    )

    output_dir = tmp_path / "output"
    with patch("skydiscover.search.default_discovery_controller.LLMPool", FakeLLMPool):
        run_discovery(
            evaluator=str(evaluator_file),
            initial_program=str(seed_file),
            config=config,
            output_dir=str(output_dir),
            cleanup=False,
        )

    import json

    summary = json.loads((output_dir / "ba_auc.json").read_text())
    assert summary["budget"] == pytest.approx(1.0)
    assert summary["final_incumbent"] == pytest.approx(0.9)
    # Seed point at cost 0 plus one point per iteration.
    assert summary["num_points"] == 3
    assert summary["points"][0]["cost"] == pytest.approx(0.0)
    assert summary["points"][0]["score"] == pytest.approx(0.4)
    # The fake LLM is unpriced, so nothing was spent: the incumbent is held
    # across the whole window from the first improvement onward.
    assert summary["ba_auc_normalized"] == pytest.approx(0.9)

    curve_lines = (output_dir / "budget_curve.jsonl").read_text().strip().splitlines()
    assert len(curve_lines) == 3
    assert json.loads(curve_lines[-1])["incumbent"] == pytest.approx(0.9)


def test_checkpoint_restores_strategy_and_continuous_budget_curve(tmp_path):
    first_spend = FakeSpend(0.25)
    first = make_controller(budget=100.0, spend=first_spend)
    first.strategy_escalation_patience = 8
    first.strategy_escalation_interval = 8
    first._next_strategy_escalation = 16
    first._strategy_escalation_gap = 16
    first._guide_impl_costs = [0.04]
    first._guide_implementation_due = True
    first._guide_implementation_due_reason = "test promotion"
    first._anchor_next_cheap_to_incumbent = True
    first.long_horizon_scheduler = True
    first.long_horizon_exploration_interval = 12
    first._next_long_horizon_exploration = 24
    first._strategy_replays_by_guide_call = {2: 1}
    active = Strategy(
        index=1,
        title="productive",
        plan="continue this plan",
        baseline_score=0.4,
        best_score=0.9,
        rounds=2,
        attempts=6,
        implementations=5,
        cumulative_gain=0.5,
        cumulative_relative_gain=0.4,
        productive_rounds=2,
        implementation_spend=0.03,
        last_used_iteration=7,
    )
    queued = Strategy(index=2, title="next", plan="different plan")
    first.ledger = StrategyLedger(entries=[active, queued])
    first._strategy = active
    first._strategy_queue = [queued]
    first._barren_rounds = 1
    first._stagnation = 7
    first._seen_solutions = {"digest-a"}
    first._score_ledger[0.9] = 3
    first.curve.observe(0.4, iteration=0)
    first.curve.observe(0.9, iteration=7, candidates=3)
    first.save_checkpoint_state(str(tmp_path))

    resumed_spend = FakeSpend()
    resumed = make_controller(budget=100.0, spend=resumed_spend)
    resumed.strategy_escalation_patience = 8
    resumed.strategy_escalation_interval = 8
    resumed._next_strategy_escalation = 8
    resumed._strategy_escalation_gap = 8
    resumed._strategy_queue = []
    resumed._load_checkpoint_state(str(tmp_path))

    assert resumed.curve.spent() == pytest.approx(0.25)
    assert len(resumed.curve.points) == 2
    assert resumed.curve.incumbent == pytest.approx(0.9)
    assert resumed._strategy is resumed.ledger.entries[0]
    assert resumed._strategy.plan == "continue this plan"
    assert resumed._strategy_queue == [resumed.ledger.entries[1]]
    assert resumed._barren_rounds == 1
    assert resumed._stagnation == 7
    assert resumed._next_strategy_escalation == 16
    assert resumed._strategy_escalation_gap == 16
    assert resumed._guide_impl_costs == [pytest.approx(0.04)]
    assert resumed._guide_implementation_due is True
    assert resumed._guide_implementation_due_reason == "test promotion"
    assert resumed._anchor_next_cheap_to_incumbent is True
    assert resumed._strategy_replays_by_guide_call == {2: 1}
    assert resumed._next_long_horizon_exploration == 24
    assert resumed._strategy.cumulative_gain == pytest.approx(0.5)
    assert resumed._strategy.cumulative_relative_gain == pytest.approx(0.4)
    assert resumed._strategy.last_used_iteration == 7
    assert resumed._seen_solutions == {"digest-a"}
    assert resumed._score_ledger == OrderedDict([(0.9, 3)])

    resumed_spend.value = 0.05
    resumed.curve.observe(0.95, iteration=8, candidates=2)
    assert resumed.curve.points[-1].cost == pytest.approx(0.30)


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Zero-cost recovery of unusable diffs
# ----------------------------------------------------------------------


def _recovering_controller(enabled=True):
    controller = make_controller()
    controller.config = SimpleNamespace(language="python")
    controller.language = "python"
    controller.recover_unusable_diffs = enabled
    return controller


PARENT = "def solve():\n    # the original\n    return 0\n" * 3


def test_a_full_rewrite_is_salvaged_from_a_failed_diff():
    """The call is already paid for; re-reading its response costs nothing."""
    controller = _recovering_controller()
    response = "Here is the file:\n```python\n" + PARENT.replace("0", "1") + "```\n"

    recovered = controller._recover_full_rewrite(response, PARENT)

    assert recovered is not None
    assert "return 1" in recovered


def test_a_response_with_no_program_is_not_salvaged():
    controller = _recovering_controller()
    assert controller._recover_full_rewrite("no code here at all", PARENT) is None


def test_a_fragment_is_not_mistaken_for_a_whole_program():
    controller = _recovering_controller()
    response = "```python\nreturn 1\n```"
    assert controller._recover_full_rewrite(response, PARENT) is None


def test_an_unchanged_program_is_not_salvaged():
    controller = _recovering_controller()
    response = "```python\n" + PARENT + "```"
    assert controller._recover_full_rewrite(response, PARENT) is None


def test_salvage_is_off_unless_asked_for():
    """Measured not to pay, so a failed diff stays failed by default."""
    controller = _recovering_controller(enabled=False)
    response = "```python\n" + PARENT.replace("0", "1") + "```"
    assert controller._recover_full_rewrite(response, PARENT) is None


# ----------------------------------------------------------------------
# Opening move
# ----------------------------------------------------------------------


def test_the_opening_call_asks_for_a_complete_solution():
    """dv_1 at minimal c_1 is almost the whole metric; an increment wastes it."""
    controller = make_controller()
    controller.ambitious_first_call = True
    controller.curve.observe(0.36)  # only the free seed point so far

    text = controller._opening_move_instruction()

    assert "opening attempt" in text
    assert "Do not make an incremental edit" in text


def test_the_instruction_stops_once_the_seed_is_beaten():
    controller = make_controller()
    controller.ambitious_first_call = True
    controller.curve.observe(0.36)
    controller.curve.observe(0.9, iteration=1)

    assert controller._opening_move_instruction() == ""


def test_the_instruction_survives_a_wasted_opening_call():
    """A call that produced nothing leaves the next call facing the same gap."""
    controller = make_controller()
    controller.ambitious_first_call = True
    controller.curve.observe(0.36)
    controller.curve.observe(None, iteration=1)  # wasted call

    assert "opening attempt" in controller._opening_move_instruction()


def test_the_opening_instruction_can_be_disabled():
    controller = make_controller()
    controller.ambitious_first_call = False
    controller.curve.observe(0.36)

    assert controller._opening_move_instruction() == ""


# ----------------------------------------------------------------------
# Two-tier generation: one expensive plan, many cheap implementations
# ----------------------------------------------------------------------


def test_width_comes_from_implementations_per_strategy():
    controller = make_controller(implementations=4)
    assert controller._next_candidate_count() == 4


def test_forecast_covers_implementations_and_a_strategy_only_when_due():
    controller = make_controller(implementations=3)
    controller._impl_costs = [0.002]
    controller._strategy_costs = [0.010]

    # No active strategy -> the next round must buy one.
    assert controller._strategy is None
    assert controller._forecast_cost(3) == pytest.approx(0.002 * 3 + 0.010)

    # A productive active strategy is reused without another expensive call.
    controller._strategy = Strategy(index=1, title="t", plan="p")
    controller._barren_rounds = 0
    assert controller._forecast_cost(3) == pytest.approx(0.006)

    # The first non-improving round retires it, so the next forecast includes
    # a fresh strategy.
    controller._barren_rounds = 1
    assert controller._forecast_cost(3) == pytest.approx(0.016)


def test_no_forecast_before_any_implementation_has_been_priced():
    assert make_controller()._forecast_cost(3) is None


def test_forecast_prices_a_scheduled_guide_implementation_separately():
    controller = make_controller(implementations=3)
    controller._impl_costs = [0.002]
    controller._strategy_costs = [0.010]
    controller._guide_impl_costs = [0.030]
    controller.guide_implementation_on_escalation = True
    controller.strategy_escalation_reasoning_effort = "high"
    controller.strategy_escalation_patience = 8
    controller._next_strategy_escalation = 8
    controller._stagnation = 8

    # One of the three implementation slots is replaced by the guide model.
    assert controller._forecast_cost(3) == pytest.approx(0.010 + 0.030 + 2 * 0.002)


def test_unknown_first_guide_implementation_cost_does_not_pretend_to_be_free():
    controller = make_controller(implementations=3)
    controller._impl_costs = [0.002]
    controller._strategy_costs = [0.010]
    controller.guide_implementation_on_escalation = True
    controller.strategy_escalation_reasoning_effort = "high"
    controller.strategy_escalation_patience = 8
    controller._next_strategy_escalation = 8
    controller._stagnation = 8

    assert controller._forecast_cost(3) is None


def test_a_strategy_records_what_its_implementations_achieved():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1, implementations=2)
    controller.curve.observe(0.4)
    controller._strategy = Strategy(index=1, title="t", plan="p", baseline_score=0.4)

    spend.value = 0.01
    controller._record_iteration(
        iteration=1,
        candidates=2,
        successful=[(1, outcome(0.55)), (2, outcome(0.91, solution="other"))],
        cost_before=0.0,
    )

    assert controller._strategy.best_score == pytest.approx(0.91)
    assert controller._strategy.implementations == 2
    assert controller._strategy.attempts == 2
    assert controller._strategy.improved is True
    assert controller._strategy.cumulative_gain == pytest.approx(0.51)
    assert controller._strategy.cumulative_relative_gain == pytest.approx(0.51 / 0.91)
    assert controller._strategy.productive_rounds == 1
    assert controller._strategy.barren_rounds == 0
    assert controller._strategy.last_used_iteration == 1
    assert controller._barren_rounds == 0


def test_productive_strategy_gets_one_guide_implementation_before_retirement():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=1, implementations=1)
    controller.curve.observe(0.5)
    strategy = Strategy(index=1, title="productive", plan="plan", baseline_score=0.5)
    controller._strategy = strategy

    spend.value = 0.01
    controller._record_iteration(
        iteration=1,
        candidates=1,
        successful=[(1, outcome(0.7))],
        cost_before=0.0,
    )
    assert controller._guide_implementation_due is False

    spend.value = 0.02
    controller._record_iteration(
        iteration=2,
        candidates=1,
        successful=[(1, outcome(0.7, solution="same score"))],
        cost_before=0.01,
    )

    assert controller._guide_implementation_due is True
    assert "productive strategy" in controller._guide_implementation_due_reason
    assert controller._wants_new_strategy() is False

    # Once that single promotion has been consumed, a further barren round
    # retires the strategy normally instead of repeatedly buying Terra code.
    controller._guide_implementation_due = False
    strategy.guide_implementation_attempts = 1
    assert controller._wants_new_strategy() is True


def test_a_barren_strategy_is_counted_toward_retirement():
    spend = FakeSpend()
    controller = make_controller(spend=spend, patience=2, implementations=1)
    controller.curve.observe(0.9)
    controller._strategy = Strategy(index=1, title="t", plan="p", baseline_score=0.9)

    spend.value = 0.01
    controller._record_iteration(
        iteration=1, candidates=1, successful=[(1, outcome(0.9))], cost_before=0.0
    )

    assert controller._strategy.improved is False
    assert controller._barren_rounds == 1
    assert controller._guide_implementation_due is False
