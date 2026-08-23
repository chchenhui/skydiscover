"""Tests for Budget-Aware AUC (BA-AUC) and the live budget curve."""

import json

import pytest

from skydiscover.search.utils.budget_curve import (
    TOKENS,
    USD,
    BudgetCurve,
    _place_labels,
    auc_table,
    budget_aware_auc,
    incumbent_steps,
    load_curve_samples,
    mean_incumbent_curve,
    normalized_budget_aware_auc,
    plot_budget_curves,
    welch_mean_difference_ci,
)

# ----------------------------------------------------------------------
# The integral
# ----------------------------------------------------------------------


def test_constant_score_from_zero_fills_the_window():
    # A solution held at 0.5 from cost 0 to B=10 encloses 0.5 * 10.
    assert budget_aware_auc([(0.0, 0.5)], budget=10.0) == pytest.approx(5.0)


def test_nothing_scored_encloses_nothing():
    assert budget_aware_auc([], budget=4.0) == pytest.approx(0.0)


def test_single_step_splits_the_window():
    # 0 on [0, 2), then 1.0 on [2, 10) -> 8.0
    assert budget_aware_auc([(2.0, 1.0)], budget=10.0) == pytest.approx(8.0)


def test_two_steps_accumulate():
    # 0 on [0,1), 0.4 on [1,3), 0.9 on [3,5) -> 0.4*2 + 0.9*2 = 2.6
    samples = [(1.0, 0.4), (3.0, 0.9)]
    assert budget_aware_auc(samples, budget=5.0) == pytest.approx(2.6)


def test_curve_is_best_so_far_not_last_seen():
    # A regression after the peak must not lower the curve.
    samples = [(1.0, 0.9), (2.0, 0.1)]
    assert budget_aware_auc(samples, budget=3.0) == pytest.approx(0.9 * 2)


def test_samples_beyond_the_budget_are_truncated():
    # The 1.0 arrives at cost 5, outside B=3, so it contributes nothing.
    samples = [(1.0, 0.5), (5.0, 1.0)]
    assert budget_aware_auc(samples, budget=3.0) == pytest.approx(0.5 * 2)


def test_short_run_extends_flat_to_the_budget_edge():
    # Stopping at cost 2 keeps 0.8 all the way to B=10.
    assert budget_aware_auc([(2.0, 0.8)], budget=10.0) == pytest.approx(0.8 * 8)


def test_sample_order_does_not_matter():
    forward = budget_aware_auc([(1.0, 0.4), (3.0, 0.9)], budget=5.0)
    shuffled = budget_aware_auc([(3.0, 0.9), (1.0, 0.4)], budget=5.0)
    assert forward == pytest.approx(shuffled)


def test_simultaneous_samples_take_the_max():
    # Three siblings from one batch share a cost; only the best sets the step.
    samples = [(1.0, 0.2), (1.0, 0.7), (1.0, 0.5)]
    assert budget_aware_auc(samples, budget=2.0) == pytest.approx(0.7)


def test_negative_scores_are_clamped_to_the_floor():
    assert budget_aware_auc([(0.0, -5.0)], budget=2.0) == pytest.approx(0.0)


def test_floor_is_configurable():
    assert budget_aware_auc([], budget=2.0, floor=0.25) == pytest.approx(0.5)


def test_zero_or_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        budget_aware_auc([(1.0, 1.0)], budget=0.0)


def test_normalization_divides_by_the_window():
    samples = [(1.0, 0.4), (3.0, 0.9)]
    assert normalized_budget_aware_auc(samples, budget=5.0) == pytest.approx(2.6 / 5.0)


def test_normalized_auc_of_a_perfect_free_solution_is_one():
    assert normalized_budget_aware_auc([(0.0, 1.0)], budget=7.0) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Properties the metric is chosen for
# ----------------------------------------------------------------------


def test_dominating_curve_scores_at_least_as_high():
    worse = [(1.0, 0.5), (4.0, 0.8)]
    better = [(1.0, 0.6), (4.0, 0.9)]
    assert budget_aware_auc(better, 10.0) >= budget_aware_auc(worse, 10.0)


def test_front_loading_strictly_wins_at_equal_final_score():
    early = [(1.0, 0.9)]
    late = [(8.0, 0.9)]
    assert budget_aware_auc(early, 10.0) > budget_aware_auc(late, 10.0)


def test_spend_after_the_last_improvement_is_free():
    """The metric is invariant past the final improvement -- waste costs only opportunity."""
    stopped = [(1.0, 0.9)]
    kept_going = [(1.0, 0.9), (2.0, 0.3), (3.0, 0.5), (4.0, 0.9)]
    assert budget_aware_auc(stopped, 10.0) == pytest.approx(budget_aware_auc(kept_going, 10.0))


def test_budget_choice_can_reverse_the_ranking():
    """Cheap-and-stuck beats slow-and-strong at small B, and loses at large B."""
    cheap_then_stuck = [(1.0, 0.8)]
    slow_then_strong = [(9.0, 1.0)]
    assert normalized_budget_aware_auc(cheap_then_stuck, 5.0) > normalized_budget_aware_auc(
        slow_then_strong, 5.0
    )
    assert normalized_budget_aware_auc(cheap_then_stuck, 100.0) < normalized_budget_aware_auc(
        slow_then_strong, 100.0
    )


# ----------------------------------------------------------------------
# Live recorder
# ----------------------------------------------------------------------


class FakeSpend:
    """Stand-in for the global cost tracker."""

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_curve_records_cost_at_observation_time():
    spend = FakeSpend()
    curve = BudgetCurve(budget=10.0, cost_fn=spend)

    curve.observe(0.3, iteration=0)
    spend.value = 2.0
    curve.observe(0.9, iteration=1)

    assert [(p.cost, p.score) for p in curve.points] == [(0.0, 0.3), (2.0, 0.9)]
    # 0.3 on [0,2), 0.9 on [2,10) -> 0.6 + 7.2
    assert curve.auc() == pytest.approx(7.8)
    assert curve.normalized_auc() == pytest.approx(0.78)


def test_incumbent_never_falls():
    spend = FakeSpend()
    curve = BudgetCurve(budget=4.0, cost_fn=spend)

    curve.observe(0.8)
    spend.value = 1.0
    point = curve.observe(0.1)

    assert point.score == pytest.approx(0.1)
    assert point.incumbent == pytest.approx(0.8)
    assert curve.incumbent == pytest.approx(0.8)


def test_observing_none_records_a_step_that_bought_nothing():
    spend = FakeSpend()
    curve = BudgetCurve(budget=4.0, cost_fn=spend)

    spend.value = 1.0
    point = curve.observe(None, iteration=3)

    assert point.cost == pytest.approx(1.0)
    assert point.score == pytest.approx(0.0)
    assert curve.auc() == pytest.approx(0.0)


def test_remaining_and_exhausted_track_the_window():
    spend = FakeSpend()
    curve = BudgetCurve(budget=5.0, cost_fn=spend)

    assert curve.remaining() == pytest.approx(5.0)
    assert not curve.exhausted()

    spend.value = 5.0
    assert curve.remaining() == pytest.approx(0.0)
    assert curve.exhausted()


def test_unbounded_curve_has_infinite_remaining_and_no_auc():
    curve = BudgetCurve(budget=None, cost_fn=FakeSpend())
    curve.observe(0.5)

    assert curve.remaining() == float("inf")
    assert curve.auc() is None
    assert curve.normalized_auc() is None
    # A budget can still be supplied after the fact.
    assert curve.auc(2.0) == pytest.approx(1.0)


def test_restored_curve_keeps_points_and_offsets_new_spend():
    first_spend = FakeSpend()
    first = BudgetCurve(budget=10.0, cost_fn=first_spend)
    first.observe(0.4, iteration=0)
    first_spend.value = 2.0
    first.observe(0.9, iteration=1)

    resumed_spend = FakeSpend()
    resumed = BudgetCurve(budget=10.0, cost_fn=resumed_spend)
    resumed.restore(first.to_dict())

    assert resumed.spent() == pytest.approx(2.0)
    assert resumed.incumbent == pytest.approx(0.9)
    assert [(point.cost, point.incumbent) for point in resumed.points] == [
        (0.0, 0.4),
        (2.0, 0.9),
    ]

    resumed_spend.value = 1.5
    resumed.observe(0.95, iteration=2)
    assert resumed.spent() == pytest.approx(3.5)
    assert resumed.points[-1].cost == pytest.approx(3.5)


def test_invalid_budget_and_unit_are_rejected():
    with pytest.raises(ValueError):
        BudgetCurve(budget=0.0, cost_fn=FakeSpend())
    with pytest.raises(ValueError):
        BudgetCurve(unit="euros", cost_fn=FakeSpend())


def test_token_unit_is_accepted():
    curve = BudgetCurve(budget=1000.0, unit=TOKENS, cost_fn=FakeSpend())
    assert curve.unit == TOKENS


def test_summary_line_mentions_spend_incumbent_and_auc():
    spend = FakeSpend()
    curve = BudgetCurve(budget=2.0, unit=USD, cost_fn=spend)
    curve.observe(0.5)

    line = curve.summary_line()
    assert "spent=$" in line
    assert "incumbent=0.500000" in line
    assert "normalized_BA-AUC@$2.0000=0.500000" in line


# ----------------------------------------------------------------------
# Persistence and offline re-scoring
# ----------------------------------------------------------------------


def test_write_then_reload_reproduces_the_auc(tmp_path):
    spend = FakeSpend()
    curve = BudgetCurve(budget=10.0, cost_fn=spend)
    curve.observe(0.3, iteration=0)
    spend.value = 2.0
    curve.observe(0.9, iteration=1, candidates=3)

    curve.write(str(tmp_path))

    summary = json.loads((tmp_path / "ba_auc.json").read_text())
    assert summary["budget"] == pytest.approx(10.0)
    assert summary["ba_auc"] == pytest.approx(7.8)
    assert summary["ba_auc_normalized"] == pytest.approx(0.78)
    assert summary["final_incumbent"] == pytest.approx(0.9)
    assert summary["unit"] == USD

    samples = load_curve_samples(str(tmp_path / "budget_curve.jsonl"))
    assert samples == [(0.0, 0.3), (2.0, 0.9)]
    assert budget_aware_auc(samples, 10.0) == pytest.approx(7.8)
    # Offline re-scoring at a different budget needs no re-run.
    assert budget_aware_auc(samples, 4.0) == pytest.approx(0.6 + 1.8)


def test_write_to_no_output_dir_is_a_noop():
    BudgetCurve(budget=1.0, cost_fn=FakeSpend()).write(None)


def test_auc_table_sweeps_runs_and_budgets():
    table = auc_table(
        {"early": [(1.0, 0.9)], "late": [(8.0, 0.9)]},
        budgets=[5.0, 100.0],
    )

    assert table["early"][5.0] > table["late"][5.0]
    assert table["early"][100.0] > table["late"][100.0]
    assert set(table) == {"early", "late"}


def test_mean_curve_area_equals_mean_member_auc():
    first = [(0.0, 0.2), (1.0, 0.8)]
    second = [(0.0, 0.4), (2.0, 1.0)]

    mean_curve = mean_incumbent_curve([first, second])

    assert [cost for cost, _ in mean_curve] == [0.0, 1.0, 2.0]
    assert [score for _, score in mean_curve] == pytest.approx([0.3, 0.6, 0.9])
    expected = (budget_aware_auc(first, 5.0) + budget_aware_auc(second, 5.0)) / 2
    assert budget_aware_auc(mean_curve, 5.0) == pytest.approx(expected)


def test_mean_curve_rejects_empty_group():
    with pytest.raises(ValueError, match="empty"):
        mean_incumbent_curve([])


def test_welch_interval_reports_candidate_minus_baseline():
    difference, lower, upper, degrees = welch_mean_difference_ci(
        candidate=[0.8, 0.9, 1.0, 0.9],
        baseline=[0.5, 0.6, 0.7, 0.6],
    )

    assert difference == pytest.approx(0.3)
    assert lower < difference < upper
    assert lower > 0
    assert degrees > 0


def test_welch_interval_handles_identical_zero_variance_groups():
    difference, lower, upper, degrees = welch_mean_difference_ci(
        candidate=[0.9, 0.9], baseline=[0.9, 0.9]
    )

    assert (difference, lower, upper) == pytest.approx((0.0, 0.0, 0.0))
    assert degrees == float("inf")


def test_welch_interval_validates_sample_count_and_confidence():
    with pytest.raises(ValueError, match="at least two"):
        welch_mean_difference_ci(candidate=[0.9], baseline=[0.8, 0.9])
    with pytest.raises(ValueError, match="confidence"):
        welch_mean_difference_ci(candidate=[0.8, 0.9], baseline=[0.7, 0.8], confidence=1.0)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def test_incumbent_steps_keeps_only_the_rises():
    samples = [(1.0, 0.4), (2.0, 0.1), (3.0, 0.9), (4.0, 0.9)]
    assert incumbent_steps(samples) == [(1.0, 0.4), (3.0, 0.9)]


def test_incumbent_steps_starts_at_the_first_sample():
    assert incumbent_steps([(2.0, 0.5)]) == [(2.0, 0.5)]


def test_incumbent_steps_of_nothing_is_empty():
    assert incumbent_steps([]) == []


def test_incumbent_steps_clamps_to_the_floor():
    assert incumbent_steps([(1.0, -3.0), (2.0, 0.5)]) == [(1.0, 0.0), (2.0, 0.5)]


def test_colliding_labels_are_pushed_apart():
    placed = _place_labels([(0.9, "a", "#000"), (0.9, "b", "#111")], 0.05, (0.0, 1.0))
    ys = [y_label for _, y_label, _, _ in placed]
    assert ys[1] - ys[0] == pytest.approx(0.05)


def test_pushed_labels_stay_inside_the_axes():
    """A label shoved off the top disappears — worse than the collision."""
    entries = [(0.96, "a", "#000"), (0.96, "b", "#111"), (0.96, "c", "#222")]
    placed = _place_labels(entries, 0.05, (0.02, 0.98))

    ys = [y_label for _, y_label, _, _ in placed]
    assert max(ys) <= 0.98 + 1e-9
    assert min(ys) >= 0.02 - 1e-9
    assert ys == sorted(ys)


def test_uncrowded_labels_are_left_on_their_lines():
    placed = _place_labels([(0.2, "a", "#000"), (0.8, "b", "#111")], 0.05, (0.0, 1.0))
    assert [(y_true, y_label) for y_true, y_label, _, _ in placed] == [(0.2, 0.2), (0.8, 0.8)]


def test_plot_writes_an_image(tmp_path):
    path = tmp_path / "curves.png"
    plot_budget_curves(
        {"a": [(0.0, 0.3), (0.01, 0.9)], "b": [(0.0, 0.3), (0.02, 0.95)]},
        str(path),
    )
    assert path.exists() and path.stat().st_size > 0


def test_plot_accepts_a_budget_and_the_dark_theme(tmp_path):
    path = tmp_path / "dark.png"
    plot_budget_curves(
        {"a": [(0.0, 0.3), (0.01, 0.9)]},
        str(path),
        budget=0.05,
        theme="dark",
    )
    assert path.exists() and path.stat().st_size > 0


def test_plot_rejects_empty_runs_and_unknown_themes(tmp_path):
    with pytest.raises(ValueError):
        plot_budget_curves({}, str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        plot_budget_curves({"a": [(1.0, 0.5)]}, str(tmp_path / "x.png"), theme="sepia")


def test_plot_needs_a_nonzero_x_range(tmp_path):
    # Everything at cost 0 and no budget: there is no window to draw.
    with pytest.raises(ValueError):
        plot_budget_curves({"a": [(0.0, 0.5)]}, str(tmp_path / "x.png"))
