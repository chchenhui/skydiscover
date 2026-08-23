#!/usr/bin/env python3
"""Score run directories with Budget-Aware AUC and print a budget sweep.

BA-AUC is defined for a fixed budget ``B``. This utility can report the
normalized metric at one or more chosen budget values.

Runs produced by the current EfficientEvolve controller carry their own
``budget_curve.jsonl`` and are read directly.  Older runs -- including the
OpenEvolve external backend -- are reconstructed from their logs and stored
programs, which is how the baseline table in
``skydiscover/search/efficientevolve/README.md`` was produced.

Usage:
    python scripts/score_ba_auc.py openevolve_output/*/
    python scripts/score_ba_auc.py out/run_a out/run_b --budgets 0.5 10 100
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skydiscover.llm.pricing import Usage, resolve_pricing  # noqa: E402
from skydiscover.search.utils.budget_curve import (  # noqa: E402
    auc_table,
    load_curve_samples,
    mean_incumbent_curve,
    plot_budget_curves,
    welch_mean_difference_ci,
)
from skydiscover.utils.metrics import get_score  # noqa: E402

Sample = Tuple[float, float]

# "... total_tokens=6346, total_cost=$0.005632" in a SkyDiscover run log.
_CALL_COST = re.compile(r"total_tokens=\d+, total_cost=\$([0-9.]+)")
_ITER_PROGRAM = re.compile(r"Iteration (\d+): Program ([0-9a-f-]{8,36})")


def _resolve_program(programs: Dict[str, dict], logged_id: str) -> Optional[dict]:
    """Resolve either a full UUID or a controller's unambiguous short id."""
    exact = programs.get(logged_id)
    if exact is not None:
        return exact
    matches = [record for program_id, record in programs.items() if program_id.startswith(logged_id)]
    return matches[0] if len(matches) == 1 else None


def _stored_programs(run_dir: str) -> Dict[str, dict]:
    """Every program record saved under this run's checkpoints, newest wins."""
    programs: Dict[str, dict] = {}
    pattern = os.path.join(run_dir, "checkpoints", "checkpoint_*", "programs", "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if "id" in record:
            programs[record["id"]] = record
    return programs


def _seed_sample(programs: Dict[str, dict]) -> List[Sample]:
    """The initial program is free, so it anchors the curve at cost 0."""
    for record in programs.values():
        if not record.get("parent_id"):
            return [(0.0, get_score(record.get("metrics") or {}))]
    return []


def _log_path(run_dir: str) -> Optional[str]:
    logs = sorted(glob.glob(os.path.join(run_dir, "logs", "*.log")))
    return logs[-1] if logs else None


def _usage_event_costs(run_dir: str) -> List[float]:
    """Per-call cost from ``llm_usage_events.jsonl`` (OpenEvolve backend)."""
    path = os.path.join(run_dir, "llm_usage_events.jsonl")
    if not os.path.exists(path):
        return []

    costs: List[float] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            pricing = resolve_pricing(event.get("model", ""), event.get("provider"))
            usage = Usage(**{k: v for k, v in (event.get("usage") or {}).items()})
            cost = pricing.cost(usage) if pricing else None
            costs.append(cost or 0.0)
    return costs


def _chronological_log_samples(text: str, programs: Dict[str, dict]) -> List[Sample]:
    """Replay native SkyDiscover cost and result events in log order.

    Native controllers log each priced LLM response before they log the
    solution whose evaluation followed that response.  Replaying those two
    event types preserves the cost of retries and algorithm overhead such as
    EvoX label/meta-search calls or AdaEvolve paradigm generation.  Those
    calls deliberately create no score sample: the incumbent stays flat until
    a later solution is actually evaluated.

    The external OpenEvolve backend is handled separately because its
    per-call usage is written to ``llm_usage_events.jsonl`` rather than mixed
    into its progress log.
    """
    samples = _seed_sample(programs)
    cumulative = 0.0
    saw_cost = False

    for line in text.splitlines():
        cost_match = _CALL_COST.search(line)
        if cost_match:
            cumulative += float(cost_match.group(1))
            saw_cost = True

        program_match = _ITER_PROGRAM.search(line)
        if program_match:
            program = _resolve_program(programs, program_match.group(2))
            if program is not None:
                samples.append((cumulative, get_score(program.get("metrics") or {})))

    if not saw_cost:
        return []

    # Preserve trailing failed/meta calls in the reported spend.  A score of
    # zero cannot lower the best-so-far curve, but marks its true right edge.
    if not samples or cumulative > samples[-1][0]:
        samples.append((cumulative, 0.0))
    return samples


def reconstruct(run_dir: str) -> List[Sample]:
    """Rebuild ``(cumulative_cost, score)`` samples for one run directory."""
    curve_file = os.path.join(run_dir, "budget_curve.jsonl")
    if os.path.exists(curve_file):
        return load_curve_samples(curve_file)

    log = _log_path(run_dir)
    if log is None:
        raise ValueError(f"{run_dir}: no budget_curve.jsonl and no logs/*.log to reconstruct from")

    programs = _stored_programs(run_dir)
    with open(log, errors="replace") as handle:
        text = handle.read()

    # Program ids in the order the log reports them, which is completion order.
    completed = []
    for _, logged_id in _ITER_PROGRAM.findall(text):
        program = _resolve_program(programs, logged_id)
        if program is not None:
            completed.append(program["id"])
    # Candidates from one batch share an iteration and therefore a cost step.
    per_iteration: Dict[int, List[float]] = {}
    for raw_iteration, logged_id in _ITER_PROGRAM.findall(text):
        program = _resolve_program(programs, logged_id)
        if program is not None:
            score = get_score(program.get("metrics") or {})
            per_iteration.setdefault(int(raw_iteration), []).append(score)

    # Native SkyDiscover logs carry per-call cost and candidate events in one
    # chronological stream.  This is more exact than assigning call n to
    # candidate n, especially for EvoX and AdaEvolve whose controller-level
    # LLM calls do not directly produce solution candidates.
    usage_event_costs = _usage_event_costs(run_dir)
    if not usage_event_costs:
        chronological = _chronological_log_samples(text, programs)
        if chronological:
            return chronological

    # The external OpenEvolve backend persists structured per-call usage but
    # logs only an end-of-run aggregate.  Its call completion order matches
    # its candidate completion order, including concurrent workers.
    costs = usage_event_costs or [float(value) for value in _CALL_COST.findall(text)]
    if not costs:
        raise ValueError(f"{run_dir}: no per-call cost found in the log or usage events")

    samples = _seed_sample(programs)
    # One LLM call per iteration (batched controllers) vs one call per program
    # (OpenEvolve backend).
    batched = len(costs) < len(completed)
    cumulative = 0.0
    for index, cost in enumerate(costs):
        cumulative += cost
        if batched:
            scores = per_iteration.get(index + 1, [])
        elif index < len(completed):
            scores = [get_score(programs[completed[index]].get("metrics") or {})]
        else:
            scores = []
        samples.append((cumulative, max(scores) if scores else 0.0))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", help="run output directories")
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0, 10.0, 100.0],
        help="budget windows B to score at (default: 0.1 0.5 1 10 100)",
    )
    parser.add_argument(
        "--plot",
        metavar="PATH",
        help="also draw the cost/performance curves to this image path",
    )
    parser.add_argument(
        "--plot-budget",
        type=float,
        help="mark B on the plot and clip the x-axis there (default: largest observed cost)",
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="light",
        help="plot theme (default: light)",
    )
    parser.add_argument(
        "--aggregate",
        action="append",
        default=[],
        metavar="NAME=GLOB",
        help=(
            "replace individual output/plot curves with pointwise means for a repeated-run "
            "group; repeat this option and quote shell globs"
        ),
    )
    args = parser.parse_args()

    runs: Dict[str, List[Sample]] = {}
    for run_dir in args.run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        try:
            runs[name] = reconstruct(run_dir)
        except (ValueError, OSError) as exc:
            print(f"skipped {name}: {exc}", file=sys.stderr)

    if not runs:
        print("no runs could be scored", file=sys.stderr)
        return 1

    display_runs = runs
    group_spend: Dict[str, float] = {}
    group_final: Dict[str, float] = {}
    group_metric_sd: Dict[str, Dict[float, float]] = {}
    group_member_tables: Dict[str, List[Dict[float, float]]] = {}
    if args.aggregate:
        display_runs = {}
        for specification in args.aggregate:
            if "=" not in specification:
                parser.error(f"--aggregate must be NAME=GLOB, got {specification!r}")
            group_name, pattern = specification.split("=", 1)
            group_name = group_name.strip()
            members = [samples for name, samples in runs.items() if fnmatch.fnmatch(name, pattern)]
            if not group_name or not members:
                parser.error(f"--aggregate {specification!r} matched no runs or has an empty name")
            display_runs[group_name] = mean_incumbent_curve(members)
            member_tables = [
                auc_table({"member": samples}, args.budgets)["member"] for samples in members
            ]
            group_member_tables[group_name] = member_tables
            group_metric_sd[group_name] = {
                budget: (
                    statistics.stdev(table[budget] for table in member_tables)
                    if len(member_tables) > 1
                    else 0.0
                )
                for budget in args.budgets
            }
            group_spend[group_name] = sum(
                max((cost for cost, _ in samples), default=0.0) for samples in members
            ) / len(members)
            group_final[group_name] = sum(
                max((score for _, score in samples), default=0.0) for samples in members
            ) / len(members)

    table = auc_table(display_runs, args.budgets)
    width = (
        max(
            max(len(name), len(f"{name} sample SD") if name in group_metric_sd else 0)
            for name in display_runs
        )
        + 2
    )
    header = "run".ljust(width) + "".join(f"B=${b:<10.4f}".rjust(14) for b in args.budgets)
    print(header)
    print("-" * len(header))
    for name, samples in display_runs.items():
        row = "".join(f"{table[name][b]:.6f}".rjust(14) for b in args.budgets)
        print(name.ljust(width) + row)
        if name in group_metric_sd:
            sd_row = "".join(
                f"{group_metric_sd[name][budget]:.6f}".rjust(14) for budget in args.budgets
            )
            print(f"{name} sample SD".ljust(width) + sd_row)
    print()
    for name, samples in display_runs.items():
        total = group_spend.get(name, max(cost for cost, _ in samples))
        best = group_final.get(name, max(score for _, score in samples))
        qualifier = "mean " if name in group_spend else ""
        print(
            f"{name.ljust(width)} {qualifier}spent ${total:.4f}, "
            f"{qualifier}final best {best:.6f}"
        )

    if len(group_member_tables) == 2:
        baseline_name, candidate_name = group_member_tables
        print(
            f"\nWelch 95% CI for {candidate_name} - {baseline_name} " "(independent repeated runs):"
        )
        for budget in args.budgets:
            candidate_values = [member[budget] for member in group_member_tables[candidate_name]]
            baseline_values = [member[budget] for member in group_member_tables[baseline_name]]
            difference, lower, upper, degrees = welch_mean_difference_ci(
                candidate_values, baseline_values
            )
            print(
                f"  B=${budget:g}: Δ={difference:+.6f}, "
                f"95% CI [{lower:+.6f}, {upper:+.6f}], df={degrees:.2f}"
            )

    if args.plot:
        plot_budget_curves(
            display_runs,
            args.plot,
            budget=args.plot_budget,
            theme=args.theme,
            # Only a single-run plot is shaded, so the subtitle must not
            # promise an area that is not drawn.
            subtitle=(
                "Best score found so far against cumulative LLM spend; shaded area is BA-AUC"
                if len(display_runs) == 1
                else "Best score found so far against cumulative LLM spend"
            ),
        )
        print(f"\nwrote plot to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
