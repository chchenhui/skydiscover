# EfficientEvolve

EfficientEvolve uses an expensive model to choose a search strategy and a cheap model to turn that
strategy into several independently evaluated programs. It optimizes solution quality per dollar,
not only the final score.

## Budget-Aware AUC

After iteration `i` finishes, record:

- `C_i`: cumulative cost of every LLM call through that iteration;
- `P_i`: the global best `combined_score` after evaluating that iteration.

These points define the best-so-far performance–cost step curve `p(c)`. For a fixed `B > 0`:

```text
BA-AUC(B)            = ∫₀ᴮ p(c) dc
normalized BA-AUC(B) = BA-AUC(B) / B
```

The normalized value is the average best-so-far score over cost interval `[0, B]`. Earlier gains
count more because they remain active over more of the interval.

Repository conventions:

- The initial program is at cost `0`.
- The algorithm prints cumulative cost and best score once per iteration.
- The curve is non-decreasing: a worse candidate never lowers it.
- If the run spends less than `B`, its final best score is extended flat to `B`.
- Work and improvements after `B` do not affect `BA-AUC(B)`.
- Cost is USD when model pricing is known; otherwise it falls back to tokens.

For `B = 100`, a run does **not** need to spend $100. A run ending at $0.30 keeps its final score
from $0.30 to $100. Consequently, `normalized BA-AUC(100)` is close to final performance; report
actual spend and a smaller budget such as `$0.5` or `$1` when early efficiency also matters.

Each run writes:

```text
budget_curve.jsonl   iteration-level (cost, best score) points
ba_auc.json          raw and normalized BA-AUC at configured B
```

Score existing runs or draw the curve with:

```bash
python scripts/score_ba_auc.py RUN_DIR... --budgets 0.5 1 100
python scripts/score_ba_auc.py RUN_DIR... --budgets 100 --plot comparison.png --plot-budget 100
```

## Algorithm

The default loop is:

```text
1. Run and record one Luna direct probe. This gives a cheap candidate its full early BA-AUC value
   without hiding it behind a concurrent batch.
2. On the next iteration, run and record one low-effort Terra direct implementation against the
   current incumbent. This bounded hedge makes the cheap opening robust to parse and quality
   failures while avoiding a prose/code multitask prompt.
3. If both direct paths stall, Terra proposes a portfolio of strategies and includes one executable
   reference program in the same response. Evaluate that reference before buying Luna candidates.
4. On the next iteration after an improvement, keep the same strategy and ask Luna for multiple
   independent program candidates.
5. If that batch improves the incumbent, keep the strategy again. If it does not, retire it and
   activate the next queued strategy or ask Terra for a new portfolio.
6. Insert every usable program into the island/MAP-Elites database and update the global incumbent.
```

Thus an improving strategy is unchanged across iterations. The model generates multiple **program
candidates**, not multiple replacement strategies, during those reuse iterations.

The two opening stages are sequential on purpose: BA-AUC can only credit a candidate after the
algorithm observes it. A concurrent three-candidate Luna batch would charge all three calls before
moving the curve. If the Terra hedge improves, the unused strategy call is skipped and the next
round cheaply exploits the stronger incumbent. If it fails, its cost is still recorded separately
and the prose portfolio receives normal budget admission next round. Opening Terra code uses low
reasoning effort, comparable to a low-effort direct baseline; medium effort remains available for
rarer long-plateau code insurance. When a positive `target_score` is configured and the cheap probe
already reaches `opening_guide_defer_target_ratio` (0.9 by default), the guide hedge is deferred
while cheap pilots keep improving and fires on their first stall. Without an explicit target, the
robust hedge remains unconditional.

After the opening, plateau guide calls can return three distinct strategies in one response and
queue them. Strategies are ranked by predicted score subject to validity and implementability. For
continuous min/max or ratio objectives, the guide explicitly considers auxiliary-variable
constraint reformulations and deterministic multistart optimization instead of proposing only
random perturbations.

Luna candidates use anytime pilot-first racing: evaluate a small pilot group, then launch the
remaining candidates only if a pilot is usable and competitive. If a pilot already improves the
incumbent, the batch ends immediately and the next iteration continues from the stronger parent.
This moves the gain left on the BA-AUC curve and avoids paying for candidates generated from a
parent that is already stale. Once a productive strategy has demonstrated a high usable-program
rate, its pilot group automatically shrinks to one. Direct Luna climbing uses an even simpler
lossless race: run one pilot first and stop only if it improves; on failure or no gain, launch all
remaining candidates, so plateau exploration width is never reduced.

For long runs, guide strategies are treated as learned search operators. Each activation receives
scale-free credit for actual global-incumbent gain, plus implementation and failure statistics.
After a paid portfolio is exhausted, a bounded UCB scheduler may replay a different historically
productive operator before buying the next portfolio:

```text
operator value = cumulative relative incumbent gain / cheap-call-equivalent attempts
                 + beta * sqrt(log(1 + all attempts) / operator attempts)
```

Replay failures remain attached to the original operator, so its empirical value falls instead of
creating a fresh-looking duplicate. The default permits only one replay per paid portfolio; after
that, a fresh guide call is forced, and a strategy must have improved the incumbent in at least two
separate rounds before it is eligible. When prices are known, guide-model implementation spend is
converted to the equivalent number of cheap calls before computing the exploitation term. This
gives exploitation a cost advantage without allowing an
early strategy to monopolize an unbounded run. The guide prompt also retains a small hall of fame
from the full run alongside recent failures, so early breakthroughs are not forgotten when the
history window fills.

Long runs also retain three escape paths:

- exponentially backed-off high-reasoning strategy calls and Terra implementations on sustained
  plateaus;
- an occasional unguided Terra implementation before buying another strategy portfolio.
- every 12 stagnant rounds, one cheap unguided candidate from a least-used non-incumbent
  MAP-Elites parent.

If that unguided candidate improves, the new portfolio is skipped and the next iteration continues
from the stronger incumbent with Luna. These paths prevent a difficult implementation from being
permanently limited by the cheap model, and prevent incumbent anchoring from disabling quality-
diversity exploration, while keeping most candidate generation cheap.

If a benchmark defines a sufficient normalized score, set `target_score` (for example `1.0`).
EfficientEvolve then stops before the next iteration as soon as the incumbent reaches that value,
within `target_score_tolerance`. This lowers target cost and wall time without changing the
best-so-far BA-AUC curve. It is deliberately unset by default because many objectives have no
known ceiling and scores above a reference value can still be meaningful. A second target-aware
rule promotes one implementation of a productive-but-stalled strategy to Terra after the incumbent
reaches `target_aware_guide_promotion_ratio` (0.98 by default). This concentrates strong-model
coding on the last mile without adding linear promotion cost to unbounded, target-free searches.

`strategy_when: stalled` remains available as an ablation that runs Luna before buying the first
strategy. `strategy_when: always` buys a new strategy every iteration, and `two_tier: false` skips
the guide tier.

## Minimal configuration

```yaml
max_iterations: 10

llm:
  models:
    - name: gpt-5.6-luna
  guide_models:
    - name: gpt-5.6-terra

search:
  type: efficientevolve
  database:
    budget_usd: 100.0
    target_score: 1.0  # only when 1.0 is a known sufficient task target
    opening_cascade: true
    opening_cheap_candidates: 1
    opening_guide_reasoning_effort: low
    opening_guide_defer_target_ratio: 0.9
    target_aware_guide_promotion_ratio: 0.98
    strategy_when: reuse_on_improvement
    strategy_reference_candidate: true
    implementations_per_strategy: 3
    initial_strategies_per_guide_call: 1
    strategies_per_guide_call: 3
    initial_strategy_reasoning_effort: low
    adaptive_implementation_racing: true
    adaptive_unguided_racing: true
    pilot_candidates: 2
    stop_racing_on_improvement: true
    adaptive_pilot_sizing: true
    long_horizon_scheduler: true
    strategy_replays_per_portfolio: 1
    strategy_replay_min_productive_rounds: 2
    strategy_replay_ucb_exploration: 0.15
    long_horizon_exploration_interval: 12
    long_horizon_exploration_candidates: 1
```

Keep `low` as the robust opening default. `none` is much cheaper and can work well on simple,
well-specified solver reformulations, but should be enabled only after a representative evaluation;
on harder constructive tasks it can produce a weak first strategy.

Important implementation files:

```text
controller.py                         iteration state machine and budget admission
strategy.py                           strategy/reference prompts and parsing
database.py                           islands, MAP-Elites, archive and migration
skydiscover/search/utils/budget_curve.py  curve integration and plotting
```

BA-AUC comparisons are empirical. With stochastic models, no method can guarantee a strictly higher
curve on every seed and every unbounded run. Use repeated seeds, report mean and sample deviation,
and compare both final performance and actual spend.
