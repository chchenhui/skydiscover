# Current EfficientEvolve vs baselines, seed 74

All runs use `benchmarks/math/minimizing_max_min_dist/2`, the same initial
program, evaluator, task prompt, random seed 74, and a 20-main-iteration
limit. The four baselines use `gpt-5.6-terra` with low reasoning effort.
Current EfficientEvolve uses its intended two-tier setup: Luna for the cheap
opening candidate and Terra for guide promotion.

This is a full-configuration dollar-efficiency comparison, not a same-model
algorithm-only ablation. EfficientEvolve includes target-aware stopping and
stopped after iteration 2; the requested baselines completed all 20 main
iterations. Cost-to-target and BA-AUC therefore provide the clearest anytime
comparison, while total cost also measures each controller's stopping policy.

## Main results

| Algorithm | First >=0.99 | First >=1.0 | Final score | Total cost | Calls | Tokens | Active wall time | Best-program eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientEvolve | iter 2 / $0.048058 | iter 2 / $0.048058 | 1.0000028088804540 | $0.048058 | 2 | 7,275 | 55 s | 0.375 s |
| OpenEvolve Native | iter 1 / $0.021500 | iter 10 / $0.301900 | 1.0000028088804542 | $0.518832 | 20 | 78,201 | 466 s | 4.869 s |
| GEPA Native | iter 3 / $0.212074 | not reached | 0.9914820086153843 | $1.772115 | 53 | 296,086 | 1,450 s | 0.00017 s |
| AdaEvolve | iter 1 / $0.038600 | iter 9 / $0.275700 | 1.0000028088804558 | $0.858040 | 23 | 151,861 | 2,762 s | 290.242 s |
| EvoX | iter 1 / $0.055552 | iter 1 / $0.055552 | 1.0000028088804558 | $1.179770 | 56 | 288,476 | 1,055 s | 0.687 s |

The threshold costs are reconstructed chronologically from every priced LLM
call, including failed retries and controller/meta-search calls. Total costs
come from each run's unrounded `llm_usage.json`.

## Normalized BA-AUC

BA-AUC is the area under the best-score-so-far curve over `[0, B]`, divided by
`B`. Higher is better.

| Algorithm | B=$0.01 | B=$0.05 | B=$0.10 | B=$0.20 | B=$0.50 | B=$1.00 | B=$100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientEvolve | **0.582134** | **0.700313** | **0.850158** | **0.925080** | **0.970034** | **0.985018** | **0.999853** |
| OpenEvolve Native | 0.019791 | 0.573655 | 0.782568 | 0.887025 | 0.953075 | 0.976539 | 0.999768 |
| GEPA Native | 0.019791 | 0.019791 | 0.019791 | 0.208380 | 0.673752 | 0.832617 | 0.989893 |
| AdaEvolve | 0.019791 | 0.241337 | 0.616409 | 0.803946 | 0.920290 | 0.960146 | 0.999604 |
| EvoX | 0.019791 | 0.019791 | 0.455476 | 0.727739 | 0.891097 | 0.945550 | 0.999458 |

## Interpretation

- EfficientEvolve wins BA-AUC at every tested budget. Its $0.001924 Luna
  opening raises the incumbent from 0.01979 to 0.71607; because that result is
  below the 0.9 defer threshold, Terra is promoted immediately and reaches the
  reference at cumulative cost $0.048058.
- OpenEvolve and AdaEvolve are cheaper to reach the looser 0.99 threshold, but
  need 6.28x and 5.74x EfficientEvolve's cost, respectively, to reach 1.0.
- EvoX reaches 1.0 in its first solution iteration for $0.055552, only 15.6%
  above EfficientEvolve's strict-target cost. Its fixed 20-round continuation,
  meta-strategy calls, validation failures, and retries increase total cost to
  24.55x EfficientEvolve's cost.
- GEPA is brittle on this seed: repeated SEARCH/REPLACE mismatch retries consume
  53 calls, and the best result remains the triangular-lattice score 0.991482.
- AdaEvolve reaches 1.0 at iteration 9, but several evolved programs take
  roughly 200--328 seconds to evaluate. Since runtime is not part of the score,
  the search has no pressure to avoid these expensive candidates.

## Scope

The random seed controls database and selection randomness, but remote LLM
generation is not guaranteed to be deterministic. Seed 74 is a useful repeat,
not enough by itself for a statistically reliable ranking across tasks.
