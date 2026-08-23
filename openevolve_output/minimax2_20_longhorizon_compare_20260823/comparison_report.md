# EfficientEvolve vs OpenEvolve: 20-iteration comparison

Task: `benchmarks/math/minimizing_max_min_dist/2`

Seed: `73`

The comparison uses each algorithm's intended configuration: EfficientEvolve uses
Luna for candidate generation and Terra for guidance; native OpenEvolve uses Terra
with low reasoning effort. Both runs use the same initial program, evaluator,
system prompt, random seed, and 20-iteration limit.

## Result

Both algorithms reached the same final score to numerical precision:

| Algorithm | Final `combined_score` | Cost to score >= 1.0 | Full-run API cost | API calls | Tokens |
|---|---:|---:|---:|---:|---:|
| EfficientEvolve | 1.0000028088804562 | $0.140985 | $0.581032 | 59 | 487,409 |
| OpenEvolve | 1.0000028088804558 | $0.191100 | $0.495200 | 20 | 102,205 |

EfficientEvolve reached the near-optimal target at iteration 6 and used 26.2%
less API cost to reach it. OpenEvolve was cheaper at moderate targets: it reached
0.90 at $0.0422 versus $0.12738, and 0.95 at $0.0976 versus $0.140985.

## BA-AUC

| Budget B | EfficientEvolve | OpenEvolve | Relative improvement |
|---:|---:|---:|---:|
| $0.05 | 0.433926 | 0.160252 | +170.8% |
| $0.10 | 0.662276 | 0.541198 | +22.4% |
| $0.20 | 0.811275 | 0.752841 | +7.8% |
| $0.50 | 0.924512 | 0.901138 | +2.6% |
| $1.00 | 0.962257 | 0.950570 | +1.2% |
| $100.00 | 0.999625 | 0.999508 | +0.01% |

EfficientEvolve wins BA-AUC at every tested budget. The advantage is largest at
small budgets because its first guided round quickly raises the incumbent, and its
anytime racing reaches the optimum before OpenEvolve does.

## Efficiency interpretation

- Search-to-high-quality efficiency: EfficientEvolve wins in this run.
- Fixed 20-iteration total cost: OpenEvolve wins. EfficientEvolve costs 17.3% more.
- Model/sample efficiency: OpenEvolve wins; EfficientEvolve performs more candidate
  evaluations and uses 4.77x as many tokens.
- Wall-clock efficiency: OpenEvolve wins in this run (about 6.5 minutes versus about
  13 minutes of active EfficientEvolve runtime), although the EfficientEvolve run
  was interrupted once by a local native-library crash.

EfficientEvolve's anytime racing activated at iterations 4 and 6, avoiding one
additional candidate call in each round. Its long-horizon diversity lane activated
at iteration 19 but did not improve an already optimal incumbent. Conservative UCB
strategy replay did not activate because no strategy met the required repeated,
BA-significant productivity threshold.

## Measurement caveat

EfficientEvolve encountered a local SciPy/multiprocessing `corrupted double-linked
list` crash after checkpoint 15 and was resumed safely from that checkpoint with
single-threaded BLAS settings. Two paid Luna calls immediately before the crash are
not present in the checkpoint-derived `budget_curve.jsonl`. Therefore the BA-AUC
tool reports a curve spend of approximately $0.5717, while merged API logs give the
actual full-run spend of $0.581032. These omitted calls happened after the optimal
incumbent had already been found, so adding them as a flat segment does not change
the reported BA-AUC values at the tested budgets.

This is a controlled single-seed case study, not a statistical conclusion. Multiple
seeds and task instances are needed to establish a reliable average advantage.
