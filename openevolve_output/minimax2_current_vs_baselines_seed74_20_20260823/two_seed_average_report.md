# Seed 73/74 mean comparison

This report averages the two independent runs at random seeds 73 and 74 for
`benchmarks/math/minimizing_max_min_dist/2`.

The seed-73 EfficientEvolve member is the later current-controller rerun at
`minimax2_eff_current_deferred_seed73_20_20260823/run_current`, not the older
EfficientEvolve run in the original seed-73 baseline report. The seed-73 EvoX
member uses its retained checkpoint prefix plus successful resumed suffix, so
discarded post-checkpoint calls are excluded from its fair path.

## Mean outcome and resource use

Target-cost means include successful runs only and are therefore accompanied
by the success count. Total-cost, final-score, calls, tokens, and wall-time
means always include both seeds.

| Algorithm | Mean final score | >=0.99 success / mean cost | >=1.0 success / mean cost | Mean total cost | Mean calls | Mean tokens | Mean active wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientEvolve | 1.000002809 | 2/2 / $0.026364 | 2/2 / **$0.027771** | **$0.027771** | **2.5** | **8,453** | **55 s** |
| OpenEvolve Native | 1.000002809 | 2/2 / $0.106300 | 2/2 / $0.246500 | $0.507031 | 20.0 | 90,203 | 429 s |
| GEPA Native | 0.995742409 | 2/2 / $0.116737 | 1/2 / $0.192208 | $1.308878 | 39.5 | 251,584 | 974 s |
| AdaEvolve | 0.980507886 | 1/2 / $0.038600 | 1/2 / $0.275700 | $0.784834 | 22.0 | 140,038 | 1,631 s |
| EvoX | 1.000002809 | 2/2 / $0.048602 | 2/2 / $0.188015 | $1.245197 | 60.5 | 312,247 | 2,278 s |

## Mean normalized BA-AUC

Each BA-AUC value is computed independently from that seed's chronological
best-score-versus-cumulative-cost curve, then the two scalar values are
averaged. Higher is better.

| Algorithm | B=$0.01 | B=$0.05 | B=$0.10 | B=$0.20 | B=$0.50 | B=$1.00 | B=$100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientEvolve | **0.679038** | **0.827752** | **0.913877** | **0.956940** | **0.982778** | **0.991390** | **0.999917** |
| OpenEvolve Native | 0.019791 | 0.366954 | 0.661883 | 0.819933 | 0.927107 | 0.963555 | 0.999638 |
| GEPA Native | 0.019791 | 0.297695 | 0.401666 | 0.548624 | 0.814650 | 0.905196 | 0.994837 |
| AdaEvolve | 0.019791 | 0.195773 | 0.497866 | 0.694710 | 0.855348 | 0.916896 | 0.979872 |
| EvoX | 0.019791 | 0.100908 | 0.521114 | 0.758428 | 0.902346 | 0.951175 | 0.999515 |

## Interpretation

- EfficientEvolve has the highest mean BA-AUC at every tested budget and
  reaches the strict target on both seeds.
- EfficientEvolve's mean strict-target and total cost are both $0.027771,
  because target-aware stopping ends each run as soon as the target is found.
- OpenEvolve and EvoX also reach 1.0 on both seeds, but their successful-run
  mean target costs are 8.88x and 6.77x EfficientEvolve's, respectively.
- GEPA and AdaEvolve reach 1.0 on only one of two seeds. Their reported target
  cost is conditional on that one success and must not be read as a 2-seed
  unconditional mean.
- Mean full-run cost is 18.26x higher for OpenEvolve, 47.13x for GEPA, 28.26x
  for AdaEvolve, and 44.84x for EvoX relative to EfficientEvolve.

Two seeds are enough to reveal substantial seed sensitivity, especially for
GEPA and AdaEvolve, but not enough for reliable confidence intervals or a
general cross-task claim.
