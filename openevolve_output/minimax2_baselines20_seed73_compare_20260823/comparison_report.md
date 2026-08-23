# Five-algorithm comparison on minimizing_max_min_dist/2

All runs use the same initial program, evaluator, task prompt, seed 73, and a
20-main-iteration limit. OpenEvolve, EvoX, AdaEvolve, and GEPA use
`gpt-5.6-terra` with low reasoning effort. EfficientEvolve uses its intended
two-tier configuration: Luna for implementations and Terra for guidance.

This is therefore a dollar-efficiency comparison of full algorithm
configurations, not a same-model algorithm-only ablation.

## Main results

| Algorithm | Final score | Cost to >=0.99 | Cost to >=1.0 | Fair completed-path cost | API calls | Tokens | Approx. active wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientEvolve | 1.0000028088804562 | $0.140985 | $0.140985 | $0.571692 | 57 | 459,892 | ~12.8 min |
| OpenEvolve | 1.0000028088804558 | $0.191100 | $0.191100 | $0.495231 | 20 | 102,205 | 6.5 min |
| GEPA | 1.0000028088804556 | $0.021400 | $0.192208 | $0.845642 | 26 | 207,081 | 8.3 min |
| EvoX | 1.0000028088804513 | $0.041652 | $0.320478 | $1.310624 | 65 | 336,018 | ~58 min |
| AdaEvolve | 0.9610129630271512 | not reached | not reached | $0.711628 | 21 | 128,214 | 8.3 min |

GEPA reaches 0.99148 on its first mutation for only $0.0214, giving it the
best early curve. EfficientEvolve is the cheapest algorithm to cross the
strict 1.0 reference. OpenEvolve has the lowest total cost when all algorithms
are forced to run 20 iterations.

## Normalized BA-AUC

| Algorithm | B=$0.05 | B=$0.10 | B=$0.20 | B=$0.50 | B=$1.00 | B=$100 |
|---|---:|---:|---:|---:|---:|---:|
| GEPA | **0.575598** | **0.783540** | **0.888867** | **0.955549** | **0.977776** | **0.999781** |
| EfficientEvolve | 0.433926 | 0.662276 | 0.811275 | 0.924512 | 0.962257 | 0.999625 |
| EvoX | 0.182025 | 0.586753 | 0.789118 | 0.913596 | 0.956799 | 0.999571 |
| OpenEvolve | 0.160252 | 0.541198 | 0.752841 | 0.901138 | 0.950570 | 0.999508 |
| AdaEvolve | 0.150209 | 0.379323 | 0.585475 | 0.790405 | 0.873645 | 0.960139 |

GEPA wins BA-AUC at every tested budget on this single seed. Its advantage is
mostly caused by getting almost all of the available score in the first call;
its exact-reference cost and full-run cost are not the best.

## Mechanism-level observations

- EfficientEvolve reaches the strict reference at iteration 6 through guided
  portfolio generation and anytime racing. Its cheap-model implementation
  tier makes the strict-target cost best, but continuing to 20 iterations
  adds about $0.431 without a material score gain.
- OpenEvolve has the simplest and most predictable resource profile: one
  Terra call per iteration. It is not first in BA-AUC or strict-target cost,
  but it is cheapest to run for the full fixed horizon.
- GEPA makes 20 mutation calls plus 6 merge calls. Early reflective mutation
  is excellent on this task; accepted merges mostly preserve rather than
  improve the incumbent, so most later merge/search spend is redundant after
  the reference is reached.
- EvoX reaches the reference at iteration 6, but frequently evolves and
  validates search-program code. Several meta-strategies fail validation and
  retry. More importantly, many evolved solution programs take 150-260
  seconds to evaluate, causing severe wall-time degradation. The objective
  has no runtime penalty, so EvoX has no selection pressure against this.
- AdaEvolve continues making incremental progress through iteration 20, but
  its two-island allocation spreads early compute across lineages and does
  not find the reference construction in this run. One paradigm-generation
  call adds overhead near the late plateau.

## Interruption accounting

EfficientEvolve suffered a native SciPy/multiprocessing crash after checkpoint
15. Its actual billed totals are 59 calls, 487,409 tokens, and $0.581032; the
fair completed path excludes two discarded post-checkpoint calls.

EvoX was gracefully stopped when an interactive message interrupted the
original process, then resumed from checkpoint 15. Its fair completed path is
the retained checkpoint prefix plus the successful resumed suffix. The actual
billed total, including eight discarded post-checkpoint calls, is 73 calls,
381,442 tokens, and $1.500600. Resume connectivity/label calls remain in the
fair path because they were needed by the completed resumed controller.

## Scope

This is one task instance and one seed. Random seed controls database and
selection randomness, but remote LLM generation is not guaranteed to be fully
deterministic. Repeated seeds are needed before treating the ranking as a
general result.
