# OpenEvolve Native Luna baseline: minimax2, seed 73

- Task: `benchmarks/math/minimizing_max_min_dist/2`
- Search: `openevolve_native`
- Model: `gpt-5.6-luna`, reasoning effort `low`
- Iterations: 20
- Random seed: 73

## Result

- Best iteration: 18
- `min_max_ratio`: 0.07350484985573853
- `combined_score`: 0.9474235703132188
- Test `combined_score`: 0.9474235703132188
- LLM calls: 22 (two first-attempt evaluation failures were retried)
- Tokens: 82,767
- Cost: $0.0364474
- Diff failures: 0

## Terra reference with the same task, seed, and 20-iteration limit

- Best iteration: 10
- `combined_score`: 1.0000028088804558
- LLM calls: 20
- Tokens: 102,205
- Cost: $0.4952308

Normalized BA-AUC reconstructed from the run logs:

| Budget | Luna | Terra |
| ---: | ---: | ---: |
| $0.01 | 0.701644 | 0.019791 |
| $0.05 | 0.882345 | 0.160252 |
| $0.10 | 0.914884 | 0.541198 |
| $0.50 | 0.940916 | 0.901138 |

Luna finished with a lower best score but used about 7.36% of Terra's cost.
