# Raw statistical operation decompilation: seeds 793 / 794

Protocol: `docs/RAW_NUMERIC_PROTOCOL.md`. Both formal runs were independently replayed from worlds through AST execution, pair construction, mapping retraining, and metric reconstruction.

Each run contains 1,024 fit and 4,096 test worlds, 384 non-reused training pairs, 1,024 non-reused test pairs, eight structural-equivalence groups, and three neural cuts.

## Aggregate numerical and behavioral results

| Seed | Site | Method | Natural probe NMSE | Target NMSE | No-intervention target NMSE | Collateral NMSE | Informative pairs | Behavior accuracy |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 793 | representation | numeric | 1.141 | 1.308 | 1.638 | 0.557 | 134 | 26.9% |
| 793 | representation | behavior_only | 1.141 | 79.987 | 1.638 | 50.087 | 134 | 32.8% |
| 793 | representation | shuffled | 1.141 | 2.624 | 1.638 | 1.214 | 134 | 14.9% |
| 793 | representation | random | 1.141 | 28.760 | 1.638 | 27.456 | 134 | 3.0% |
| 793 | head_linear | numeric | 1.135 | 1.198 | 1.631 | 0.552 | 134 | 17.2% |
| 793 | head_linear | behavior_only | 1.135 | 161.234 | 1.631 | 152.750 | 134 | 24.6% |
| 793 | head_linear | shuffled | 1.135 | 2.418 | 1.631 | 1.458 | 134 | 10.4% |
| 793 | head_linear | random | 1.135 | 140.952 | 1.631 | 78.186 | 134 | 9.0% |
| 793 | head_tanh | numeric | 0.777 | 1.383 | 1.851 | 0.353 | 134 | 23.1% |
| 793 | head_tanh | behavior_only | 0.777 | 17.089 | 1.851 | 6.940 | 134 | 23.1% |
| 793 | head_tanh | shuffled | 0.777 | 1.785 | 1.851 | 0.632 | 134 | 20.9% |
| 793 | head_tanh | random | 0.777 | 9.921 | 1.851 | 7.542 | 134 | 5.2% |
| 794 | representation | numeric | 44.177 | 260.779 | 269.069 | 1.482 | 113 | 8.0% |
| 794 | representation | behavior_only | 44.177 | 416.633 | 269.069 | 58.153 | 113 | 22.1% |
| 794 | representation | shuffled | 44.177 | 284.702 | 269.069 | 3.104 | 113 | 14.2% |
| 794 | representation | random | 44.177 | 329.554 | 269.069 | 15.323 | 113 | 3.5% |
| 794 | head_linear | numeric | 44.152 | 263.322 | 268.978 | 1.466 | 113 | 6.2% |
| 794 | head_linear | behavior_only | 44.152 | 602.726 | 268.978 | 234.642 | 113 | 19.5% |
| 794 | head_linear | shuffled | 44.152 | 270.844 | 268.978 | 2.058 | 113 | 7.1% |
| 794 | head_linear | random | 44.152 | 321.311 | 268.978 | 22.870 | 113 | 0.9% |
| 794 | head_tanh | numeric | 39.841 | 270.854 | 269.799 | 1.118 | 113 | 9.7% |
| 794 | head_tanh | behavior_only | 39.841 | 347.933 | 269.799 | 49.267 | 113 | 21.2% |
| 794 | head_tanh | shuffled | 39.841 | 270.283 | 269.799 | 1.682 | 113 | 12.4% |
| 794 | head_tanh | random | 39.841 | 312.009 | 269.799 | 16.293 | 113 | 0.9% |

## Numeric method by operation group

| Seed | Site | Group | Natural NMSE | Target NMSE | No-intervention NMSE |
|---:|---|---|---:|---:|---:|
| 793 | representation | std_y | 0.568 | 0.875 | 2.844 |
| 793 | representation | var_stat_x | 0.357 | 0.940 | 1.271 |
| 793 | representation | std_x | 0.142 | 0.404 | 0.911 |
| 793 | representation | corr | 0.004 | 1.594 | 1.694 |
| 793 | representation | mean_x | 0.236 | 0.649 | 1.962 |
| 793 | representation | var_stat_y | 7.140 | 4.333 | 0.771 |
| 793 | representation | variance_ratio | 0.403 | 1.829 | 1.711 |
| 793 | representation | mean_y | 0.276 | 0.677 | 1.709 |
| 793 | head_linear | std_y | 0.558 | 0.985 | 2.794 |
| 793 | head_linear | var_stat_x | 0.320 | 0.809 | 1.305 |
| 793 | head_linear | std_x | 0.133 | 0.355 | 0.947 |
| 793 | head_linear | corr | 0.022 | 0.649 | 1.635 |
| 793 | head_linear | mean_x | 0.235 | 0.443 | 1.955 |
| 793 | head_linear | var_stat_y | 7.166 | 3.634 | 0.758 |
| 793 | head_linear | variance_ratio | 0.381 | 0.915 | 1.703 |
| 793 | head_linear | mean_y | 0.263 | 1.789 | 1.687 |
| 793 | head_tanh | std_y | 0.253 | 2.486 | 3.447 |
| 793 | head_tanh | var_stat_x | 0.080 | 1.055 | 1.474 |
| 793 | head_tanh | std_x | 0.022 | 0.601 | 1.063 |
| 793 | head_tanh | corr | 0.033 | 1.814 | 1.729 |
| 793 | head_tanh | mean_x | 0.139 | 1.192 | 2.132 |
| 793 | head_tanh | var_stat_y | 5.433 | 0.484 | 0.475 |
| 793 | head_tanh | variance_ratio | 0.099 | 1.816 | 2.426 |
| 793 | head_tanh | mean_y | 0.153 | 1.679 | 1.671 |
| 794 | representation | std_y | 0.296 | 0.615 | 1.851 |
| 794 | representation | var_stat_x | 162.527 | 0.797 | 1.939 |
| 794 | representation | std_x | 2.948 | 0.731 | 1.550 |
| 794 | representation | corr | 0.005 | 3.298 | 2.271 |
| 794 | representation | mean_x | 0.336 | 1.003 | 1.878 |
| 794 | representation | var_stat_y | 1.746 | 0.792 | 2.068 |
| 794 | representation | variance_ratio | 185.260 | 1900.201 | 1954.981 |
| 794 | representation | mean_y | 0.295 | 0.552 | 1.484 |
| 794 | head_linear | std_y | 0.283 | 0.370 | 1.798 |
| 794 | head_linear | var_stat_x | 162.255 | 0.572 | 2.017 |
| 794 | head_linear | std_x | 2.921 | 0.487 | 1.611 |
| 794 | head_linear | corr | 0.016 | 3.123 | 2.227 |
| 794 | head_linear | mean_x | 0.327 | 0.594 | 1.792 |
| 794 | head_linear | var_stat_y | 1.682 | 1.582 | 2.017 |
| 794 | head_linear | variance_ratio | 185.446 | 1919.248 | 1954.390 |
| 794 | head_linear | mean_y | 0.290 | 0.625 | 1.492 |
| 794 | head_tanh | std_y | 0.049 | 1.990 | 2.627 |
| 794 | head_tanh | var_stat_x | 146.637 | 1.487 | 3.228 |
| 794 | head_tanh | std_x | 1.877 | 0.927 | 1.439 |
| 794 | head_tanh | corr | 0.053 | 0.491 | 2.135 |
| 794 | head_tanh | mean_x | 0.205 | 1.358 | 2.130 |
| 794 | head_tanh | var_stat_y | 0.620 | 4.339 | 5.955 |
| 794 | head_tanh | variance_ratio | 169.053 | 1968.656 | 1955.085 |
| 794 | head_tanh | mean_y | 0.233 | 1.095 | 1.501 |

## Interpretation

- This is the first experiment in the repository that intervenes below the final statistical feature leaves. An internal variance exchange creates a hybrid computation from source variance and target-world downstream operations.
- Seed 793 shows moderate numeric target errors around one normalized variance. Behavior-only mappings have much larger numeric errors, confirming that behavioral agreement alone does not identify numerical semantics.
- In seed 794, the variance-ratio group has target NMSE around 1,900 and dominates the aggregate. Numeric training only slightly improves its no-intervention error. Several other groups improve, but the full eight-operation mechanism is not recovered.
- Behavior interchange accuracy remains low and behavior-only is often higher. Numerical recovery and categorical intervention fidelity are therefore not interchangeable claims.
- Dependence-kernel bandwidth/centering and cross-fit regression internals are still opaque operations in the current CDIR trace. Complete raw statistical algorithm recovery remains open.
- Seeds 793/794 are now observed results and cannot serve as untouched confirmation data for a revised method.
