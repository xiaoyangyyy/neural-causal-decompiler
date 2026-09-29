# Dependence-kernel internal mechanism audit: seeds 893 / 894

Protocol: `docs/DEPENDENCE_NUMERIC_PROTOCOL.md`. Both runs were independently replayed through world generation, expanded CDIR execution, causal pair construction, mapping retraining, controls and metrics.

Each run contains 5,120 worlds, 18 dependence-step groups, 384 training pairs and 1,024 test pairs. There are 129 held-out compatible pair masks after excluding causal ancestor/descendant combinations.

## Aggregate results

| Seed | Site | Method | Natural NMSE | Target NMSE | No-intervention NMSE | Collateral NMSE | Informative pairs | Behavior accuracy |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 893 | representation | numeric | 0.253 | 1.930 | 1.727 | 0.866 | 20 | 15.0% |
| 893 | representation | behavior_only | 0.253 | 43.044 | 1.727 | 47.550 | 20 | 15.0% |
| 893 | representation | shuffled | 0.253 | 2.681 | 1.727 | 1.909 | 20 | 20.0% |
| 893 | representation | random | 0.253 | 17.166 | 1.727 | 11.492 | 20 | 0.0% |
| 893 | head_linear | numeric | 0.267 | 1.934 | 1.672 | 0.922 | 20 | 0.0% |
| 893 | head_linear | behavior_only | 0.267 | 771.216 | 1.672 | 596.133 | 20 | 20.0% |
| 893 | head_linear | shuffled | 0.267 | 3.864 | 1.672 | 3.472 | 20 | 0.0% |
| 893 | head_linear | random | 0.267 | 363.637 | 1.672 | 420.444 | 20 | 0.0% |
| 893 | head_tanh | numeric | 0.201 | 1.535 | 1.755 | 0.597 | 20 | 35.0% |
| 893 | head_tanh | behavior_only | 0.201 | 95.119 | 1.755 | 118.453 | 20 | 35.0% |
| 893 | head_tanh | shuffled | 0.201 | 2.273 | 1.755 | 1.229 | 20 | 5.0% |
| 893 | head_tanh | random | 0.201 | 28.265 | 1.755 | 39.742 | 20 | 0.0% |
| 894 | representation | numeric | 0.244 | 2.312 | 1.889 | 1.079 | 24 | 4.2% |
| 894 | representation | behavior_only | 0.244 | 72.991 | 1.889 | 77.031 | 24 | 4.2% |
| 894 | representation | shuffled | 0.244 | 2.943 | 1.889 | 2.284 | 24 | 8.3% |
| 894 | representation | random | 0.244 | 23.185 | 1.889 | 21.649 | 24 | 0.0% |
| 894 | head_linear | numeric | 0.257 | 1.739 | 1.805 | 0.629 | 24 | 0.0% |
| 894 | head_linear | behavior_only | 0.257 | 347.241 | 1.805 | 315.062 | 24 | 8.3% |
| 894 | head_linear | shuffled | 0.257 | 2.825 | 1.805 | 2.903 | 24 | 8.3% |
| 894 | head_linear | random | 0.257 | 94.448 | 1.805 | 106.041 | 24 | 4.2% |
| 894 | head_tanh | numeric | 0.219 | 1.883 | 1.914 | 0.746 | 24 | 0.0% |
| 894 | head_tanh | behavior_only | 0.219 | 283.444 | 1.914 | 255.487 | 24 | 0.0% |
| 894 | head_tanh | shuffled | 0.219 | 2.846 | 1.914 | 2.027 | 24 | 4.2% |
| 894 | head_tanh | random | 0.219 | 172.794 | 1.914 | 118.051 | 24 | 4.2% |

## Numeric mapping by dependence step

| Seed | Site | Feature | Step | Natural NMSE | Target NMSE | No-intervention NMSE |
|---:|---|---|---|---:|---:|---:|
| 893 | representation | dep_xy | numerator | 0.135 | 2.632 | 1.520 |
| 893 | representation | resdep_xy | bandwidth_1 | 0.252 | 0.852 | 1.750 |
| 893 | representation | dep_xy | bandwidth_1 | 0.312 | 1.806 | 1.872 |
| 893 | representation | resdep_xy | bandwidth_0 | 0.037 | 1.847 | 1.696 |
| 893 | representation | resdep_yx | bandwidth_1 | 0.247 | 1.594 | 2.024 |
| 893 | representation | resdep_yx | bandwidth_0 | 0.040 | 1.214 | 1.752 |
| 893 | representation | resdep_xy | denominator | 0.263 | 2.060 | 1.909 |
| 893 | representation | dep_xy | energy_1 | 0.057 | 2.285 | 2.080 |
| 893 | representation | resdep_yx | numerator | 0.664 | 1.277 | 1.136 |
| 893 | representation | resdep_xy | numerator | 0.627 | 2.692 | 1.494 |
| 893 | representation | dep_xy | denominator | 0.044 | 1.611 | 1.777 |
| 893 | representation | dep_xy | bandwidth_0 | 0.271 | 1.540 | 1.539 |
| 893 | representation | resdep_yx | energy_0 | 0.057 | 1.480 | 1.546 |
| 893 | representation | resdep_xy | energy_0 | 0.051 | 1.907 | 1.754 |
| 893 | representation | resdep_xy | energy_1 | 0.590 | 3.507 | 1.641 |
| 893 | representation | dep_xy | energy_0 | 0.051 | 2.120 | 2.119 |
| 893 | representation | resdep_yx | denominator | 0.253 | 1.817 | 1.673 |
| 893 | representation | resdep_yx | energy_1 | 0.610 | 2.286 | 1.840 |
| 893 | head_linear | dep_xy | numerator | 0.136 | 2.951 | 1.515 |
| 893 | head_linear | resdep_xy | bandwidth_1 | 0.251 | 0.737 | 1.799 |
| 893 | head_linear | dep_xy | bandwidth_1 | 0.314 | 2.138 | 1.839 |
| 893 | head_linear | resdep_xy | bandwidth_0 | 0.045 | 1.697 | 1.619 |
| 893 | head_linear | resdep_yx | bandwidth_1 | 0.241 | 2.644 | 1.927 |
| 893 | head_linear | resdep_yx | bandwidth_0 | 0.052 | 1.453 | 1.720 |
| 893 | head_linear | resdep_xy | denominator | 0.271 | 2.354 | 1.790 |
| 893 | head_linear | dep_xy | energy_1 | 0.112 | 2.216 | 2.027 |
| 893 | head_linear | resdep_yx | numerator | 0.659 | 1.512 | 1.064 |
| 893 | head_linear | resdep_xy | numerator | 0.599 | 1.609 | 1.446 |
| 893 | head_linear | dep_xy | denominator | 0.074 | 1.861 | 1.737 |
| 893 | head_linear | dep_xy | bandwidth_0 | 0.257 | 1.719 | 1.481 |
| 893 | head_linear | resdep_yx | energy_0 | 0.112 | 2.002 | 1.652 |
| 893 | head_linear | resdep_xy | energy_0 | 0.091 | 1.833 | 1.705 |
| 893 | head_linear | resdep_xy | energy_1 | 0.601 | 2.414 | 1.483 |
| 893 | head_linear | dep_xy | energy_0 | 0.091 | 1.668 | 2.045 |
| 893 | head_linear | resdep_yx | denominator | 0.266 | 2.097 | 1.623 |
| 893 | head_linear | resdep_yx | energy_1 | 0.636 | 1.903 | 1.652 |
| 893 | head_tanh | dep_xy | numerator | 0.047 | 0.655 | 1.503 |
| 893 | head_tanh | resdep_xy | bandwidth_1 | 0.099 | 1.537 | 1.891 |
| 893 | head_tanh | dep_xy | bandwidth_1 | 0.123 | 1.401 | 2.318 |
| 893 | head_tanh | resdep_xy | bandwidth_0 | 0.044 | 1.515 | 1.633 |
| 893 | head_tanh | resdep_yx | bandwidth_1 | 0.096 | 0.465 | 1.836 |
| 893 | head_tanh | resdep_yx | bandwidth_0 | 0.055 | 1.412 | 1.803 |
| 893 | head_tanh | resdep_xy | denominator | 0.236 | 1.901 | 1.769 |
| 893 | head_tanh | dep_xy | energy_1 | 0.137 | 2.023 | 1.964 |
| 893 | head_tanh | resdep_yx | numerator | 0.450 | 1.253 | 1.233 |
| 893 | head_tanh | resdep_xy | numerator | 0.438 | 1.523 | 1.640 |
| 893 | head_tanh | dep_xy | denominator | 0.099 | 1.687 | 1.749 |
| 893 | head_tanh | dep_xy | bandwidth_0 | 0.118 | 1.326 | 1.675 |
| 893 | head_tanh | resdep_yx | energy_0 | 0.137 | 1.598 | 1.530 |
| 893 | head_tanh | resdep_xy | energy_0 | 0.137 | 1.932 | 1.785 |
| 893 | head_tanh | resdep_xy | energy_1 | 0.508 | 1.681 | 1.653 |
| 893 | head_tanh | dep_xy | energy_0 | 0.137 | 2.056 | 2.006 |
| 893 | head_tanh | resdep_yx | denominator | 0.236 | 1.885 | 1.797 |
| 893 | head_tanh | resdep_yx | energy_1 | 0.519 | 1.795 | 1.835 |
| 894 | representation | dep_xy | numerator | 0.141 | 2.827 | 2.254 |
| 894 | representation | resdep_xy | bandwidth_1 | 0.221 | 1.501 | 1.775 |
| 894 | representation | dep_xy | bandwidth_1 | 0.311 | 2.373 | 2.551 |
| 894 | representation | resdep_xy | bandwidth_0 | 0.037 | 2.228 | 2.277 |
| 894 | representation | resdep_yx | bandwidth_1 | 0.227 | 0.970 | 1.684 |
| 894 | representation | resdep_yx | bandwidth_0 | 0.035 | 1.339 | 1.403 |
| 894 | representation | resdep_xy | denominator | 0.259 | 2.061 | 1.700 |
| 894 | representation | dep_xy | energy_1 | 0.059 | 2.440 | 2.121 |
| 894 | representation | resdep_yx | numerator | 0.581 | 2.538 | 1.872 |
| 894 | representation | resdep_xy | numerator | 0.624 | 3.014 | 2.257 |
| 894 | representation | dep_xy | denominator | 0.044 | 1.781 | 2.090 |
| 894 | representation | dep_xy | bandwidth_0 | 0.285 | 2.660 | 1.323 |
| 894 | representation | resdep_yx | energy_0 | 0.059 | 2.167 | 1.927 |
| 894 | representation | resdep_xy | energy_0 | 0.051 | 1.915 | 2.251 |
| 894 | representation | resdep_xy | energy_1 | 0.596 | 2.805 | 1.491 |
| 894 | representation | dep_xy | energy_0 | 0.051 | 1.704 | 1.934 |
| 894 | representation | resdep_yx | denominator | 0.244 | 5.191 | 1.461 |
| 894 | representation | resdep_yx | energy_1 | 0.565 | 2.182 | 1.548 |
| 894 | head_linear | dep_xy | numerator | 0.136 | 1.865 | 2.219 |
| 894 | head_linear | resdep_xy | bandwidth_1 | 0.215 | 0.599 | 1.754 |
| 894 | head_linear | dep_xy | bandwidth_1 | 0.330 | 1.222 | 2.455 |
| 894 | head_linear | resdep_xy | bandwidth_0 | 0.044 | 2.384 | 2.217 |
| 894 | head_linear | resdep_yx | bandwidth_1 | 0.215 | 0.538 | 1.610 |
| 894 | head_linear | resdep_yx | bandwidth_0 | 0.042 | 1.419 | 1.410 |
| 894 | head_linear | resdep_xy | denominator | 0.259 | 1.912 | 1.735 |
| 894 | head_linear | dep_xy | energy_1 | 0.100 | 1.697 | 1.982 |
| 894 | head_linear | resdep_yx | numerator | 0.588 | 2.039 | 1.703 |
| 894 | head_linear | resdep_xy | numerator | 0.598 | 2.165 | 2.173 |
| 894 | head_linear | dep_xy | denominator | 0.074 | 1.709 | 2.115 |
| 894 | head_linear | dep_xy | bandwidth_0 | 0.299 | 1.121 | 1.276 |
| 894 | head_linear | resdep_yx | energy_0 | 0.100 | 1.717 | 1.771 |
| 894 | head_linear | resdep_xy | energy_0 | 0.098 | 1.842 | 2.046 |
| 894 | head_linear | resdep_xy | energy_1 | 0.609 | 3.041 | 1.287 |
| 894 | head_linear | dep_xy | energy_0 | 0.098 | 2.001 | 1.865 |
| 894 | head_linear | resdep_yx | denominator | 0.247 | 1.492 | 1.404 |
| 894 | head_linear | resdep_yx | energy_1 | 0.574 | 2.355 | 1.440 |
| 894 | head_tanh | dep_xy | numerator | 0.108 | 2.227 | 2.396 |
| 894 | head_tanh | resdep_xy | bandwidth_1 | 0.169 | 0.659 | 1.937 |
| 894 | head_tanh | dep_xy | bandwidth_1 | 0.102 | 1.649 | 2.858 |
| 894 | head_tanh | resdep_xy | bandwidth_0 | 0.039 | 1.285 | 2.320 |
| 894 | head_tanh | resdep_yx | bandwidth_1 | 0.191 | 1.254 | 1.605 |
| 894 | head_tanh | resdep_yx | bandwidth_0 | 0.040 | 1.423 | 1.465 |
| 894 | head_tanh | resdep_xy | denominator | 0.291 | 1.781 | 1.741 |
| 894 | head_tanh | dep_xy | energy_1 | 0.104 | 2.534 | 1.988 |
| 894 | head_tanh | resdep_yx | numerator | 0.449 | 1.760 | 1.862 |
| 894 | head_tanh | resdep_xy | numerator | 0.485 | 2.448 | 2.334 |
| 894 | head_tanh | dep_xy | denominator | 0.105 | 2.154 | 2.013 |
| 894 | head_tanh | dep_xy | bandwidth_0 | 0.137 | 1.585 | 1.312 |
| 894 | head_tanh | resdep_yx | energy_0 | 0.104 | 1.995 | 1.862 |
| 894 | head_tanh | resdep_xy | energy_0 | 0.119 | 2.123 | 2.128 |
| 894 | head_tanh | resdep_xy | energy_1 | 0.574 | 1.762 | 1.453 |
| 894 | head_tanh | dep_xy | energy_0 | 0.119 | 2.221 | 2.024 |
| 894 | head_tanh | resdep_yx | denominator | 0.252 | 1.614 | 1.527 |
| 894 | head_tanh | resdep_yx | energy_1 | 0.552 | 3.198 | 1.571 |

## Interpretation

- Natural readout is strong: pooled NMSE is 0.20–0.27 across cuts and seeds. The hidden state contains substantial information about kernel bandwidths, energies, numerators and denominators.
- Interchange is much weaker. Numeric mappings have pooled target NMSE 1.54–2.31. Only head-tanh in seed 893 and head-linear/head-tanh in seed 894 modestly improve over the no-intervention baseline.
- Behavior-only mappings destroy numerical semantics by one to three orders of magnitude. This confirms that matching class changes does not identify the underlying numerical computation.
- Informative categorical samples are scarce (20 and 24), and intervention accuracy ranges from 0% to 35%. No claim of causal abstraction equivalence is supported.
- The experiment exposes dependence internals but cross-fit regression remains opaque. The complete raw causal-discovery algorithm is still not recovered.
- Seeds 893/894 have been observed and cannot be reused as untouched confirmation sets.
