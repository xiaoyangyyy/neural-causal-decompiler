# Cross-fit regression internal mechanism audit: seeds 993 / 994

Protocol: `docs/REGRESSION_NUMERIC_PROTOCOL.md`. Both runs were independently replayed from world generation through regression tracing, pair construction, mapping retraining, controls and metrics.

Each run has 5,120 worlds, 28 synchronized regression groups, 384 training pairs, 1,024 test pairs and 338 held-out compatible pair masks.

## Aggregate results

| Seed | Site | Method | Natural NMSE | Target NMSE | No-intervention NMSE | Collateral NMSE | Informative pairs | Behavior accuracy |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 993 | representation | numeric | 0.556 | 2.429 | 1.460 | 1.311 | 191 | 13.6% |
| 993 | representation | behavior_only | 0.556 | 151.420 | 1.460 | 158.849 | 191 | 25.1% |
| 993 | representation | shuffled | 0.556 | 3.994 | 1.460 | 2.263 | 191 | 15.7% |
| 993 | representation | random | 0.556 | 43.089 | 1.460 | 41.076 | 191 | 1.6% |
| 993 | head_linear | numeric | 0.567 | 2.002 | 1.369 | 1.272 | 191 | 4.7% |
| 993 | head_linear | behavior_only | 0.567 | 408.093 | 1.369 | 567.221 | 191 | 23.6% |
| 993 | head_linear | shuffled | 0.567 | 2.409 | 1.369 | 2.185 | 191 | 2.6% |
| 993 | head_linear | random | 0.567 | 267.993 | 1.369 | 327.219 | 191 | 4.7% |
| 993 | head_tanh | numeric | 0.642 | 1.891 | 1.446 | 0.976 | 191 | 13.6% |
| 993 | head_tanh | behavior_only | 0.642 | 106.734 | 1.446 | 99.572 | 191 | 18.8% |
| 993 | head_tanh | shuffled | 0.642 | 2.338 | 1.446 | 1.386 | 191 | 6.3% |
| 993 | head_tanh | random | 0.642 | 50.814 | 1.446 | 60.666 | 191 | 2.6% |
| 994 | representation | numeric | 0.608 | 2.997 | 1.796 | 1.594 | 153 | 7.8% |
| 994 | representation | behavior_only | 0.608 | 286.454 | 1.796 | 307.040 | 153 | 19.0% |
| 994 | representation | shuffled | 0.608 | 3.874 | 1.796 | 2.840 | 153 | 9.2% |
| 994 | representation | random | 0.608 | 120.136 | 1.796 | 115.564 | 153 | 0.0% |
| 994 | head_linear | numeric | 0.612 | 2.383 | 1.675 | 1.533 | 153 | 2.6% |
| 994 | head_linear | behavior_only | 0.612 | 266.391 | 1.675 | 330.682 | 153 | 12.4% |
| 994 | head_linear | shuffled | 0.612 | 3.647 | 1.675 | 2.584 | 153 | 1.3% |
| 994 | head_linear | random | 0.612 | 238.443 | 1.675 | 211.556 | 153 | 1.3% |
| 994 | head_tanh | numeric | 0.710 | 2.385 | 1.697 | 1.413 | 153 | 0.0% |
| 994 | head_tanh | behavior_only | 0.710 | 401.443 | 1.697 | 358.439 | 153 | 13.7% |
| 994 | head_tanh | shuffled | 0.710 | 2.957 | 1.697 | 2.013 | 153 | 2.6% |
| 994 | head_tanh | random | 0.710 | 232.511 | 1.697 | 187.644 | 153 | 1.3% |

## Numeric mapping by regression step

| Seed | Site | Direction | Fold | Step | Natural NMSE | Target NMSE | No-intervention NMSE |
|---:|---|---|---:|---|---:|---:|---:|
| 993 | representation | X_given_Y | 1 | beta_0 | 0.399 | 2.350 | 1.764 |
| 993 | representation | Y_given_X | 0 | beta_1 | 0.673 | 2.260 | 1.754 |
| 993 | representation | X_given_Y | 1 | beta_2 | 0.642 | 1.720 | 1.906 |
| 993 | representation | X_given_Y | 0 | beta_3 | 1.048 | 2.034 | 1.165 |
| 993 | representation | Y_given_X | 0 | mean_0 | 0.339 | 1.950 | 1.146 |
| 993 | representation | Y_given_X | 0 | beta_3 | 1.048 | 9.142 | 1.396 |
| 993 | representation | Y_given_X | 1 | beta_4 | 0.757 | 2.734 | 1.144 |
| 993 | representation | X_given_Y | 1 | beta_1 | 0.584 | 1.521 | 0.988 |
| 993 | representation | Y_given_X | 0 | beta_4 | 0.797 | 3.230 | 1.584 |
| 993 | representation | X_given_Y | 0 | beta_4 | 0.770 | 3.623 | 1.013 |
| 993 | representation | Y_given_X | 1 | beta_3 | 0.916 | 2.054 | 1.150 |
| 993 | representation | X_given_Y | 1 | std_0 | 0.162 | 1.549 | 1.194 |
| 993 | representation | Y_given_X | 1 | std_0 | 0.223 | 2.461 | 2.061 |
| 993 | representation | Y_given_X | 1 | beta_0 | 0.384 | 2.419 | 2.116 |
| 993 | representation | X_given_Y | 1 | beta_3 | 1.005 | 2.035 | 1.057 |
| 993 | representation | Y_given_X | 1 | beta_2 | 0.506 | 0.876 | 1.340 |
| 993 | representation | X_given_Y | 0 | beta_1 | 0.549 | 1.374 | 1.183 |
| 993 | representation | Y_given_X | 0 | beta_2 | 0.720 | 2.274 | 1.898 |
| 993 | representation | X_given_Y | 1 | beta_4 | 0.743 | 1.515 | 1.078 |
| 993 | representation | X_given_Y | 1 | mean_0 | 0.281 | 1.660 | 1.352 |
| 993 | representation | X_given_Y | 0 | mean_0 | 0.281 | 2.054 | 1.275 |
| 993 | representation | Y_given_X | 1 | mean_0 | 0.339 | 1.298 | 1.190 |
| 993 | representation | Y_given_X | 0 | std_0 | 0.258 | 3.221 | 1.794 |
| 993 | representation | X_given_Y | 0 | std_0 | 0.182 | 1.667 | 1.399 |
| 993 | representation | X_given_Y | 0 | beta_0 | 0.394 | 2.273 | 2.403 |
| 993 | representation | X_given_Y | 0 | beta_2 | 0.584 | 4.914 | 1.454 |
| 993 | representation | Y_given_X | 0 | beta_0 | 0.367 | 1.988 | 1.510 |
| 993 | representation | Y_given_X | 1 | beta_1 | 0.614 | 1.555 | 1.480 |
| 993 | head_linear | X_given_Y | 1 | beta_0 | 0.393 | 1.952 | 1.661 |
| 993 | head_linear | Y_given_X | 0 | beta_1 | 0.662 | 2.246 | 1.510 |
| 993 | head_linear | X_given_Y | 1 | beta_2 | 0.583 | 1.921 | 1.782 |
| 993 | head_linear | X_given_Y | 0 | beta_3 | 1.041 | 1.404 | 1.087 |
| 993 | head_linear | Y_given_X | 0 | mean_0 | 0.360 | 1.853 | 1.136 |
| 993 | head_linear | Y_given_X | 0 | beta_3 | 1.003 | 3.025 | 1.452 |
| 993 | head_linear | Y_given_X | 1 | beta_4 | 0.707 | 4.337 | 0.915 |
| 993 | head_linear | X_given_Y | 1 | beta_1 | 0.540 | 2.050 | 0.927 |
| 993 | head_linear | Y_given_X | 0 | beta_4 | 0.746 | 2.187 | 1.648 |
| 993 | head_linear | X_given_Y | 0 | beta_4 | 0.755 | 1.349 | 1.035 |
| 993 | head_linear | Y_given_X | 1 | beta_3 | 0.860 | 1.130 | 1.002 |
| 993 | head_linear | X_given_Y | 1 | std_0 | 0.299 | 0.906 | 0.871 |
| 993 | head_linear | Y_given_X | 1 | std_0 | 0.420 | 2.490 | 1.831 |
| 993 | head_linear | Y_given_X | 1 | beta_0 | 0.387 | 1.535 | 2.143 |
| 993 | head_linear | X_given_Y | 1 | beta_3 | 0.974 | 1.166 | 0.921 |
| 993 | head_linear | Y_given_X | 1 | beta_2 | 0.490 | 1.918 | 1.343 |
| 993 | head_linear | X_given_Y | 0 | beta_1 | 0.538 | 2.224 | 1.163 |
| 993 | head_linear | Y_given_X | 0 | beta_2 | 0.713 | 1.373 | 1.836 |
| 993 | head_linear | X_given_Y | 1 | beta_4 | 0.708 | 1.191 | 1.090 |
| 993 | head_linear | X_given_Y | 1 | mean_0 | 0.289 | 1.624 | 1.263 |
| 993 | head_linear | X_given_Y | 0 | mean_0 | 0.289 | 1.878 | 1.273 |
| 993 | head_linear | Y_given_X | 1 | mean_0 | 0.360 | 1.538 | 1.084 |
| 993 | head_linear | Y_given_X | 0 | std_0 | 0.463 | 1.990 | 1.419 |
| 993 | head_linear | X_given_Y | 0 | std_0 | 0.352 | 1.234 | 1.149 |
| 993 | head_linear | X_given_Y | 0 | beta_0 | 0.400 | 2.434 | 2.332 |
| 993 | head_linear | X_given_Y | 0 | beta_2 | 0.566 | 1.253 | 1.319 |
| 993 | head_linear | Y_given_X | 0 | beta_0 | 0.372 | 4.942 | 1.567 |
| 993 | head_linear | Y_given_X | 1 | beta_1 | 0.601 | 2.666 | 1.411 |
| 993 | head_tanh | X_given_Y | 1 | beta_0 | 0.392 | 2.030 | 1.954 |
| 993 | head_tanh | Y_given_X | 0 | beta_1 | 0.886 | 1.898 | 1.763 |
| 993 | head_tanh | X_given_Y | 1 | beta_2 | 0.662 | 2.400 | 1.877 |
| 993 | head_tanh | X_given_Y | 0 | beta_3 | 1.107 | 1.023 | 1.033 |
| 993 | head_tanh | Y_given_X | 0 | mean_0 | 0.431 | 1.896 | 1.200 |
| 993 | head_tanh | Y_given_X | 0 | beta_3 | 1.095 | 2.300 | 1.571 |
| 993 | head_tanh | Y_given_X | 1 | beta_4 | 0.801 | 1.507 | 0.884 |
| 993 | head_tanh | X_given_Y | 1 | beta_1 | 0.676 | 1.502 | 1.091 |
| 993 | head_tanh | Y_given_X | 0 | beta_4 | 0.892 | 1.847 | 1.817 |
| 993 | head_tanh | X_given_Y | 0 | beta_4 | 0.849 | 1.083 | 0.947 |
| 993 | head_tanh | Y_given_X | 1 | beta_3 | 0.926 | 1.241 | 1.011 |
| 993 | head_tanh | X_given_Y | 1 | std_0 | 0.330 | 1.279 | 1.031 |
| 993 | head_tanh | Y_given_X | 1 | std_0 | 0.463 | 2.212 | 2.021 |
| 993 | head_tanh | Y_given_X | 1 | beta_0 | 0.411 | 4.618 | 2.183 |
| 993 | head_tanh | X_given_Y | 1 | beta_3 | 1.032 | 1.109 | 1.008 |
| 993 | head_tanh | Y_given_X | 1 | beta_2 | 0.553 | 1.374 | 1.287 |
| 993 | head_tanh | X_given_Y | 0 | beta_1 | 0.657 | 1.633 | 1.226 |
| 993 | head_tanh | Y_given_X | 0 | beta_2 | 0.837 | 2.219 | 1.791 |
| 993 | head_tanh | X_given_Y | 1 | beta_4 | 0.828 | 2.354 | 1.260 |
| 993 | head_tanh | X_given_Y | 1 | mean_0 | 0.349 | 1.958 | 1.373 |
| 993 | head_tanh | X_given_Y | 0 | mean_0 | 0.349 | 1.499 | 1.314 |
| 993 | head_tanh | Y_given_X | 1 | mean_0 | 0.431 | 1.199 | 1.136 |
| 993 | head_tanh | Y_given_X | 0 | std_0 | 0.512 | 5.186 | 1.809 |
| 993 | head_tanh | X_given_Y | 0 | std_0 | 0.392 | 0.786 | 1.213 |
| 993 | head_tanh | X_given_Y | 0 | beta_0 | 0.386 | 2.336 | 2.208 |
| 993 | head_tanh | X_given_Y | 0 | beta_2 | 0.621 | 1.536 | 1.534 |
| 993 | head_tanh | Y_given_X | 0 | beta_0 | 0.367 | 1.433 | 1.450 |
| 993 | head_tanh | Y_given_X | 1 | beta_1 | 0.733 | 1.571 | 1.373 |
| 994 | representation | X_given_Y | 1 | beta_0 | 0.389 | 2.137 | 2.553 |
| 994 | representation | Y_given_X | 0 | beta_1 | 0.674 | 5.222 | 1.534 |
| 994 | representation | X_given_Y | 1 | beta_2 | 0.730 | 9.831 | 2.715 |
| 994 | representation | X_given_Y | 0 | beta_3 | 1.006 | 3.018 | 1.345 |
| 994 | representation | Y_given_X | 0 | mean_0 | 0.382 | 2.814 | 1.777 |
| 994 | representation | Y_given_X | 0 | beta_3 | 1.050 | 2.742 | 1.339 |
| 994 | representation | Y_given_X | 1 | beta_4 | 0.841 | 3.655 | 2.362 |
| 994 | representation | X_given_Y | 1 | beta_1 | 0.629 | 1.262 | 1.213 |
| 994 | representation | Y_given_X | 0 | beta_4 | 0.795 | 3.331 | 1.463 |
| 994 | representation | X_given_Y | 0 | beta_4 | 0.802 | 2.258 | 1.258 |
| 994 | representation | Y_given_X | 1 | beta_3 | 0.974 | 1.277 | 0.634 |
| 994 | representation | X_given_Y | 1 | std_0 | 0.413 | 4.471 | 3.877 |
| 994 | representation | Y_given_X | 1 | std_0 | 0.193 | 4.229 | 2.689 |
| 994 | representation | Y_given_X | 1 | beta_0 | 0.357 | 1.475 | 1.390 |
| 994 | representation | X_given_Y | 1 | beta_3 | 1.136 | 4.593 | 2.575 |
| 994 | representation | Y_given_X | 1 | beta_2 | 0.553 | 1.923 | 1.564 |
| 994 | representation | X_given_Y | 0 | beta_1 | 0.635 | 1.763 | 1.230 |
| 994 | representation | Y_given_X | 0 | beta_2 | 0.675 | 4.057 | 0.830 |
| 994 | representation | X_given_Y | 1 | beta_4 | 0.889 | 2.733 | 1.726 |
| 994 | representation | X_given_Y | 1 | mean_0 | 0.424 | 2.195 | 1.389 |
| 994 | representation | X_given_Y | 0 | mean_0 | 0.424 | 3.290 | 2.169 |
| 994 | representation | Y_given_X | 1 | mean_0 | 0.382 | 2.121 | 1.766 |
| 994 | representation | Y_given_X | 0 | std_0 | 0.219 | 3.152 | 1.752 |
| 994 | representation | X_given_Y | 0 | std_0 | 0.478 | 2.191 | 2.333 |
| 994 | representation | X_given_Y | 0 | beta_0 | 0.396 | 2.332 | 1.917 |
| 994 | representation | X_given_Y | 0 | beta_2 | 0.620 | 1.596 | 1.477 |
| 994 | representation | Y_given_X | 0 | beta_0 | 0.392 | 1.813 | 1.845 |
| 994 | representation | Y_given_X | 1 | beta_1 | 0.569 | 2.630 | 1.949 |
| 994 | head_linear | X_given_Y | 1 | beta_0 | 0.399 | 1.692 | 2.496 |
| 994 | head_linear | Y_given_X | 0 | beta_1 | 0.578 | 2.236 | 1.392 |
| 994 | head_linear | X_given_Y | 1 | beta_2 | 0.660 | 3.087 | 2.759 |
| 994 | head_linear | X_given_Y | 0 | beta_3 | 0.963 | 3.138 | 1.290 |
| 994 | head_linear | Y_given_X | 0 | mean_0 | 0.385 | 2.071 | 1.758 |
| 994 | head_linear | Y_given_X | 0 | beta_3 | 1.039 | 1.805 | 1.111 |
| 994 | head_linear | Y_given_X | 1 | beta_4 | 0.816 | 2.867 | 1.984 |
| 994 | head_linear | X_given_Y | 1 | beta_1 | 0.611 | 1.420 | 1.187 |
| 994 | head_linear | Y_given_X | 0 | beta_4 | 0.783 | 2.176 | 1.407 |
| 994 | head_linear | X_given_Y | 0 | beta_4 | 0.772 | 2.681 | 1.160 |
| 994 | head_linear | Y_given_X | 1 | beta_3 | 0.955 | 1.397 | 0.666 |
| 994 | head_linear | X_given_Y | 1 | std_0 | 0.607 | 3.683 | 3.507 |
| 994 | head_linear | Y_given_X | 1 | std_0 | 0.320 | 1.910 | 2.184 |
| 994 | head_linear | Y_given_X | 1 | beta_0 | 0.369 | 4.127 | 1.342 |
| 994 | head_linear | X_given_Y | 1 | beta_3 | 1.090 | 3.077 | 2.503 |
| 994 | head_linear | Y_given_X | 1 | beta_2 | 0.505 | 1.633 | 1.419 |
| 994 | head_linear | X_given_Y | 0 | beta_1 | 0.606 | 1.610 | 1.222 |
| 994 | head_linear | Y_given_X | 0 | beta_2 | 0.639 | 1.143 | 0.805 |
| 994 | head_linear | X_given_Y | 1 | beta_4 | 0.869 | 2.261 | 1.755 |
| 994 | head_linear | X_given_Y | 1 | mean_0 | 0.451 | 1.698 | 1.420 |
| 994 | head_linear | X_given_Y | 0 | mean_0 | 0.451 | 3.686 | 1.900 |
| 994 | head_linear | Y_given_X | 1 | mean_0 | 0.385 | 1.794 | 1.724 |
| 994 | head_linear | Y_given_X | 0 | std_0 | 0.335 | 2.018 | 1.299 |
| 994 | head_linear | X_given_Y | 0 | std_0 | 0.635 | 2.671 | 2.196 |
| 994 | head_linear | X_given_Y | 0 | beta_0 | 0.391 | 2.451 | 1.884 |
| 994 | head_linear | X_given_Y | 0 | beta_2 | 0.580 | 1.269 | 1.322 |
| 994 | head_linear | Y_given_X | 0 | beta_0 | 0.398 | 1.870 | 1.815 |
| 994 | head_linear | Y_given_X | 1 | beta_1 | 0.530 | 5.358 | 1.710 |
| 994 | head_tanh | X_given_Y | 1 | beta_0 | 0.438 | 2.633 | 2.602 |
| 994 | head_tanh | Y_given_X | 0 | beta_1 | 0.748 | 1.728 | 1.287 |
| 994 | head_tanh | X_given_Y | 1 | beta_2 | 0.795 | 6.611 | 2.541 |
| 994 | head_tanh | X_given_Y | 0 | beta_3 | 1.041 | 1.980 | 1.064 |
| 994 | head_tanh | Y_given_X | 0 | mean_0 | 0.423 | 1.877 | 1.664 |
| 994 | head_tanh | Y_given_X | 0 | beta_3 | 1.221 | 2.109 | 1.375 |
| 994 | head_tanh | Y_given_X | 1 | beta_4 | 0.899 | 3.008 | 2.159 |
| 994 | head_tanh | X_given_Y | 1 | beta_1 | 0.868 | 1.913 | 1.036 |
| 994 | head_tanh | Y_given_X | 0 | beta_4 | 0.947 | 2.052 | 1.516 |
| 994 | head_tanh | X_given_Y | 0 | beta_4 | 0.878 | 1.686 | 1.001 |
| 994 | head_tanh | Y_given_X | 1 | beta_3 | 1.033 | 1.435 | 0.668 |
| 994 | head_tanh | X_given_Y | 1 | std_0 | 0.700 | 4.368 | 3.899 |
| 994 | head_tanh | Y_given_X | 1 | std_0 | 0.404 | 2.602 | 1.920 |
| 994 | head_tanh | Y_given_X | 1 | beta_0 | 0.355 | 2.617 | 1.713 |
| 994 | head_tanh | X_given_Y | 1 | beta_3 | 1.193 | 2.998 | 2.532 |
| 994 | head_tanh | Y_given_X | 1 | beta_2 | 0.546 | 2.021 | 1.415 |
| 994 | head_tanh | X_given_Y | 0 | beta_1 | 0.809 | 1.748 | 1.449 |
| 994 | head_tanh | Y_given_X | 0 | beta_2 | 0.690 | 0.733 | 0.710 |
| 994 | head_tanh | X_given_Y | 1 | beta_4 | 1.061 | 2.014 | 1.564 |
| 994 | head_tanh | X_given_Y | 1 | mean_0 | 0.495 | 2.544 | 1.409 |
| 994 | head_tanh | X_given_Y | 0 | mean_0 | 0.495 | 2.853 | 1.962 |
| 994 | head_tanh | Y_given_X | 1 | mean_0 | 0.423 | 2.942 | 1.872 |
| 994 | head_tanh | Y_given_X | 0 | std_0 | 0.441 | 1.679 | 1.452 |
| 994 | head_tanh | X_given_Y | 0 | std_0 | 0.811 | 1.905 | 2.225 |
| 994 | head_tanh | X_given_Y | 0 | beta_0 | 0.388 | 2.260 | 1.932 |
| 994 | head_tanh | X_given_Y | 0 | beta_2 | 0.740 | 1.518 | 1.392 |
| 994 | head_tanh | Y_given_X | 0 | beta_0 | 0.390 | 2.651 | 1.727 |
| 994 | head_tanh | Y_given_X | 1 | beta_1 | 0.646 | 2.643 | 1.719 |

## Interpretation

- Natural probes recover substantial coefficient and normalization information (pooled NMSE 0.56–0.71), but this does not imply that the readout directions implement the computation.
- Numeric interchange target NMSE is 1.89–3.00 and is worse than the no-intervention baseline in every aggregate comparison. It is nevertheless far better than behavior-only and random controls.
- Behavior-only mappings can attain higher categorical accuracy while increasing numerical error by two or three orders of magnitude. Categorical effects alone are inadequate evidence for algorithm recovery.
- Informative-pair behavior accuracy remains low: at most 13.6% for the numeric method. The full cross-fit regression mechanism has not been causally aligned.
- All declared raw statistical operations are now executable and internally addressable, but the frozen teachers do not show high-fidelity interchange under the tested linear orthogonal mappings.
- Seeds 993/994 are observed and cannot be reused for confirmation of a revised mapping model.
