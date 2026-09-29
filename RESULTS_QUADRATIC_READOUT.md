# Quadratic-readout internal intervention results

Version 0.17 compares the existing linear ridge probe with an explicit diagonal-quadratic ridge probe under identical rank-one orthogonal warm starts, biorthogonal intervention training, frozen teachers, 54 symbolic scalar groups, and held-out intervention pairs. The protocol was frozen before seeds 3193/3194.

Each run used 1,024 fit worlds, 8,192 test worlds, 384 single-variable training pairs, 2,048 held-out single/two-variable test pairs, and all 1,411 compatible masks. Both complete replays regenerated worlds, traces, probes, mappings, controls, predictions, and metrics.

The table reports weighted NMSE as `natural / targeted / collateral`.

| Seed | Site | Linear biorthogonal | Quadratic biorthogonal |
|---:|---|---:|---:|
| 3193 | representation | 12.118 / 2.718 / 13.209 | 11.536 / 16.822 / 25.222 |
| 3193 | head_linear | 12.133 / 5.582 / 15.308 | 11.401 / 15.534 / 23.475 |
| 3193 | head_tanh | 11.631 / 2.460 / 12.768 | 11.641 / 6.200 / 15.866 |
| 3194 | representation | 22.704 / 23.567 / 12.336 | 22.051 / 30.980 / 22.537 |
| 3194 | head_linear | 22.741 / 23.557 / 13.028 | 21.885 / 30.383 / 18.353 |
| 3194 | head_tanh | 22.379 / 22.196 / 11.916 | 22.231 / 29.055 / 25.024 |

Quadratic natural-state NMSE improved at two sites in seed 3193 and all three sites in seed 3194. Targeted intervention NMSE worsened at every site in both seeds, and collateral NMSE also increased everywhere. In seed 3193 representation, quadratic targeted NMSE (16.822) was slightly worse than its shuffled-target control (16.547); the control gate therefore also failed.

The preregistered rule failed. More flexible natural-state measurement does not provide a more faithful intervention audit when patched hidden states leave the observed manifold. This is evidence against using an unconstrained quadratic probe as a remedy for the R5 error floor. It does not show that the symbolic variables are absent, and it does not establish exact circuit identity or complete decompilation.

Machine-readable evidence: `validation/quadratic_readout_acceptance.json`.
