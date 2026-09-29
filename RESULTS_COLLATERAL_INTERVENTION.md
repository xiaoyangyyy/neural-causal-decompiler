# Results: collateral-preserving tangent interventions

## Outcome

The frozen acceptance rule failed. The candidate reduced inactive linear-readout
coordinate leakage at every site in both seeds, but that proxy did not reliably
reduce collateral numerical error. Seed 3593 improved collateral NMSE at
`representation` and `head_linear`; seed 3594 worsened it at all three
sites. R5 remains incomplete.

Both confirmation runs used 9,216 disjoint worlds, 384 fixed training pairs,
2,048 held-out pairs, all 1,411 compatible masks, all 54 scalar groups, and
three sites. Complete regeneration and refitting replay passed.

## Weighted results

| Seed | Site | Baseline target | Preserved target | Baseline collateral | Preserved collateral | Baseline leakage | Preserved leakage |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3593 | representation | 1.8170 | 1.7229 | 9.8937 | 9.8452 | 0.2830 | 0.2481 |
| 3593 | head_linear | 1.7429 | 1.7205 | 9.9395 | 9.9142 | 1.1965 | 1.1066 |
| 3593 | head_tanh | 2.0461 | 2.0409 | 9.9274 | 9.9661 | 0.9179 | 0.7230 |
| 3594 | representation | 1.8196 | 1.9329 | 11.5219 | 11.6795 | 0.2332 | 0.1920 |
| 3594 | head_linear | 1.8276 | 1.8248 | 11.6371 | 11.6733 | 0.7348 | 0.6421 |
| 3594 | head_tanh | 2.1140 | 2.3316 | 11.7512 | 12.0717 | 0.4555 | 0.3403 |

Active-coordinate residuals remained below the frozen `1e-6` bound. Both
methods continued to beat ordinary biorthogonal writes on targeted NMSE at all
six seed/site cases.

## Interpretation

Suppressing changes in the other 53 linear probe coordinates does not ensure
that the teacher's nonlinear suffix preserves the corresponding symbolic
quantities. The remaining collateral error can arise from unmeasured hidden
directions, curved local geometry, probe error, or a mismatch between traced
symbolic variables and the teacher's actual computation. Further tuning of the
inactive-coordinate ridge against these held-out runs would be post-hoc and is
not justified.

Machine-readable evidence is
`validation/collateral_intervention_acceptance.json`; the frozen design is
`docs/COLLATERAL_PRESERVING_TANGENT_PROTOCOL.md`.
