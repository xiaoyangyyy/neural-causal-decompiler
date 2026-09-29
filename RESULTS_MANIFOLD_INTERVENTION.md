# Results: manifold-constrained internal interventions

## Outcome

The frozen acceptance rule failed because one collateral-error comparison missed
the strict no-increase requirement. The central numerical result nevertheless
replicated: natural-PCA tangent writes lowered weighted targeted NMSE at all
three sites in both confirmation seeds and beat equal-rank random subspaces by
orders of magnitude. This is finite intervention evidence, not proof of exact
circuit identity or causal truth. R5 remains incomplete.

Both runs covered 9,216 disjoint worlds, 384 training pairs, 2,048 held-out
pairs, all 1,411 compatible masks, all 54 scalar trace groups, and all three
sites. Complete world regeneration, target reconstruction, PCA refitting, and
mapping retraining replay passed.

## Weighted numerical results

| Seed | Site | PCA rank | Ordinary target | Natural-PCA target | Ordinary collateral | Natural-PCA collateral |
|---:|---|---:|---:|---:|---:|---:|
| 3393 | representation | 8 | 2.6886 | 1.7811 | 12.9795 | 12.4308 |
| 3393 | head_linear | 5 | 6.6769 | 1.6978 | 18.7429 | 12.4778 |
| 3393 | head_tanh | 9 | 2.3427 | 1.9045 | 12.5107 | 12.5148 |
| 3394 | representation | 8 | 4.0563 | 1.7066 | 15.0233 | 13.2018 |
| 3394 | head_linear | 5 | 5.8314 | 1.6400 | 15.6261 | 13.3008 |
| 3394 | head_tanh | 8 | 3.3733 | 1.8553 | 14.6022 | 13.3165 |

The maximum active-coordinate residual was below `7.5e-15`. Random-subspace
targeted NMSE ranged from 1,426 to 18,531, so equal dimension alone does not
explain the improvement.

## Interpretation

The learned natural hidden subspace contains directions that exchange the
linear read coordinates with substantially less numerical damage than the
trained unconstrained write basis. This directly supports the off-manifold
failure hypothesis raised by version 0.17.

The result does not satisfy the pre-registered rule. Seed 3393 at
`head_tanh` increased collateral NMSE by about 0.0040, and behavioral
interchange accuracy was mixed rather than consistently improved. A subsequent
method must preserve inactive coordinates or optimize collateral behavior
without selecting on these held-out results.

The machine-readable decision is
`validation/manifold_intervention_acceptance.json`; the frozen design is
`docs/MANIFOLD_CONSTRAINED_INTERVENTION_PROTOCOL.md`.
