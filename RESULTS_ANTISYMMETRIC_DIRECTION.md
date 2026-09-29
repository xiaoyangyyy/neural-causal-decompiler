# Results: antisymmetric direction graph head

## Outcome

The frozen rule failed. Exact node-swap structure improved every aggregate
metric in seed 4393, but the improvement did not replicate in seed 4394.
Across both seeds, exact-graph accuracy increased by only 0.38 percentage
points and directed-target accuracy was effectively unchanged. R9 remains
incomplete.

Each seed used 2,400 shared worlds. The candidate retained the factorized
skeleton path and replaced only the generic orientation outputs with an exact
antisymmetric directed score plus a symmetric undirected score. Both formal
runs passed complete retraining and zero-tolerance checkpoint replay.

## Aggregate metrics

| Seed | Head | Exact graph | Mean SHD | Skeleton F1 | Directed accuracy |
|---:|---|---:|---:|---:|---:|
| 4393 | factorized | 14.51% | 3.0597 | 0.7673 | 0.4543 |
| 4393 | antisymmetric | 15.63% | 2.9674 | 0.7777 | 0.4731 |
| 4394 | factorized | 12.85% | 2.6694 | 0.7670 | 0.4180 |
| 4394 | antisymmetric | 12.50% | 2.7396 | 0.7585 | 0.3990 |

Pooled exact accuracy rose from 13.68% to 14.06%, pooled SHD fell slightly from
2.8646 to 2.8535, and pooled skeleton F1 rose from 0.7671 to 0.7681. Pooled
directed-target accuracy changed from 0.43616 to 0.43608. Environment exact
changes were +3.13 points ID, +1.04 function, -1.04 noise, -1.04 scale, and
-0.17 intervention.

## Interpretation

The exact swap constraint is mathematically correct but is not a stable
statistical remedy. It helps when the learned representation exposes a clean
antisymmetric direction signal, as in seed 4393, and harms when useful
direction evidence is mixed with symmetric components or when the constraint
changes shared-representation optimization, as in seed 4394. A future direction
method should treat swap structure as a soft calibration or ensemble component
and must demonstrate stability across independent seeds.

Machine-readable evidence is
`validation/antisymmetric_direction_acceptance.json`; the frozen protocol is
`docs/ANTISYMMETRIC_DIRECTION_PROTOCOL.md`.
