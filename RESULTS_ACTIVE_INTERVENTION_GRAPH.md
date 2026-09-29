# Results: active intervention graph discovery

## Outcome

The frozen rule passed every criterion. Two known intervention levels per node
increased pooled exact-graph accuracy by 33.33 percentage points and pooled
directed-target accuracy by 38.57 points. Skeleton F1 improved, SHD fell, both
seeds replicated, and exact accuracy improved in all five environments.

Each seed used 2,400 shared worlds and 25,600 interventional datasets
(`2 * nodes` per world), each with 96 paired-exogenous samples. The control and
candidate had the same 31,828 parameters, initial weights, worlds, batch
orders, optimizer, and decoder. Both formal runs passed full observational and
interventional feature regeneration, retraining, prediction replay, and
zero-tolerance checkpoint comparison.

## Aggregate metrics

| Seed | Evidence | Exact graph | Mean SHD | Skeleton F1 | Directed accuracy |
|---:|---|---:|---:|---:|---:|
| 4793 | observational padded | 17.43% | 2.9132 | 0.7861 | 0.4856 |
| 4793 | active intervention | 49.65% | 1.3868 | 0.9146 | 0.8647 |
| 4794 | observational padded | 17.22% | 2.7771 | 0.7846 | 0.4882 |
| 4794 | active intervention | 51.67% | 1.0389 | 0.9246 | 0.8806 |

Pooled exact accuracy rose from 17.33% to 50.66%; directed-target accuracy rose
from 0.4869 to 0.8727; skeleton F1 rose from 0.7854 to 0.9196; SHD fell from
2.8451 to 1.2128. Exact gains were +30.73 points ID, +50.87 function, +32.81
noise, +23.78 scale, and +28.47 intervention.

## Interpretation

The previous direction failures were primarily information-limited, rather
than solvable by graph-head symmetry alone. Paired interventions provide a
large ordered causal signal, while observational conditional-dependence
features help separate direct from downstream effects. Scale remains the
hardest environment and 8-node exact recovery remains far from perfect.

This result establishes a strong interventional R9 branch under a specific
query model: known intervention target and level, two levels per node, and
shared exogenous draws between observation and intervention. It does not prove
purely observational identifiability, historical-teacher fidelity, exact graph
recovery, or general causal discovery. R9 remains incomplete in its broader
observational/decompilation scope.

Machine-readable evidence is `validation/active_intervention_acceptance.json`;
the frozen protocol is `docs/ACTIVE_INTERVENTION_GRAPH_PROTOCOL.md`.
