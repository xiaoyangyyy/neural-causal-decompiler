# Results: active end-to-end SCM recovery

## Outcome

The frozen rule failed. Active intervention graph inference reduced structural
edit distance and improved downstream causal metrics, but the graph gain did
not replicate in exact accuracy and the end-to-end improvements missed their
predeclared margins.

Both formal seeds covered all 15 combinations of 3/5/8 nodes and five
environments. They passed world and feature regeneration, graph re-inference,
neural mechanism retraining, symbolic refitting, and metric replay for the
observational, active, and oracle-DAG branches.

## Aggregate metrics

| Seed | Graph branch | Exact graph | Mean SHD | Structured truth NMSE | Structured intervention MAE |
|---:|---|---:|---:|---:|---:|
| 4993 | observational | 26.67% | 3.0000 | 0.5384 | 0.1738 |
| 4993 | active | 33.33% | 2.6000 | 0.5108 | 0.1660 |
| 4993 | oracle diagnostic | 100.00% | 0.0000 | 0.2378 | 0.0600 |
| 4994 | observational | 26.67% | 5.4667 | 0.5430 | 0.1566 |
| 4994 | active | 20.00% | 2.6667 | 0.4053 | 0.1056 |
| 4994 | oracle diagnostic | 100.00% | 0.0000 | 0.1249 | 0.0307 |

Pooled exact graph accuracy was unchanged at 26.67%, while pooled SHD fell from
4.2333 to 2.6333. Active structured truth NMSE fell from 0.5407 to 0.4580, a
15.29% reduction, and intervention-effect MAE fell from 0.1652 to 0.1358, a
17.80% reduction. Both downstream measures improved in each seed, but their
frozen requirement was 20%.

Within the active branch, structured equations used 1.036 rather than 2.506
nonconstant atoms, but pooled symbolic-to-neural NMSE increased from 0.003752
to 0.004128 and the improvement did not replicate. Active truth and
intervention errors remained 2.53 and 3.00 times their oracle-DAG diagnostics,
exceeding the frozen 1.75 and 2.00 limits.

## Interpretation

Lower graph edit distance predicts useful downstream progress better than exact
graph accuracy in these small confirmation samples: seed 4994 lost exact-graph
accuracy while obtaining the larger truth and intervention improvements.
Nevertheless, the remaining graph errors still dominate the gap to the oracle
branch, and the structured equation constraint is not yet a stable
neural-fidelity improvement.

This experiment supplies reproducible partial R10 evidence under active
interventions. It does not establish stable end-to-end SCM recovery, purely
observational identifiability, historical-teacher fidelity, or general
equation recovery. R9 and R10 remain incomplete.

Machine-readable evidence is
`validation/active_end_to_end_acceptance.json`; the frozen protocol is
`docs/ACTIVE_END_TO_END_PROTOCOL.md`.