# Results: end-to-end inferred-graph mechanism recovery

## Outcome

The frozen end-to-end rule failed. Hierarchy-constrained equations remained
shorter and improved pooled inferred-graph neural fidelity by 18.7%, but the
20% threshold was missed and the improvement did not replicate in seed 3794.
Graph error dominated the final SCM gap: inferred-graph structured truth NMSE
was 2.32 times the oracle-graph branch, and intervention-effect MAE was 3.11
times the oracle branch. R10 therefore remains incomplete end to end, with R9
now directly measured as the limiting stage.

Both seeds covered 15 new worlds: 3/5/8 nodes crossed with ID, function, noise,
scale, and intervention environments. Graph re-inference, neural conditional
mechanism retraining, symbolic refitting, and metric replay all passed.

## Aggregate results

| Seed | Branch | Method | Neural NMSE | Truth NMSE | Intervention MAE | Atoms |
|---:|---|---|---:|---:|---:|---:|
| 3793 | inferred | sparse | 0.008947 | 0.492462 | 0.704494 | 2.555 |
| 3793 | inferred | structured | 0.004690 | 0.491803 | 0.134674 | 1.152 |
| 3793 | oracle | structured | 0.004698 | 0.249674 | 0.037549 | 0.907 |
| 3794 | inferred | sparse | 0.005067 | 0.450254 | 0.286728 | 2.735 |
| 3794 | inferred | structured | 0.006704 | 0.448291 | 0.177298 | 1.258 |
| 3794 | oracle | structured | 0.006557 | 0.155270 | 0.062748 | 1.021 |

The frozen graph teachers achieved exact-DAG accuracy 0% in seed 3793 and
13.3% in seed 3794, with mean adjacency SHD 4.47 and 4.20. The inferred
structured branch passed the truth/intervention non-degradation safeguard and
used fewer atoms, but failed replicated neural improvement and oracle-gap
requirements.

## Interpretation

The structured symbolic search remains useful after graph inference: it sharply
reduces extrapolation-sensitive intervention error and equation length. It
cannot compensate for wrong parent sets. The paired design provides direct
end-to-end evidence that improving equation extraction alone will not complete
the neural-to-SCM objective.

Machine-readable evidence is
`validation/end_to_end_mechanism_acceptance.json`; the frozen protocol is
`docs/END_TO_END_MECHANISM_PROTOCOL.md`.
