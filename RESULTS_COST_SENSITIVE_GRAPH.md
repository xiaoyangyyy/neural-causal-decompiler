# Cost-sensitive factorized graph programs

## Frozen question

Version 0.14 tests whether teacher-edge omissions in the 0.13 factorized program can be reduced by weighting positive skeleton tokens. The protocol was frozen in `docs/COST_SENSITIVE_FACTORIZED_GRAPH_PROTOCOL.md` before confirmation seeds 2593/2594 were generated.

Skeleton candidates used positive weights 1.0, 1.5, 2.0, and 3.0. The source refinement partition jointly selected weight and `and`/`or` aggregation using teacher exact-graph fidelity. Orientation fitting, bidirectional reconciliation, conflict handling, and acyclic projection remained unchanged. Truth did not enter fitting or selection.

## Coverage and replay

| Seed | Frozen source | New worlds | Teacher/program comparisons | Replay |
|---|---:|---:|---:|---|
| 2593 | relational 493 | 480 | 960 | passed |
| 2594 | relational 494 | 480 | 960 | passed |
| Total | two independent sources | 960 | 1,920 | passed |

Each run includes both modes, 3/5/8 nodes, five environments, 32 worlds per node/environment cell, and every family stratum.

## Teacher fidelity

| Scope | Unweighted exact graph | Cost-sensitive exact graph | Delta | Unweighted active-pair | Cost-sensitive active-pair |
|---|---:|---:|---:|---:|---:|
| Seed 2593 | 19.69% | 20.83% | +1.15 pp | 49.08% | 50.61% |
| Seed 2594 | 18.96% | 18.96% | 0.00 pp | 44.95% | 44.95% |
| Pooled | 19.32% | 19.90% | +0.57 pp | 47.14% | 47.99% |

Seed 2593 selected weight 1.0 without relations and weight 3.0 with relations. Seed 2594 selected weight 1.0 in both modes, so the selector safely retained the unweighted baseline.

## Truth diagnostic and decision

Pooled exact truth accuracy changed from 7.92% to 7.45%; frozen teacher accuracy was 14.27%. Truth is diagnostic and does not alter the teacher-fidelity selection, but the decline prevents interpreting the behavioral gain as causal recovery.

Both runs replayed, every seed/mode met exact-fidelity nondecline, and pooled active-pair fidelity did not decline. The preregistered decision nevertheless failed because the pooled exact gain was +0.57 percentage points rather than the required +2 points. Cost-sensitive skeleton fitting is therefore a negative replicated-improvement result. R9 and end-to-end neural-to-SCM recovery remain incomplete.

Machine-readable evidence: `validation/cost_sensitive_graph_acceptance.json`.
