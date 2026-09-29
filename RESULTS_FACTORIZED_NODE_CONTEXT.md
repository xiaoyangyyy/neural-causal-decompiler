# Results: factorized node-context graph teacher

## Outcome

The frozen rule failed, although the factorized head produced the most
consistent graph-structure improvement so far. Exact-graph accuracy increased
in both seeds, mean SHD decreased in both, pooled skeleton F1 improved, and no
environment lost exact accuracy. The pooled exact gain was 1.35 percentage
points rather than the required 2 points, seed 4194 had a small skeleton-F1
decline, and directed-target accuracy decreased. R9 remains incomplete.

Each seed used 2,400 shared worlds. The two equal-parameter models shared the
node-context representation initialization, batch order, optimizer, epochs,
and checkpoint rule. Complete retraining and zero-tolerance checkpoint replay
passed.

## Aggregate metrics

| Seed | Head | Exact graph | Mean SHD | Skeleton F1 | Directed accuracy |
|---:|---|---:|---:|---:|---:|
| 4193 | flat | 13.13% | 3.7146 | 0.7553 | 0.5258 |
| 4193 | factorized | 15.76% | 3.2319 | 0.7761 | 0.4860 |
| 4194 | flat | 16.94% | 3.0910 | 0.7805 | 0.4952 |
| 4194 | factorized | 17.01% | 2.6382 | 0.7726 | 0.4633 |

Pooled exact accuracy rose from 15.03% to 16.39%; skeleton F1 rose from
0.7679 to 0.7744; SHD fell from 3.4028 to 2.9351. Exact-graph changes were
positive in every environment: +1.74 points ID, +0.69 function, +2.78 noise,
+1.39 scale, and +0.17 intervention. Pooled directed-target accuracy fell from
0.5105 to 0.4747.

## Interpretation

Separating edge presence from conditional orientation reduces structural
errors more reliably than a flat four-class objective. The remaining bottleneck
is now specifically orientation, rather than skeleton selection or graph
sparsity. Future work should retain the factorized skeleton head and add
direction-specific structural supervision without using held-out truth.

Machine-readable evidence is
`validation/factorized_node_context_acceptance.json`; the frozen protocol is
`docs/FACTORIZED_NODE_CONTEXT_PROTOCOL.md`.
