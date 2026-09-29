# Results: node-context graph teacher

## Outcome

The frozen rule failed. Explicit node context raised exact-graph accuracy in
both confirmation seeds and improved the pooled rate from 11.15% to 12.29%,
but the 1.15-point gain was below the required 3 points. Mean SHD improved in
seed 3993 and worsened in 3994; pooled skeleton F1 and directed-target accuracy
also declined. R9 remains incomplete.

Each seed used 2,400 isolated worlds across 3/5/8 nodes and five test
environments. Baseline and candidate shared worlds, labels, base-layer
initialization, batch order, optimizer, and checkpoint rule. Complete model
retraining and exact checkpoint replay passed.

## Aggregate metrics

| Seed | Model | Exact graph | Mean SHD | Skeleton F1 | Directed accuracy | Parameters |
|---:|---|---:|---:|---:|---:|---:|
| 3993 | baseline | 11.74% | 3.5486 | 0.7578 | 0.4757 | 15,412 |
| 3993 | node context | 12.71% | 3.1951 | 0.7446 | 0.4467 | 31,636 |
| 3994 | baseline | 10.56% | 3.0264 | 0.7630 | 0.4648 | 15,412 |
| 3994 | node context | 11.88% | 3.3771 | 0.7450 | 0.4839 | 31,636 |

Pooled exact-graph changes by environment were +3.47 points for ID, -1.22 for
function, +4.17 for noise, -1.74 for scale, and +1.04 for intervention. The
frozen OOD decline bound passed, but the principal magnitude and structural
quality safeguards did not.

## Interpretation

Aggregating incident edge embeddings before global attention supplies useful
graph-level information, especially in ID and noise-shift worlds. The added
capacity also trades skeleton quality against exact decisions and remains
seed-sensitive. This is a new architecture result; it does not show that the
historical teacher encoded the same node-context computation.

Machine-readable evidence is `validation/node_context_graph_acceptance.json`;
the frozen design is `docs/NODE_CONTEXT_GRAPH_PROTOCOL.md`.
