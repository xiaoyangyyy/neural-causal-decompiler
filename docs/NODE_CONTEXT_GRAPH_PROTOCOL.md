# Node-context graph teacher protocol

## Question

The version 0.20 end-to-end audit identified inferred parent sets as the dominant
SCM error source. Earlier relational attention only added relation-specific
attention biases, while its edge encoder remained local. This experiment tests
a stronger permutation-equivariant architecture that explicitly constructs node
context before classifying each edge.

## Frozen comparison

The baseline is the existing `GraphDiscoverer`. The candidate uses the same
pair-statistic front end and pair encoder, then computes for every node the mean
of its outgoing and incoming off-diagonal edge embeddings. Each directed edge
receives its local embedding, source incoming/outgoing summaries, target
incoming/outgoing summaries, and a global off-diagonal mean. A shared MLP
combines these six blocks before the existing global attention and symmetric
four-class head.

Both models use identical worlds, labels, normalization, class-weighted loss,
optimizer, batching, epoch budget, dev cross-entropy checkpoint rule, decoder,
and acyclic projection. The candidate has additional context-MLP parameters;
parameter counts are reported. This is a new teacher architecture comparison,
not a claim that the historical teacher contained this computation.

## Data and budgets

- Development seed: 3992.
- Confirmation seeds: 3993 and 3994.
- Nodes: 3, 5, 8; 96 samples per world.
- Per node count: 256 train worlds, 64 dev worlds, and 96 worlds in each of
  ID/function/noise/scale/intervention test environments.
- Width 48, 40 epochs, batch size 24.
- Both confirmation runs require world regeneration, full model retraining,
  checkpoint/prediction replay, and metric replay.

## Frozen acceptance rule

The node-context hypothesis passes only if:

1. both seeds completely replay all models, worlds, predictions, and metrics;
2. candidate pooled exact-graph accuracy is higher in each seed;
3. pooled across both seeds, exact-graph accuracy improves by at least 3
   percentage points;
4. mean pair SHD is lower in each seed;
5. pooled skeleton F1 and directed-target accuracy are no lower; and
6. within each of the five environments, pooled exact-graph accuracy does not
   decline by more than 2 percentage points.

All node sizes and environments are reported. Passing would improve the graph
teacher used by future end-to-end recovery; it would not decompile historical
teachers or prove causal identifiability.
