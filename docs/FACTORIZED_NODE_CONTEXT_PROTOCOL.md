# Factorized node-context graph protocol

## Question

Version 0.21 improved exact-graph accuracy with explicit node context but traded
away skeleton and direction quality. The historical four-class edge loss couples
edge presence and orientation. This experiment tests whether factorizing those
decisions preserves the node-context exact-graph gain while improving structural
quality.

## Frozen comparison

Both models use the version 0.21 node-context representation, identical worlds,
normalization, shared representation initialization, batch order, optimizer,
epoch budget, dev selection, probability symmetrization, decoder, and acyclic
projection.

- `flat_node_context`: the existing four-class head and weighted cross entropy.
- `factorized_node_context`: a binary skeleton logit plus a three-class
  orientation head trained only on present-edge pairs. The loss is the sum of
  class-balanced skeleton BCE and class-balanced conditional orientation cross
  entropy. Four-class decision scores are reconstructed so the best edge class beats
  no-edge exactly when the skeleton log-odds is positive; its identity is the
  conditional orientation argmax. Softmax scores are retained for dev
  checkpointing, while training uses the two factorized losses.

The shared head trunk and total final output width remain the same; parameter
counts are reported. No truth labels from test worlds enter selection.

## Data and budgets

- Development seed: 4192.
- Confirmation seeds: 4193 and 4194.
- Nodes 3/5/8, 96 samples per world.
- Per node count: 256 train, 64 dev, and 96 worlds in each of
  ID/function/noise/scale/intervention.
- Width 48, 40 epochs, batch size 24.
- Complete world regeneration, model retraining, checkpoint replay, prediction
  replay, and metric replay are required.

## Frozen acceptance rule

Pass only if:

1. both seeds completely replay;
2. factorized exact-graph accuracy is higher in each seed and improves by at
   least 2 pooled percentage points;
3. skeleton F1 is no lower in each seed;
4. pooled directed-target accuracy is no lower;
5. mean pair SHD is lower in each seed; and
6. no environment's pooled exact-graph accuracy declines by more than 2 points.

All cells are reported without subset selection. Passing would improve a new
graph teacher architecture; it would not decompile historical teachers or prove
causal identifiability.
