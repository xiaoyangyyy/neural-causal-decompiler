# Pair-consistent factorized training protocol

## Question

Version 0.23 showed that a hard antisymmetric direction architecture is
seed-dependent. The deployed factorized model already symmetrizes decision
scores across `(i,j)` and `(j,i)`, while version 0.22 trains its component loss
on unsymmetrized ordered-pair logits. This experiment tests whether applying
the same pair consistency during training improves direction without removing
representational capacity.

## Frozen comparison

Both branches use the identical version 0.22 factorized node-context model,
parameter count, worlds, normalization, initialization, batch order, optimizer,
loss weights, epoch budget, dev checkpoint rule, inference symmetrization,
decoder, and acyclic projection.

- `raw_factorized`: skeleton BCE and conditional orientation CE act on the raw
  ordered-pair component logits, as in version 0.22.
- `pair_consistent`: before the same losses, skeleton logits are averaged over
  `(i,j),(j,i)`, while orientation logits are averaged after swapping the two
  directed labels and preserving the undirected label.

Only the training loss input differs. Test labels do not enter training or
selection.

## Data and budgets

- Development seed: 4592.
- Confirmation seeds: 4593 and 4594.
- Nodes 3/5/8, 96 samples per world.
- Per node count: 256 train, 64 dev, and 96 worlds in each of
  ID/function/noise/scale/intervention.
- Width 48, 40 epochs, batch size 24.
- Complete world regeneration, model retraining, checkpoint replay, prediction
  replay, and metric replay are required.

## Frozen acceptance rule

Pass only if:

1. both seeds completely replay;
2. pair-consistent exact-graph accuracy is higher in each seed and improves by
   at least 2 pooled percentage points;
3. directed-target accuracy is higher in each seed and improves by at least 2
   pooled percentage points;
4. skeleton F1 is no lower in each seed;
5. mean pair SHD is lower in each seed; and
6. no environment's pooled exact-graph accuracy declines by more than 2 points.

All cells are reported. Passing would improve a new graph teacher architecture;
it would not establish causal identifiability or historical-teacher fidelity.
