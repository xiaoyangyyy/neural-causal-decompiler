# Antisymmetric direction graph protocol

## Question

Version 0.22 made skeleton prediction more consistent by factorizing edge
presence from orientation, but directed-target accuracy declined. This
experiment tests whether encoding the exact node-swap symmetry of direction in
the head improves orientation without sacrificing the factorized skeleton gain.

## Frozen comparison

Both models use identical worlds, normalization, node-context representation,
shared compatible initialization, batch order, optimizer, factorized losses,
epoch budget, dev selection, decoder, and acyclic projection.

- `factorized_node_context`: version 0.22's unconstrained three-class
  conditional orientation head.
- `antisymmetric_direction`: the same head trunk and unchanged skeleton path.
  For ordered-pair trunk states `z_ij,z_ji`, a bias-free linear head maps
  `(z_ij-z_ji)/2` to `d_ij`; the directed conditional logits are
  `[d_ij,-d_ij]`. A separate linear head maps `(z_ij+z_ji)/2` to the undirected
  logit. Swapping nodes therefore exchanges the two directed logits exactly
  and preserves the undirected logit.

The candidate's compatible weights are projected from the baseline orientation
head before training: the signed direction weight is half the difference of
the two directed-class weights, and the undirected head copies the baseline's
third orientation class. Parameter counts are reported. Test labels do not
enter training, checkpoint selection, or initialization.

## Data and budgets

- Development seed: 4392.
- Confirmation seeds: 4393 and 4394.
- Nodes 3/5/8, 96 samples per world.
- Per node count: 256 train, 64 dev, and 96 worlds in each of
  ID/function/noise/scale/intervention.
- Width 48, 40 epochs, batch size 24.
- Complete world regeneration, model retraining, checkpoint replay, prediction
  replay, and metric replay are required.

## Frozen acceptance rule

Pass only if:

1. both seeds completely replay;
2. candidate exact-graph accuracy is higher in each seed and improves by at
   least 2 pooled percentage points;
3. directed-target accuracy is higher in each seed and improves by at least 2
   pooled percentage points;
4. skeleton F1 is no lower in each seed;
5. mean pair SHD is lower in each seed; and
6. no environment's pooled exact-graph accuracy declines by more than 2 points.

All cells are reported. Passing would improve a new graph teacher architecture;
it would not establish causal identifiability or historical-teacher fidelity.
