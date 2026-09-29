# Active intervention graph protocol

## Question

Versions 0.21-0.24 changed graph heads and training symmetry but did not produce
stable direction gains. The multivariate SCM interface already supports paired
surgical interventions, yet graph teachers have only consumed observational
samples. This experiment tests whether explicit active response evidence closes
part of the R9 direction gap.

## Frozen comparison

Both branches use the same factorized node-context architecture, parameter
count, worlds, initialization, normalization procedure, batch order, optimizer,
loss, epoch budget, dev selection, decoder, and acyclic projection. Both inputs
have 24 channels.

- `observational_padded`: the existing 20 graph features plus four zero channels.
- `active_intervention`: the same 20 features plus four ordered response
  summaries. For each source node, interventions set its observed value to the
  observational mean plus or minus one observational standard deviation while
  preserving the world's exogenous draw. For every ordered source-target pair,
  features record normalized signed contrast, contrast RMS, and mean absolute
  changes from observation at the two intervention levels.

The features use samples and known intervention targets/values only. They do
not read graph or equation metadata. Effects on descendants can be nonzero, so
the existing conditional-dependence features remain necessary to distinguish
direct from indirect relations. This is an interventional graph-discovery
experiment, not a claim about purely observational identifiability.

## Data and budgets

- Development seed: 4792.
- Confirmation seeds: 4793 and 4794.
- Nodes 3/5/8, 96 samples per world.
- Per node count: 256 train, 64 dev, and 96 worlds in each of
  ID/function/noise/scale/intervention.
- Width 48, 40 epochs, batch size 24.
- Complete world regeneration, observational/interventional feature
  regeneration, model retraining, checkpoint replay, prediction replay, and
  metric replay are required.

## Frozen acceptance rule

Pass only if:

1. both seeds completely replay;
2. active exact-graph accuracy is higher in each seed and improves by at least
   10 pooled percentage points;
3. directed-target accuracy is higher in each seed and improves by at least 10
   pooled percentage points;
4. skeleton F1 is no lower in each seed;
5. mean pair SHD is lower in each seed; and
6. no environment's pooled exact-graph accuracy declines.

All cells are reported. Passing would establish a strong interventional R9
result on the benchmark; it would not establish observational identifiability,
historical-teacher fidelity, or general causal discovery.
