# Symmetric decoded-skeleton graph-program protocol

Frozen after the 0.14 selection audit and before new confirmation worlds are generated.

## Motivation

Post-hoc audit of all frozen 0.14 candidates found zero confirmation selection regret in three mode/source cases and only 0.21 percentage points in the fourth. The candidate representation, rather than refinement selection, is therefore the next target. Existing skeleton trees learn ordered raw teacher tokens and later combine two votes, while the evaluated object is an unordered edge in the teacher's decoded graph.

## Frozen method

- Build one unordered training row for every `i < j` pair.
- The skeleton target is edge presence in the frozen teacher's decoded graph after its historical deterministic projection. It is teacher behavior, not truth.
- For each of the 20 ordered local features, expose `min(x_ij, x_ji)` and `max(x_ij, x_ji)`. These 40 named inputs are invariant when the two nodes exchange roles.
- Fit one six-split arithmetic skeleton tree with beam width three and penalty 0.001 on source extraction worlds.
- Keep the 0.13 orientation tree, two-view swap reconciliation, conflict-to-undirected rule, and deterministic acyclic projection unchanged.
- No confirmation data, truth graph, or equation metadata enters fitting or selection.
- Primary baseline: the unweighted 0.13 factorized program regenerated from the same source. Historical local trees remain secondary controls.

## Confirmation

- Seed 2793 uses frozen relational source 493; seed 2794 uses source 494.
- Both modes, 3/5/8 nodes, all five environments, 32 worlds per cell, and 96 samples per world.
- Each run has 480 unique new worlds and 960 teacher/program comparisons.
- Complete replay covers source hashes, program fitting, world regeneration, predictions, metrics, and every mode/size/environment and family stratum.

## Decision rule

Call the symmetric decoded-skeleton program a replicated behavioral improvement only if:

1. both runs pass complete replay;
2. exact teacher-graph fidelity is no lower than the factorized baseline in every seed and mode;
3. pooled exact teacher-graph fidelity improves by at least 2 percentage points;
4. pooled active-pair fidelity is no lower; and
5. all predefined strata are reported.

Passing would establish bounded behavioral graph-program improvement. It would not prove internal circuit identity, causal truth recovery, general identification, or complete neural-to-SCM decompilation.
