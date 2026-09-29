# Factorized skeleton-orientation program protocol

Frozen before confirmation seeds are generated.

## Hypothesis

A single four-class local tree is dominated by the abundant no-edge class and
mixes two different decisions. Fit two executable teacher-supervised trees:

1. a skeleton tree predicts absent versus present for every directed pair token;
2. an orientation tree is fit only on teacher-present tokens and predicts
   `i->j`, `j->i`, or undirected.

At execution, evaluate both ordered views of each unordered pair. Convert the
reverse orientation through the fixed swap map. Skeleton aggregation (`and` or
`or`) is selected on source refinement worlds only. Matching orientation votes
are used directly; conflicting votes become undirected. The resulting directed
edges pass through the existing deterministic acyclic projection. No truth graph
or equation metadata enters fitting or selection.

## Frozen extraction and controls

- Source A/B: relational runs 493/494, respectively.
- Features: the original 20 local graph features.
- Both trees use six splits, beam width three, penalty 0.001, and arithmetic
  compositions, matching the local-tree budget.
- Training uses source extraction teacher labels. `and` versus `or` aggregation
  is chosen by exact teacher-graph fidelity on source refinement worlds, then
  all choices are frozen.
- Baseline: each source's existing `programs/<mode>/final.json` local tree.
- Teacher: each source's frozen graph network and historical argmax/acyclic
  decoding. Truth remains diagnostic.

## Confirmation

- Seed 2393 uses source 493; seed 2394 uses source 494.
- Modes: with and without relational attention.
- Nodes: 3, 5, 8.
- Environments: ID, function, noise, scale, intervention.
- 32 new worlds per size/environment cell, shared across modes: 480 unique
  worlds and 960 teacher/program comparisons per run; 96 samples per world.
- Both runs require source hashing, world regeneration, prediction/program
  replay, and metric replay.

## Decision rule

Call factorization a replicated behavioral-decompilation improvement only if:

1. both runs pass complete replay;
2. exact teacher-graph fidelity is no lower than the local tree in each seed;
3. pooled exact teacher-graph fidelity improves by at least 2 percentage points;
4. pooled active-pair fidelity is no lower; and
5. all mode/size/environment and family strata are reported.

Passing would improve behavioral graph-program extraction. It would not prove
internal circuit identity, truth recovery, general causal identification, or
complete neural-to-SCM decompilation.