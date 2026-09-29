# Cost-sensitive factorized graph-program protocol

Frozen after the 0.13 failure audit and before new confirmation worlds are generated.

## Motivation and hypothesis

The 0.13 confirmation is used only to define the next hypothesis. Factorization improved orientation agreement, while exact skeleton agreement remained about 39% and false absences greatly outnumbered false presences. The next experiment tests whether a fixed, source-selected positive-class weight for the skeleton rule can reduce teacher-edge omissions enough to improve exact graph fidelity.

## Frozen method

- Keep the 0.13 orientation tree, ordered-view swap reconciliation, conflict-to-undirected rule, and deterministic acyclic projection unchanged.
- Fit skeleton candidates on source extraction teacher labels with positive-class weights `1.0`, `1.5`, `2.0`, and `3.0`; absent tokens retain weight `1.0`.
- Use the same six splits, beam width three, penalty 0.001, arithmetic expressions, and 20 features.
- Jointly select skeleton weight and `and`/`or` aggregation by exact teacher-graph fidelity on source refinement worlds, breaking ties by all-pair fidelity, then lower weight, then `and`.
- No truth graph, equation metadata, 0.13 confirmation prediction, or new confirmation world enters fitting or selection.
- Baseline is the corresponding frozen 0.13 unweighted factorized program regenerated from the same source. The historical local tree remains a reported secondary comparator.

## Sources and confirmation

- Seed 2593 uses frozen relational source 493; seed 2594 uses source 494.
- Each run uses both modes, 3/5/8 nodes, all five environments, 32 worlds per cell, and 96 samples per world.
- This gives 480 unique new worlds and 960 teacher/program comparisons per run.
- Full replay must verify source hashes, candidate fitting and selection, world regeneration, predictions, programs, metrics, and every mode/size/environment and family stratum.

## Decision rule

Call cost-sensitive factorization a replicated behavioral-decompilation improvement only if:

1. both runs pass complete replay;
2. exact teacher-graph fidelity is no lower than the unweighted factorized baseline in each seed and each mode;
3. pooled exact teacher-graph fidelity improves by at least 2 percentage points over that baseline;
4. pooled active-pair fidelity is no lower; and
5. all predefined strata are reported.

Passing would establish a bounded behavioral improvement for these frozen teachers. It would not establish internal circuit identity, truth recovery, general causal identification, or complete neural-to-SCM decompilation.
