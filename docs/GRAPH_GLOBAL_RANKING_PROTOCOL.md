# Graph-global sparse ranking program protocol

Frozen after source-only graph-density audit and before confirmation seeds are generated.

## Hypothesis

Versions 0.13-0.15 repeatedly improved pair-level fidelity without improving exact graphs. A graph-level program must make coupled edge decisions. Frozen teachers show stable average degree across node sizes but substantial world-to-world edge-count variation. The tested program therefore predicts a world-specific edge budget and ranks all unordered pairs jointly.

## Executable program

1. Convert each unordered pair to the 40 swap-invariant min/max features defined in version 0.15.
2. Standardize with extraction-only means and scales.
3. Compute an explicit sparse logistic edge score. Fit L1 logistic candidates with `C` in `{0.01, 0.03, 0.1}` to frozen teacher decoded-skeleton labels.
4. Summarize each world's pair scores by node count, mean, standard deviation, minimum, maximum, and 25/50/75 percentiles.
5. Predict the teacher decoded edge count using an explicit ridge affine program with alpha in `{0.1, 1.0, 10.0}`; round and clip it to `[0, n(n-1)/2]`.
6. Select exactly the predicted number of highest-scoring unordered pairs with deterministic lexicographic tie breaking.
7. Reuse the version 0.13 orientation tree, two-view swap reconciliation, and conflict-to-undirected rule. Deterministic acyclic projection prioritizes directed edges by their saved sparse edge score; lexicographic order breaks exact score ties.

All fitted means, scales, coefficients, intercepts, hyperparameters, feature names, and tie-breaking semantics are serialized. No neural model call is allowed during program execution.

## Fitting and selection

- Edge scorer and count regressor fit only source extraction teacher behavior.
- Select the nine `(C, alpha)` candidates by exact teacher-graph fidelity on source refinement worlds, then active-pair fidelity, then fewer nonzero coefficients, then smaller `C`, then larger alpha.
- Teacher labels use the frozen historical decoder. Truth graphs and equation metadata never enter fitting or selection.
- Primary baseline is the regenerated version 0.13 factorized program; version 0.15 symmetric tree and historical local tree are secondary diagnostics.

## Confirmation

- Seed 2993 uses frozen relational source 493; seed 2994 uses source 494.
- Both modes, 3/5/8 nodes, five environments, 32 worlds per cell, and 96 samples per world.
- Each run contains 480 unique new worlds and 960 teacher/program comparisons.
- Complete replay must regenerate fitting, selection, worlds, predictions, metrics, and every mode/size/environment and family stratum.

## Decision rule

Call graph-global sparse ranking a replicated behavioral improvement only if:

1. both runs pass complete replay;
2. exact teacher-graph fidelity is no lower than the factorized baseline in every seed and mode;
3. pooled exact teacher-graph fidelity improves by at least 2 percentage points;
4. pooled active-pair fidelity is no lower;
5. the selected program has at most 20 nonzero edge coefficients and eight count coefficients; and
6. all predefined strata are reported.

Passing would establish bounded behavioral graph-program improvement. It would not prove internal circuit identity, causal truth recovery, general identification, or complete neural-to-SCM decompilation.

