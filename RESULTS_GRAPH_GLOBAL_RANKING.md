# Graph-global sparse ranking programs

Version 0.16 tests an explicit graph-level program: a sparse L1 edge scorer, an affine per-world edge-count predictor, deterministic joint top-k selection, the frozen orientation tree, and score-prioritized acyclic projection. The protocol was frozen before confirmation seeds 2993/2994.

Both runs covered 480 new worlds and 960 teacher/program comparisons across both modes, 3/5/8 nodes, five environments, and all family strata. Both complete replays regenerated fitting, nine-candidate refinement selection, programs, worlds, predictions, and metrics.

| Scope | Factorized exact | Graph-global exact | Delta | Factorized active-pair | Graph-global active-pair |
|---|---:|---:|---:|---:|---:|
| Seed 2993 | 17.92% | 15.10% | -2.81 pp | 46.79% | 47.83% |
| Seed 2994 | 15.83% | 14.06% | -1.77 pp | 47.46% | 46.49% |
| Pooled | 16.88% | 14.58% | -2.29 pp | 47.11% | 47.18% |

Pooled exact truth accuracy fell from 9.06% to 5.52%; frozen teacher truth accuracy was 13.07%. The selected program exceeded 20 edge coefficients in at least one mode in both runs, so the complexity gate also failed.

The preregistered rule failed. A world-level edge budget makes decisions jointly, but it propagates count and ranking errors across the graph and materially reduces exact graph recovery. This rejects the tested graph-global top-k explanation of the frozen teachers. It does not prove that all graph-global programs fail, nor does it establish internal circuit identity, causal truth recovery, or complete neural-to-SCM decompilation. R9 remains incomplete.

Machine-readable evidence: `validation/graph_global_acceptance.json`.
