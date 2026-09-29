# Symmetric decoded-skeleton graph programs

Version 0.15 replaces ordered-token skeleton supervision with unordered pair rows, 40 swap-invariant min/max features, and labels from the frozen teacher's final decoded skeleton. Orientation and projection remain identical to version 0.13. The protocol was frozen before seeds 2793/2794.

Both runs covered 480 new worlds, 960 teacher/program comparisons, both modes, 3/5/8 nodes, five environments, and all family strata. Both complete replays regenerated source supervision, programs, worlds, predictions, and metrics.

| Scope | Factorized exact | Symmetric exact | Delta | Factorized active-pair | Symmetric active-pair |
|---|---:|---:|---:|---:|---:|
| Seed 2793 | 20.00% | 20.00% | 0.00 pp | 48.21% | 48.78% |
| Seed 2794 | 18.33% | 18.23% | -0.10 pp | 45.79% | 46.69% |
| Pooled | 19.17% | 19.11% | -0.05 pp | 47.07% | 47.78% |

Pooled exact truth accuracy changed from 9.53% to 9.17%; teacher truth accuracy was 12.55%. The preregistered rule failed because exact fidelity declined in one seed/mode and pooled exact fidelity did not improve by two percentage points.

The result shows that swap-invariant decoded-skeleton supervision improves some active pairs but still does not compose into better exact graphs. This rejects another local pair-tree explanation of the frozen teachers. It does not establish graph recovery, internal circuit identity, or end-to-end neural-to-SCM decompilation. R9 remains incomplete.

Machine-readable evidence: `validation/symmetric_skeleton_acceptance.json`.
