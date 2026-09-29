# Active end-to-end SCM recovery protocol

## Question

Version 0.25 strongly improved interventional graph recovery. This experiment
asks whether that graph gain improves the final explicit SCM, including
mechanism truth and intervention effects, relative to an equal-parameter
observational graph teacher and a paired oracle-DAG diagnostic.

## Frozen pipeline

For each world, both frozen teachers from the same version 0.25 run receive the
same observation. The control receives the 20 observational graph features plus
four zeros. The active teacher receives the same base features plus two-level
intervention response summaries for every source node. Each predicted partial
graph uses the same decoder and deterministic DAG completion.

For observational, active, and oracle graphs, the pipeline then trains neural
conditional mechanisms, fits both sparse and hierarchy-constrained symbolic
equations from identical neural query budgets, and evaluates neural fidelity,
mechanism truth, intervention-effect error, and equation complexity. Test truth
never repairs a predicted graph or selects an equation.

## Data and budgets

- Development seed 4992 with `runs/active_intervention_seed4792_quick`.
- Confirmation seed 4993 uses `runs/active_intervention_seed4793`; seed 4994
  uses `runs/active_intervention_seed4794`.
- Nodes 3/5/8 and ID/function/noise/scale/intervention, one new world per cell:
  15 worlds per seed.
- Per world: 96 graph-discovery samples, two intervention levels per node, 512
  mechanism observations, 120 mechanism epochs, 768 distillation queries, 512
  evaluation rows, maximum 8 terms, beam width 32.
- Both seeds require artifact integrity, world and active-feature regeneration,
  graph re-inference, neural retraining, symbolic refitting, and metric replay.

## Frozen acceptance rule

Pass only if:

1. both seeds replay completely and cover all 15 cells;
2. active graph exact accuracy is higher in each seed, gains at least 20 pooled
   percentage points, and active graph SHD is lower in each seed;
3. active structured symbolic-to-truth NMSE is lower than the observational
   branch in each seed with at least 20% pooled relative reduction;
4. active structured intervention-effect MAE is lower in each seed with at
   least 20% pooled relative reduction;
5. within the active branch, structured symbolic-to-neural NMSE is lower than
   the sparse baseline in each seed with at least 15% pooled relative reduction,
   and uses no more nonconstant atoms;
6. active structured truth NMSE is at most 1.75 times the oracle structured
   value, and active intervention MAE is at most twice the oracle value; and
7. every graph size and environment is reported without filtering.

Passing would establish finite end-to-end interventional neural-to-explicit-SCM
recovery on this benchmark. It would not establish purely observational
identifiability, unique equations, or universal causal decompilation.
