# End-to-end inferred-graph mechanism protocol

## Question

Version 0.10 showed that hierarchy-constrained symbolic search improves equation
extraction when the true DAG is supplied. This experiment reconnects graph
inference and asks whether that improvement survives inferred parent sets and
whether the resulting explicit SCM remains close to the oracle-graph branch.

Teacher fidelity and causal truth remain separate. The inferred graph comes
only from a frozen neural graph teacher; true graphs are used by the paired
oracle diagnostic and final evaluation, never to repair inferred graphs.

## Frozen pipeline

For each world:

1. apply the frozen graph teacher to the standard pair-statistic tensor;
2. decode its partially directed prediction and use the existing deterministic
   acyclic completion;
3. fit neural conditional mechanisms using the inferred parents;
4. distill both the historical sparse equation and the version 0.10
   hierarchy-constrained equation from identical neural queries;
5. repeat steps 3-4 with the true DAG as an explicitly labelled oracle
   diagnostic; and
6. evaluate graph error, symbolic-to-neural NMSE, symbolic-to-truth NMSE,
   intervention-effect MAE, and equation complexity.

The graph teacher, decoder, symbolic libraries, beam width, sparsity penalties,
query generation, residual model, and intervention evaluation are frozen.
No inferred edge or equation may be selected using test truth.

## Data and budgets

- Development seed 3792 with the frozen seed-91 graph teacher.
- Confirmation seed 3793 uses `runs/all_seed91_final/multivariate`; seed 3794
  uses `runs/all_seed92_final/multivariate`.
- Confirmation covers node counts 3/5/8 and ID/function/noise/scale/intervention
  environments, one new world per cell: 15 worlds per seed.
- Per world: 512 observational rows, 120 neural-mechanism epochs, 768
  distillation queries, 512 evaluation rows, maximum 8 terms, beam width 32.
- Both seeds require artifact integrity, world regeneration, graph re-inference,
  neural retraining, symbolic refitting, and metric replay.

## Frozen acceptance rule

The end-to-end hypothesis passes only if all conditions hold:

1. both seeds replay completely and cover all 15 cells;
2. within the inferred-graph branch, structured symbolic-to-neural NMSE is
   lower in each seed and its pooled relative reduction is at least 20%;
3. within the inferred branch, pooled symbolic-to-truth NMSE and intervention
   MAE increase by no more than 10% versus the sparse baseline;
4. inferred structured equations use no more nonconstant atoms on average;
5. inferred-graph structured truth NMSE and intervention MAE are each no more
   than 25% worse than the paired oracle-graph structured branch; and
6. all graph sizes and environments are reported without post-hoc filtering.

Passing would establish finite end-to-end neural-to-explicit-SCM recovery for
this benchmark. It would not prove graph identifiability, unique equations, or
universal causal decompilation.
