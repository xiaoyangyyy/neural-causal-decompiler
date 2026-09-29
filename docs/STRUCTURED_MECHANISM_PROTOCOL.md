# Structured mechanism search protocol (frozen before confirmation)

## Question

Does a hierarchy-constrained symbolic search recover a shorter and more faithful
executable mechanism from the same frozen neural mechanism predictions than the
historical unconstrained greedy search?

The primary target is neural-teacher fidelity. Ground-truth equations, parents,
operators, and intervention effects are retained only for diagnostics and never
enter fitting or selection.

## Frozen methods

The baseline is `sparse_symbolic_fit` without modification. The proposed
`structured_symbolic_fit` uses the same constant, linear, square, sine, cosine,
tanh, and pair-interaction library. It enforces two grammar rules: at most one
unary operator per parent, and an interaction may enter only after both parent
main effects. A deterministic four-fold beam search minimizes mean held-out
neural NMSE plus a fixed 0.08 penalty per nonconstant term. Maximum terms are 8
and beam width is 32.

Both methods receive exactly the same query coordinates and frozen neural
predictions. They share the neural mechanism, source graph, observations,
empirical residual noise, evaluation samples, and interventions. The oracle DAG
is used in this experiment to isolate equation extraction from graph error; it
is explicitly diagnostic and does not claim end-to-end graph recovery.

## Worlds and budgets

- Development seed: 1792; its results cannot be confirmation evidence.
- Confirmation seeds: 1793 and 1794.
- Node counts: 3, 5, and 8.
- Environments: ID, function, noise, scale, and intervention.
- Two preregistered worlds per node-count/environment cell, including generator
  indices 0 and 1: 30 worlds per seed.
- Per world: 512 observational rows, 120 neural-training epochs, 768
  distillation queries, and 512 fresh evaluation rows.
- Each run must pass artifact-integrity, world-regeneration, frozen-neural,
  symbolic-refit, and metric replay checks.

## Decision rule

Call the structured search a replicated mechanism-decompilation improvement
only if all conditions hold:

1. both confirmation runs pass complete replay;
2. mean symbolic-to-neural NMSE is lower in each seed;
3. the pooled relative reduction in symbolic-to-neural NMSE is at least 20%;
4. pooled symbolic-to-truth NMSE and intervention-effect MAE each increase by
   no more than 10%; and
5. the structured equations use no more nonconstant atoms on average.

Parent, operator, and interaction exactness are diagnostic. Passing this rule
would improve R10 equation extraction under a supplied oracle graph; it would
not complete R9 graph recovery, prove identifiability, or establish a general
causal theorem.