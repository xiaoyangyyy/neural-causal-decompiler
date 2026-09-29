# Teacher-distilled PC program protocol (frozen before confirmation)

## Question

Can a fully executable PC-style program, whose only hyperparameters are selected
to match a frozen neural graph teacher, reproduce complete teacher graphs more
faithfully than the historical local four-class decision tree?

The extracted program executes conditional-independence queries, records
separating sets, orients unshielded colliders, and applies deterministic Meek
R1/R2/R3 closure. It contains no graph labels or equation metadata at runtime.

## Frozen extraction

For every teacher mode and node count, evaluate the Cartesian candidates:

- Fisher partial-correlation alpha in
  `{0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2}`;
- maximum conditioning-set size in `{0, 1, 2}` capped by graph size.

Use the source run's extraction and refinement worlds only. Candidate score is
exact-graph fidelity to the frozen teacher, followed by all-pair fidelity,
lower maximum conditioning order, and alpha closest to 0.01. Ground-truth graphs
must not enter selection. The baseline is the source run's frozen
`programs/<mode>/final.json`, executed with its original local features and
acyclic projection.

Every PC execution must persist its CI tests, separating sets, collider-oriented
intermediate PDAG, and final Meek-closed PDAG. The saved program consists of its
alpha, maximum conditioning order, feature/test semantics, and deterministic
orientation algorithm.

## Sources, seeds, and budgets

- Historical development sources: relational seeds 493 and 494.
- Confirmation A: source 493, new seed 2193.
- Confirmation B: source 494, new seed 2194.
- Modes: `without_relations` and `with_relations`.
- Node counts: 3, 5, and 8.
- Environments: ID, function, noise, scale, and intervention.
- 32 new worlds per size/environment cell, shared across modes: 480 unique
  worlds and 960 teacher/program comparisons per confirmation run.
- Samples per world: 96.
- Source models, source programs, selection datasets, generated worlds,
  predictions, traces, and metrics require hash or deterministic replay.

## Decision rule

Call the PC program a replicated behavioral decompilation improvement only if:

1. both confirmation runs pass complete replay;
2. mean exact-graph fidelity to the teacher is no lower than the local-tree
   baseline in each seed;
3. pooled exact-graph fidelity improves by at least 2 percentage points;
4. pooled active-pair fidelity is no lower; and
5. all mode/size/environment and linear-Gaussian/nonlinear strata are reported.

Teacher fidelity is primary. Truth accuracy is diagnostic. Passing would show
that the frozen networks' graph behavior is better summarized by an explicit
PC-style program than by the existing local tree. It would not prove that the
network internally implements PC, that Fisher CI is valid for every nonlinear
world, or that graph/SCM recovery is generally solved.