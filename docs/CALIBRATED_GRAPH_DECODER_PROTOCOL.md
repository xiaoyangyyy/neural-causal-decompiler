# Calibrated graph decoder protocol (frozen before confirmation)

## Question

Can a development-only edge-presence threshold improve multivariate neural graph
recovery over the historical per-pair argmax decoder, without changing the
frozen graph network?

This is an R9 neural graph-inference experiment. It is separate from symbolic
program fidelity and from the oracle-graph mechanism extraction in version
0.10.

## Frozen decoder

For each unordered node pair, average the two swap-equivalent probability
vectors. Predict no edge when `1 - p(absent)` is below a threshold; otherwise
choose the largest directed/undirected class. Apply the existing deterministic
acyclic projection to directed edges and preserve explicit undirected edges.

Choose one threshold per node count from the source run's development split on
the fixed grid 0.100, 0.125, ..., 0.900. Maximize exact target-graph accuracy,
then minimize mean pair SHD, then prefer the threshold closest to 0.5 and the
smaller threshold. Development truth may select the decoder because this is a
neural graph model component. No confirmation world may affect calibration.

## Sources, seeds, and budgets

- Development evidence: historical relational seeds 493/494; test results from
  those runs are diagnostic and cannot be confirmation evidence.
- Confirmation A: frozen relational source 493 and new world seed 1993.
- Confirmation B: frozen relational source 494 and new world seed 1994.
- Evaluate both `without_relations` and `with_relations` teachers.
- Node counts: 3, 5, and 8.
- Environments: ID, function, noise, scale, and intervention.
- 32 new worlds per node-count/environment cell, shared by both teacher modes:
  480 unique worlds per confirmation run.
- Sample count remains 96. Source models, development features, and all new
  worlds/predictions must pass hash or deterministic regeneration replay.

## Decision rule

Call calibration a replicated graph-decoding improvement only if:

1. both confirmation runs pass complete replay;
2. mean exact target-graph accuracy across both modes, all sizes, and all five
   environments is no lower than argmax in each seed;
3. pooled exact target-graph accuracy improves by at least 2 percentage points;
4. mean pair SHD is no higher in either seed; and
5. all mode/size/environment strata are reported, including regressions.

Pair accuracy, skeleton F1, direction accuracy, and family-specific results are
diagnostics. Passing would improve frozen-network graph decoding only. It would
not establish symbolic decompilation fidelity, end-to-end SCM recovery, causal
identifiability, or a universal theorem.