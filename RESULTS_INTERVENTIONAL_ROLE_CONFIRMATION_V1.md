# Independent-world check of the inferred-role mechanism hybrid

The frozen rule, inferred indegree 0 -> control checkpoint and positive
indegree -> mixed-intervention checkpoint, was evaluated on all 300 declared
independent worlds (seeds 8301/8302; 3/5/8 nodes; five environment types;
ten worlds per cell). The independent checker regenerated worlds and batches,
reloaded the graph teacher and frozen mechanisms, checked checkpoint and
parent-set hashes, and recomputed local and paired-oracle-noise rollout
metrics. The complete cohort was independently verified and packaged.

| Outcome | Count |
|---|---:|
| Verified worlds | 300/300 |
| Hybrid local normalized MSE <= 0.01 | 0/300 |
| Mixed-only local normalized MSE <= 0.01 | 0/300 |
| Control-only local normalized MSE <= 0.01 | 1/300 |
| Inferred graph exactly correct | 45/300 |
| Hybrid local failure despite correct graph | 45/45 |

Exact graph recovery was 31/100 at three nodes, 12/100 at five nodes,
and 2/100 at eight nodes. The nearest hybrid maximum local MSE to the
0.01 gate was 0.012698886. The graph errors and mechanism errors are
reported separately; the correct-graph failures show that graph recovery
alone would not make this fixed mechanism rule pass.

The initial independent pass accepted 280 worlds and stopped on an approximately
1.5e-8 CPU floating-point contrast difference in the next eight-node world.
The failed handoff and finalization receipts are preserved. The verifier was
then changed *after the results were seen* to allow rel_tol=1e-6 and
abs_tol=1e-7 for floating metric replay, while continuing exact source,
model, data and identity checks and explicitly rejecting any change in the
0.01 local-gate classification. The complete 300-world replay then passed.
This is a disclosed verifier repair, not a new prospective confirmation run.

The rollout uses the true exogenous noise as an oracle diagnostic. It does
not demonstrate recovery of a noise distribution or the full intervention
distribution. The frozen rule is one candidate under the stated sample,
training and architecture budget; its failure does not refute every neural
mechanism or symbolic program. The world-level 99% intervals in the analysis
artifact are exploratory because that analysis was fixed after initial
confirmation worlds were inspected. No prospective 99% success guarantee
is claimed. The original R0-R13 ledger remains open: 0 proved, 2 refuted,
36 unresolved.

Protocol and methods: docs/INTERVENTIONAL_ROLE_CONFIRMATION_V1.md and
validation/interventional_role_confirmation_protocol_v1.json. The complete
verified summary is validation/interventional_role_confirmation_verified_v1.json;
world-level analysis is validation/interventional_role_confirmation_analysis_v1.json.
The 300-world portable archive is
validation/interventional_role_confirmation_package_v1.tar.gz, with separate
manifest and verification receipts. It contains the frozen protocol and all
300 raw unit records, including checkpoints and independent receipts.

A separate posthoc, candidate-data-only OLS diagnostic on the first ten
three-node test-id worlds passed the local gate in one world. On the
graph-correct linear-Gaussian world at index 0, its maximum local MSE was
0.007660 versus 0.038614 for the frozen neural hybrid. This diagnostic
was chosen after confirmation failures were seen, so it cannot count as a
new confirmation method. See
validation/interventional_role_ols_diagnostic_v1.json.

Two separate exact proofs show that frozen one-parent Tanh mechanisms in
worlds 0 and 5 cannot be approximated within 1% training scale by any real
linear combination in the implemented six-function unary program library.
The verified normalized grid lower bounds are 3.3300% on [-2,2] and
3.2513% on [-1,1], respectively. World 5 has an incorrect inferred graph;
its certificate concerns frozen-network-to-program fidelity only. These
finite-grammar obstructions leave the broader original R4/R10 atoms
unresolved. See docs/FROZEN_UNARY_GRAMMAR_OBSTRUCTION_V1.md.