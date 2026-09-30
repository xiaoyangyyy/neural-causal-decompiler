# Accepted three-node proof and paired-do development result

The frozen three-node learned mechanism system has a verified
deterministic program on the full box `[-1,1]^3`. An independently
installed verifier replayed the portable proof. For all eight
compatible node-intervention masks and every do value in the box,
the joint pointwise L1 error is at most **0.0050772390287866075**
under ideal real arithmetic with the exact saved weights. The result
uses the accepted child piecewise program and two root constants;
it does not include exogenous noise or claim a true causal SCM.
Evidence:
`runs/frozen_three_node_composition_v1/` and
`validation/frozen_three_node_acceptance_v1.json`.

A separate evaluator read the fixed true world only after the candidate
bundle and acceptance receipt were sealed. In that world the true
graph happens to equal the learned graph, but each root true
structural function is zero and the frozen neural root is strictly
separated from zero. The independently replayed two-target
incompatibility margins are at least **0.0212100190** (root 1) and
**0.0240793564** (root 2). Thus no scalar program can be within
1% of the saved training scale of *both* the frozen neural root and
the true root mechanism on this fixed box. This is a strict
fixed-instance statement, not a theorem about all networks or
possible programs. Evidence:
`runs/frozen_three_node_truth_gap_v1/` and
`validation/frozen_truth_gap_acceptance_v1.json`.

After those proof stages, the queued development comparison trained
one mechanism set on a mixed observation/intervention arm and one on
an observational control. Both used the same inferred graph and
512 held-out exogenous draws across nine intervention conditions.
The independent replay found:

| Metric across executed nodes | Observation control | Mixed arm |
|---|---:|---:|
| Maximum normalized MAE | 0.302995909 | 0.114460013 |
| Maximum normalized MSE | 0.091806521 | 0.013101094 |
| Maximum normalized paired-contrast absolute error | 0.385489573 | 0.004702626 |

The mixed arm improves all three reported maxima on this development
world, but **0.013101094 > 0.01**. The nine intervention conditions
reuse the same 512 exogenous draws and are not nine independent worlds.
The frozen discovery-data standard deviations are the common
normalizers. Independent confirmation worlds: **zero**. The result
is diagnostic, not a statistically certified general recovery claim.
Evidence:
`runs/interventional_mechanism_dev_comparison_v1.json`,
`validation/interventional_mechanism_dev_handoff_v1.json`, and
`validation/interventional_mechanism_dev_result_verification_v1.json`.

The proof stages and development run respected their 8 GiB
descendant memory caps, two-thread training budget, and stage time
limits. Source tests, installed-wheel replay and acceptance gates
passed. The original ledger remains **0 proved, 2 refuted,
36 unresolved**; the full R0–R13 causal-decompilation objective
is not complete.
