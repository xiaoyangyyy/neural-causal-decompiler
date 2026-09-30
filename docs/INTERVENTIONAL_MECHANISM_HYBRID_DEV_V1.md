# Posthoc structural-role hybrid on one development world

The independently replayed three-node comparison showed that
intervention-trained mechanisms helped the node with inferred
parents, while the observational control estimated the two
inferred roots more accurately on this same world. After inspecting
those results, we defined the rule:

    inferred indegree = 0  -> observational-control checkpoint
    inferred indegree > 0  -> mixed-intervention checkpoint

The graph comes from the pre-existing candidate
`graph_prediction.json`; its `oracle_graph_used` flag is false.
For this world the selected arms are [control, mixed, control] for
nodes [0,1,2]. No new checkpoint is trained. The same frozen
512 exogenous draws and nine intervention conditions are used for
all comparisons.

| Maximum over executed local mechanisms | Observation control | Mixed arm | Role hybrid |
|---|---:|---:|---:|
| Normalized MAE | 0.302995909 | 0.114460013 | 0.076496728 |
| Normalized MSE | 0.091806521 | 0.013101094 | **0.005851749** |
| Paired-contrast normalized absolute error | 0.385489573 | 0.004702626 | 0.004702626 |

The role hybrid passes the *local development-world* 0.01 maximum
normalized MSE gate. This is a **posthoc same-world selection**:
the rule was chosen after seeing this world's metrics and has zero
independent confirmation worlds. It is not evidence that the
rule generalizes, and no full SCM rollout or intervention outcome
distribution was evaluated. The metrics compare local mechanism
predictions at the true-world observed parent values; they do not
include error propagation through a recovered SCM.

The generator
`validation/evaluate_interventional_mechanism_hybrid_dev_v1.py`
first chooses model arms from the inferred graph, then composes
the already independently checked source metrics. The certificate
is `validation/interventional_mechanism_hybrid_dev_v1.json`.
A separate checker,
`validation/check_interventional_mechanism_hybrid_dev_v1.py`,
reopens the frozen mixed and control checkpoints, regenerates the
held-out exogenous draws, independently recomputes all nine
conditions and contrasts, and rejects inflated confirmation or
closure flags. Its sealed receipt is
`validation/interventional_mechanism_hybrid_dev_verification_v1.json`.
Six focused tests pass.

Before further confirmation use, freeze this role rule and the
same normalizer/threshold on new worlds without tuning on their
outcomes. The original R10 mechanism, noise and intervention
distribution claims remain unresolved.
