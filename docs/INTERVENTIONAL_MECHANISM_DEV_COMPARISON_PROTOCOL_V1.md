# Same-world observational control and paired do evaluation

The frozen intervention-aware development preflight is described in
[INTERVENTIONAL_MECHANISM_DEV_PREFLIGHT_V1.md](INTERVENTIONAL_MECHANISM_DEV_PREFLIGHT_V1.md).
This comparison tests whether genuine node interventions help recover
mechanisms on one already selected development world. It is a method check,
not an independent confirmation or a proof of the original R0–R13 claims.

The observational control was frozen in
validation/interventional_mechanism_dev_control_protocol_v1.json before its
data were generated. It uses the same 96 discovery observations and inferred
DAG as the intervention arm. Its mechanism data contain 384 fit and 128
validation observations, 512 observed rows total, with deterministic
source-specific seeds and row IDs. All 512 rows are usable for each target.
The intervention arm also collects 512 rows, but directly intervened targets
are excluded from their own mechanism fit: each target has 288 fit and 96
validation usable rows. This gives the control more usable rows per target
while keeping the collection budget equal. The candidate-facing bundles
contain no true graph or true equations. The true world is used by the data
generator and independent evaluator only.

The read-only control verifier regenerates the observational matrices,
checks source and candidate hashes, exact zero row overlap with the
intervention arm, graph/discovery equality, and the batch schedule. Its
receipt is
validation/interventional_mechanism_dev_control_verification_v1.json with
status `verified-development-control`. An isolated-copy tamper test
confirms that a modified observation archive is rejected.

Both training plans are frozen before training:
validation/interventional_mechanism_dev_training_protocol_v1.json and
validation/interventional_mechanism_dev_control_training_protocol_v1.json.
They bind the same trainer source, width 48, seed 8100, 120 epochs, one
training thread, a 12-hour stage limit, and an 8 GiB artifact limit. The
control dry-run precheck passed for all three target mechanisms; the receipt
is validation/interventional_mechanism_dev_control_training_precheck_v1.json.

The evaluation plan in
validation/interventional_mechanism_dev_evaluation_protocol_v1.json binds
both training plan hashes, the truth world and discovery data hashes, the
evaluation code hash, and the original normalized MSE threshold 0.01.
The evaluator holds out 512 exogenous draws from a distinct deterministic
source and reuses each draw across nine conditions: observation, each of
three single-node interventions at -1 and +1, and two two-node
combinations. It checks train/evaluation row overlap, reports true graph
agreement only from the evaluator, skips structural equations cut by a
direct do operation, and records per-node normalized MAE/MSE and paired
two-do contrasts. The strict comparison rule is the reduction in the
maximum executed normalized MAE. Synthetic unit tests verify zero error
for exact mechanisms, exclusion of cut equations, and a two-do error for
a wrong mechanism.

Neither training arm has run yet because the existing frozen three-node
formal proof occupies the single heavy-task slot. No performance result,
independent-world statistical guarantee, or original claim closure follows
from these preflights. The original claim ledger remains 0 proved, 2
refuted, 36 unresolved.

A durable handoff is available in
validation/advance_interventional_mechanism_dev_v1.py. It waits on the
confirmed live formal-chain process handle, verifies a terminal chain
receipt, reruns both candidate-data replays, and launches the two training
arms sequentially under Windows Job memory and time limits. It then executes
the frozen paired-do evaluator and records hashes and resources for every
stage. It never treats a stale status file as proof that the formal process
has stopped. The handoff has a no-training precheck and source tests; its
result remains development-only even if all stages succeed.

The shared normalization is the per-node standard deviation of the frozen
96-row discovery sample, clipped below at 0.05. It is fixed before either
training arm runs and is identical between arms. It is not the separately
learned `target_training_scale` of either arm, so the 0.01 comparison here
is a development diagnostic under this declared scale; passing it alone
would not satisfy an original claim whose denominator is the frozen
mechanism's own training scale.

An independent read-only verifier is
validation/verify_interventional_mechanism_dev_evaluation_v1.py. It checks
both training source and checkpoint hashes, reruns both frozen candidate-data
verifiers, regenerates the held-out exogenous draws, and recomputes every
per-node mechanism error and paired do contrast without calling the result
generator. It compares the complete result tree and rejects changed numeric
metrics, intervention coverage or scope flags. Eight targeted development
tests currently pass across the verifier, evaluator, data control and handoff.
The end-to-end replay cannot run until the queued training and evaluation
produce their result files.

A second lightweight handoff waits for the development comparison process to
finish and then runs the independent result verifier under the same 8 GiB Job
limit and a 300-second replay limit. It stores a separate verification receipt
and refuses a changed handoff, checker source, or result hash. The proof job,
the development training, and independent result replay therefore remain
sequential. The receipt will remain pending until real trained checkpoints
and the evaluation result exist.

The independent arithmetic was also prechecked on the actual frozen
three-node development world: with 512 common exogenous draws, all nine
conditions from the result generator and the separate verifier agree under
the declared numerical replay tolerance when using fixed zero predictors.
This checks the evaluation semantics before training; it is not a result
for learned mechanisms.
