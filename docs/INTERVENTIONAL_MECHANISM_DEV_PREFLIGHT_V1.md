# Intervention-aware mechanism development preflight

This development experiment targets the failure exposed by the fixed frozen
mechanism certificate: the observationally trained neural teacher can be
wrong at allowed do values before symbolic extraction begins. The experiment
does not change the accepted confirmation seeds, historical checkpoints,
original thresholds, or the 38-claim ledger. The original ledger remains
0 proved, 2 refuted, 36 unresolved.

A first frozen protocol, validation/interventional_mechanism_dev_protocol_v1.json,
failed safely. Its loader expected a 20-feature GraphDiscoverer, while the
historical checkpoint declares active_input_observational_control_v1 and has
24 input features with factorized skeleton and orientation heads. The failure
is retained in validation/interventional_mechanism_dev_preflight_v1_failed.json.
The v2 protocol binds the checkpoint SHA-256
3e1c5c4967858ba5035fd4f897dd5872c95ee70f70131e45fe72b9cfcc2345fc
and uses the architecture-tagged loader and observational zero-padding.

The v2 development world is fixed before inspection: generator seed 8100,
split dev, three nodes, world index 0. It is distinct by identity from the
100 three-node confirmation worlds under seeds 8101 and 8102 across the five
historical test environments. A separate 96-row observational sample enters
the frozen graph teacher. The inferred DAG is saved in the candidate folder;
no true graph or equation enters the trainer.

Mechanism data are fixed at 512 observed rows for the world:

- 128 observational rows: 96 fit and 32 validation.
- For each of the three nodes, do at observed values -1 and +1:
  48 fit and 16 validation rows per value.

Every source has a distinct deterministic sample seed and row IDs. For a
target mechanism, batches that directly intervene on that target are
excluded. In the fixed world, each node has 288 usable fit rows and
96 usable validation rows. The inferred parent sets are empty for nodes
0 and 2 and {0} for node 1. The independent evaluator found that the
inferred graph happens to match this one development world's true graph;
that fact is not an input to training or a general graph-recovery result.

Candidate observations and graph prediction are under
runs/interventional_mechanism_dev_v2/candidate. The true world is under
runs/interventional_mechanism_dev_v2/truth_only. The training runner reads
only the candidate directory and checks every input hash against
validation/interventional_mechanism_dev_preflight_v2.json. A separate
read-only verifier regenerates all samples and the graph prediction; its
result is validation/interventional_mechanism_dev_preflight_v2_verification.json
with status verified-development-preflight. Probability replay fixes one
PyTorch CPU thread because different thread counts alter the last floating
digits. Four source tests passed, including target intervention exclusion,
split and budget controls, checkpoint loader compatibility, and tamper
rejection.

The frozen training plan is
validation/interventional_mechanism_dev_training_protocol_v1.json: 120 epochs,
width 48, one training thread, at most 12 hours for this development stage
and an 8 GiB artifact cap. Its no-training check passed for all three
targets; see validation/interventional_mechanism_dev_training_precheck_v1.json.
Full training has not run while the existing frozen three-node formal proof
continues as the single heavy task. The next execution will train the
three mechanisms from the inferred graph and intervention observations,
then evaluate them independently on held-out do samples and true mechanisms.
No original claim is closed by the preflight.