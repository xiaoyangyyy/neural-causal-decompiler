# Frozen independent-world confirmation of the inferred-role hybrid

The development-world rule was selected after inspecting that world's mixed
and control mechanism errors. This protocol freezes the rule before new world
outcomes are inspected: an inferred root uses the observational checkpoint;
a node with inferred parents uses the mixed-intervention checkpoint. The rule
reads only the candidate graph. It does not read true graph/equation metadata.

The new cohort uses generator seeds 8301 and 8302. Each seed has 10 independent
worlds for each of 3, 5, and 8 nodes in each of the five historical test
environments, for 300 world units. The preflight checks the 300 identities
are unique and disjoint from historical 8100/8101/8102 worlds. The old
confirmation runs, checkpoints, failures, and 0.01 gate remain intact. The
protocol file fixes the graph teacher and all execution source hashes.

For each world, 96 observational rows go to the frozen graph teacher. Both
mechanism arms get 512 fresh training observations. The mixed arm divides
the budget between observation and all single-node do(-1)/do(+1) conditions;
the observational control has 384 fit and 128 validation rows. Every arm
trains 120 epochs at width 48, using the same inferred parent sets. The
role rule picks among the completed frozen checkpoints for that world.
The evaluator uses 512 held-out exogenous draws and observation, all
single-node do conditions, and two combined do conditions. Repeated do
outcomes using the same exogenous draw are paired measurements, not
independent worlds.

Two different outcomes are recorded. Local mechanism error evaluates each
checkpoint at true-world parent values. The full rollout simulates the
inferred DAG recursively. The rollout injects the true exogenous noise,
so it isolates graph/mechanism error; it is not a recovered-noise or
intervention-distribution guarantee. Graph mismatch and all
intervention-condition errors are retained. An independent checker
regenerates world/candidate observations, replays the frozen graph teacher,
checks checkpoint hashes and inferred parent sets, then independently
recomputes the metrics. Before that replay, a unit is labelled
computed-pending-independent-replay.

Each world runs alone under a Windows job object with one Torch training
thread, a 600-second unit timeout, and an 8 GiB memory limit. A stage is
bounded by 12 hours and 8 GiB of new artifacts. Completed units are
immutable and can be skipped on resume; an incomplete unit requires an
audit, not silent restart.

The frozen gate is maximum local normalized MSE <= 0.01 over every executed
node/condition. A passing world is only one independent-world observation.
A failed world refutes this specific frozen hybrid rule's universal
success over the declared cohort; it does not refute existence of another
candidate or close R0-R13. The original proof ledger remains open.

A deferred handoff watches the live stage process handle. Only after the
stage reports all 300 computed units does it launch the independent full
replay under the same 8 GiB job limit. It writes a separate status receipt;
if computation or replay fails, the status remains unresolved.

After the independent all-300 summary and handoff succeed, the portable
packager collects the frozen protocol and every unit file into a deterministic
tar.gz archive with per-file SHA-256 manifest. A separate read-only package
verifier checks every member and rejects missing, altered, duplicate or unsafe
paths. The package gate intentionally fails while the run is incomplete.
