# Research positioning and completion audit

This document separates verified engineering results from the scientific
claim the project is trying to establish. The user-level goal remains a
certified, intervention-preserving realization of a fixed neural computation,
preferably with a closed minimum-state complexity bound and evidence on
trained, nontrivial models. The original broader causal-discovery
decompilation requirements R4, R5, R8, R9, and R10 in
`FULL_REQUIREMENTS.md` remain incomplete.

Finite symbolic abstractions of continuous control systems and approximate
(bi)simulation are established ideas. Pola, Girard, and Tabuada showed
finite approximate symbolic models for bounded incrementally stable
nonlinear systems; their work explicitly discusses input quantization:
[Approximately bisimilar symbolic models for nonlinear control systems]
(https://arxiv.org/abs/0706.0246). Pola and Tabuada developed alternating
approximate bisimulation for systems with disturbances:
[Symbolic Models for Nonlinear Control Systems: Alternating Approximate
Bisimulations](https://arxiv.org/abs/0707.4205). Therefore the existence of
a quantized finite upper model is not, by itself, a novelty claim.

The project's more specific route is to treat the serialized neural network
as the ground-truth transition system, construct a directly executable
finite abstraction under declared interventions, and attach checkable upper
and lower certificates for its minimum-state complexity. Current evidence:

| System | Uniform horizon | Lower | Upper | What is verified |
|---|---|---:|---:|---|
| Frozen scalar separable ReLU | Unbounded | 7 | 9 | Exact-rational multi-switch obstruction and interval-checked executable model |
| Frozen 2D separable ReLU | Unbounded | 25 | 81 | Product upper and complete initial-output packing |
| Constructed coupled phase-crossing 2D ReLU | Unbounded | 25 | 64 | Automatic influence-grid upper and exact packing |
| Frozen trained affine ReLU, 8/32/64/128D | Unbounded | 81 | 31,104 | Exact-rational coordinate-weighted global simulation; synthetic controls |
| Trained phase-crossing ReLU, 8/32/64/128D | Unbounded | 81 | 1,375 | Automatic-grid upper on eight models; synthetic queue teacher |
| Learned local-phase ReLU, 8/32/64/128D | Unbounded | 81 | 1,125 | Automatic-grid upper on eight models; known sparse locality |

The scalar lower result is stronger than pairwise response packing because
deterministic transition closure forces multiple target overlaps during a
continuous action sweep. This is a concrete method contribution inside the
declared affine class. The project has not proved that the theorem is
unprecedented in the literature; a publication novelty claim requires a
deeper comparison with approximate automata minimization and symbolic
control lower bounds.

The next scientific gate is an externally grounded nonlinear recurrent
model, with naturally admissible controls, frozen weights, a declared state
domain, and an independently validated training/selection/test split. The checker must either prove a uniform
upper result over that domain or report unresolved. A lower result must use
network-grounded intervention witnesses or transition constraints and
exclude candidate state counts. Query efficiency should be measured against
random interventions, passive traces, and activation clustering. Exact
minimality, if claimed, requires matching independently checked lower and
upper bounds.

The generic 2D coupled verifier enumerates state and action product boxes, so
cost grows exponentially with dimension. Scaling requires compositional
certificates, local abstractions with sound interfaces, or adaptive
partitioning; merely running a higher-dimensional network without a domain
and complete proof would not meet the stated objective.


The 0.35 source, wheel, and isolated-installed modules matched byte for byte;
159 regression tests and 24 certified CLI generation/replay commands passed.
The release record is `validation/wheel_v35_run/status.json`.

## Post-0.35 trained global compositional realization

The frozen trained ReLU dynamics from the earlier continuous-scaling run now
have exact-rational, succinct, infinite-horizon simulation certificates over
the full unit state/action cubes. A coordinatewise Lipschitz-vector proof
certifies every implicit transition without enumerating an exponential table.
All four 8/32/64/128-dimensional models replay against their frozen hashes at
epsilon 0.17. A coordinate-weighted refinement certifies [81,31,104] for all four dimensions; its one-bin interior coordinates exploit sparse, contractive influence.
See RESULTS_COMPOSITIONAL_TRAINED_REALIZATION.md and
runs/compositional_scale_seed4701/summary.json and runs/weighted_compositional_scale_seed4701/summary.json.

This closes the prior "trained high-dimensional recurrent model — no global
certificate" gap for the stable affine, synthetic-control models already in
the repository. It does not establish a natural nonlinear traffic model,
efficient abstract-state complexity, matching minimality, or completion of
R4/R5/R8/R9/R10. The separate frozen scalar benchmark remains [7,9].










Release 0.36.0 is recorded in 'validation/wheel_v36_run/status.json'.
All 163 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte; five installed-package compositional generation
and replay commands passed. Wheel SHA-256:
9c42ac17aa0f8b210670b93b354d1a3302999d885e819d4d8b6e68be1755dd1e.



## Trained nonlinear global realization follow-up

Two independent training seeds across 8/32/64/128 dimensions now yield eight
frozen, phase-crossing ReLU transition models with disjoint train, selection,
and test data. Every model passes exact-rational full-unit-domain,
infinite-horizon simulation at epsilon 0.17 with minimum-state interval
[81,2250]. Both state and control phase crossings have exact nonzero
second-difference witnesses. All eight complete training and certificate
replays passed the frozen quality gates. The teacher is a synthetic
traffic-inspired ring and the feature layer is structured, so this is a
trained nonlinear network result, not an externally measured traffic
decompilation or unconstrained representation-learning result.
See RESULTS_TRAINED_NONLINEAR_GLOBAL.md and
validation/trained_nonlinear_global_acceptance.json.



Release 0.37.0 is recorded in 'validation/wheel_v37_run/status.json'.
All 165 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte. Four installed-package commands passed:
fresh nonlinear generation/replay, formal 128D nonlinear replay, and the
earlier weighted-affine replay. Wheel SHA-256:
563d7d34dfcce68210c8ee2d0fedfff08a24082b0e0476e4eea10f277b9bb805.



## Learned local phase boundaries with global certificates

A follow-up trains the two local hidden ReLU directions, thresholds, and
readouts rather than fixing a hinge dictionary. The sparse local
state/upstream/control support remains an explicit prior. Two new seeds
across 8/32/64/128 dimensions give eight frozen models with disjoint
train/selection/test data, genuine state and control nonlinearities, and
full-unit-domain unbounded-horizon certificates at epsilon 0.17. Every
minimum-state interval remains [81,2250]. Full retraining, checkpoint
selection, exact network, certificate, and metric replay pass.
A post-hoc matched-data comparison finds 1.32-1.71 times lower held-out
RMSE than the prior fixed-dictionary learner.
See RESULTS_LEARNED_LOCAL_GLOBAL.md and
validation/learned_local_global_acceptance.json.

The synthetic teacher, known locality graph, unmatched state-complexity
bounds, lack of external traffic data, and incomplete R4/R5/R8/R9/R10
remain material limitations.



Release 0.38.0 is recorded in 'validation/wheel_v38_run/status.json'.
All 167 regression tests passed. Source, wheel, and isolated-installed
Python modules matched byte for byte. Four installed-package commands
passed, including fresh local-phase generation/replay and formal 128D
full retraining/replay. Wheel SHA-256:
07db257e1306c4e7cb9ec9e704fb27ed016fb0618f753f26e88ca178b4a6a67d.



## Automatic grid synthesis from frozen network weights

A new proposal algorithm computes absolute ReLU influence matrices,
solves a contractive error-transfer relation, and greedily allocates
coordinate bins. Its floating search is untrusted; every proposed
machine is checked by the existing exact-rational full-domain verifier.
Twenty-one frozen trained and coupled networks pass complete
proposal/certificate replay. The learned-local models improve from
2250 manually supplied states to 1000-1125 automatically supplied
states; the coupled 2D model improves from 100 to 64. On the trained
affine models the automatic result is worse than the existing hand
construction (38,556-40,824 versus 31,104), so the heuristic is not
uniformly optimal. See RESULTS_AUTOMATIC_GRID_GLOBAL.md and
validation/automatic_grid_global_acceptance.json.

The method reads network weights rather than recovering support from
intervention queries, and it does not prove exact minimality or
publication-level algorithmic novelty. The synthetic-model and original
R4/R5/R8/R9/R10 gaps remain.



Release 0.39.0 is recorded in 'validation/wheel_v39_run/status.json'.
All 170 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte; five installed-package commands passed,
including fresh coupled-grid synthesis/replay and formal learned and affine
replays. Wheel SHA-256:
91a32e50d87303cb683d05f44cc576d6ed3c04128f7a5892727a5388eb958a9e.

## Version 0.40 verifier-guided interventional support

The transition support candidate now comes from behavior-only coordinate
interventions. A separate exact-rational checker proves each proposed edge
with an input-pair witness and proves each nonedge with the absence of a
nonzero-weight path. Possible paths without witnesses remain unresolved.
The 24-case study covers 20 frozen trained networks at 8/32/64/128D and
four controls. Twenty-two cases have complete support certificates:
5,191 certified edges and 105,525 certified nonedges across all cases;
the narrow-tent and path-cancellation controls each retain one unresolved
pair. See RESULTS_INTERVENTIONAL_SUPPORT.md,
docs/INTERVENTIONAL_SUPPORT_METHOD.md, and
validation/interventional_support_acceptance.json.

This establishes only coordinate-specific one-step support for frozen
networks with declared state/action coordinates. The verifier reads the
network weights; latent variables, the minimum causal quotient, external
validity, and original R4/R5/R8/R9/R10 remain open. Release 0.40.0 is
recorded in validation/wheel_v40_run/status.json: 174 tests and five
isolated-installed commands passed, with byte-identical source, wheel,
and installed modules.

## Version 0.41 exact function-level support

A bounded exact ReLU activation-region checker now addresses the two
unresolved 0.40 controls. Strict rational Fourier-Motzkin elimination
identifies feasible full-dimensional regions. A nonzero regional slope
produces an exact intervention witness; zero slopes across all regions
prove coordinate independence by continuity. The 27-case formal study
replays with 5,197 certified edges, 105,531 certified nonedges, and no
unresolved pairs. Six edge witnesses and two absence proofs require the
new region checker. Seven controls include narrow, oblique and three-input
coupled tents plus duplicate and distinct-hinge cancellations.
Evidence: RESULTS_FUNCTIONAL_SUPPORT.md,
docs/FUNCTIONAL_SUPPORT_METHOD.md, and
validation/functional_support_acceptance.json.

The behavior proposal still reads only the callable network, but the new
exact-region search reads frozen weights. This is coordinate-specific
one-step support, not general latent-variable discovery or proof of a
minimum causal quotient. Region reasoning has explicit three-input,
twelve-hidden-unit and search caps. Larger networks can still be
certified when query witnesses and structural zero paths suffice.
General large-network functional equivalence, external validity, and
original R4/R5/R8/R9/R10 remain open. Release 0.41.0 in
validation/wheel_v41_run/status.json passed 180 regression tests and
eight isolated-installed commands with byte-identical modules.

## Version 0.42 exact all-horizon behavioral quotient

For frozen ReLU networks proven globally affine and invariant on the full
state/action cube, the project now computes the exact future-response
equivalence classes. The quotient map is the rational observability row
space of (C,A), with exact induced dynamics and output checks. A cyclic
triangular proof certifies that the previously trained 8/32/64/128D
affine models have full exact quotient dimension, despite four immediate
observations. In 128D an initially invisible coordinate first changes
an output after 124 steps; the exact difference is nonzero but about
2.82e-161. A controlled counterfactual changes only effective
A[0,127] and lowers the exact quotient dimension from 128 to 4.
A separate mixed-state control has a non-coordinate one-dimensional
quotient; a constant-output control has dimension zero. A nonlinear
phase-changing control correctly remains unresolved.

The same frozen 128D model independently replays its epsilon=0.17
finite-state complexity interval [81, 40824]. Exact quotient dimension
and finite epsilon state count are different quantities. Evidence:
RESULTS_AFFINE_BEHAVIORAL_QUOTIENT.md,
docs/AFFINE_BEHAVIORAL_QUOTIENT_METHOD.md, and
validation/affine_quotient_acceptance.json. Release 0.42.0 in
validation/wheel_v42_run/status.json passed 189 regression tests and
eight isolated-installed commands with byte-identical modules.

This does not settle nonlinear all-horizon equivalence, a globally
minimum finite epsilon realization, arbitrary latent coordinate
discovery, external validity, or original R4/R5/R8/R9/R10.

## Version 0.43 invariant-slice lower bound

The frozen separable 2D ReLU benchmark now has a 28-state lower bound
at epsilon 0.101, improving the previous 25-state initial-output
packing while retaining the 81-state executable upper certificate.
The proof checks exact coordinate separability, regenerates the scalar
seven-state transition lower theorem, and constructs four disjoint
fixed-point slices with admissible constant actions. Each slice
requires seven abstract states; output separation prevents states
from being shared across slices. The exclusion applies to arbitrary
encoders, abstract outputs and deterministic action transitions for
all initial states and all continuous action words. Evidence:
RESULTS_INVARIANT_SLICE_LOWER.md,
docs/INVARIANT_SLICE_LOWER_METHOD.md, and
validation/invariant_slice_lower_acceptance.json.

This does not close the scalar 7?9 or 2D 28?81 minimum-state intervals,
nor does it automatically apply to coupled nonlinear or trained
high-dimensional systems. Release 0.43.0 in
validation/wheel_v43_run/status.json passed 192 regression tests and
five isolated-installed commands with byte-identical modules.
Original R4/R5/R8/R9/R10 remain open.
