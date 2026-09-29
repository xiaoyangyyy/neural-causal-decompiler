# Transition-aware lower certificate for continuous realization complexity

The previous scalar infinite-horizon result certified a 10-state executable
model and a five-point response packing, leaving the interval 5–10.
Version 0.34 adds a lower certificate that uses deterministic transition
closure, not just pairwise behavioral separation. It excludes every model
with one through five states and narrows the interval to

    6 <= K_N(0.101, all continuous action words) <= 10.

For sizes one through four, the combined output tolerance intervals cannot
cover the complete initial unit interval. At five states, output tolerance
limits the total overlap between state-output intervals to about 0.01. Yet
some state's concrete hull must have width at least 0.2. Under the neural
transition `x_next=0.5x+0.4a`, its image has width at least 0.1 and slides
across the action continuum. A deterministic switch between target states
would require those targets to overlap by at least 0.1, contradicting the
0.01 overlap budget.

The verifier proves the network is affine on the declared full input cube,
extracts its effective coefficients and identity observation from the actual
serialized ReLU weights, then checks every inequality with exact rational
arithmetic. It makes no assumption about the candidate encoder, output
locations, or abstract transition table. The proof and code are described
in `docs/TRANSITION_AWARE_LOWER_BOUND_METHOD.md`.

The lower run is `runs/certified_transition_lower_seed12701`. The joint
acceptance record independently replays both that run and the existing
10-state upper realization:
`validation/certified_transition_lower_acceptance.json`.

This is a genuine strengthening of the lower bound for the declared scalar
system, but it does not identify the exact minimum. The theorem requires an
affine scalar transition and identity output on the full unit interval.
It has not yet been generalized to the coupled nonlinear benchmark or a
trained high-dimensional model.


Version 0.35 follow-up: a stronger exact-rational multi-switch theorem
excludes six states as well, and a nine-center executable model improves
the upper bound to nine. The current scalar interval is 7–9; see
RESULTS_MULTISWITCH_LOWER.md, RESULTS_SHIFTED_REALIZATION.md, and
validation/certified_shifted_realization_acceptance.json.