# Certified infinite-horizon continuous realization

Version 0.32 converts the previous finite-horizon response cover into an
executable finite-state model for a frozen coordinate-separable ReLU system.
The system has continuous state and action cubes, with neural transition
`x' = 0.5x + 0.4a`, identity observation, and epsilon 0.101.

| State/action dimension | Initial-output packing lower bound | Certified executable states | Horizon |
|---:|---:|---:|---|
| 1 | 5 | 10 | Unbounded |
| 2 | 25 | 100 | Unbounded |

The verifier replayed 20 axis initial-cell checks, 100 axis action-segment
transition checks, and 310 packing-pair checks across both profiles. In each
transition check, interval propagation through the actual ReLU network proves
that the epsilon relation is preserved for every concrete state in the
current relation and every continuous action in the segment. Exact network
matching permits the axis proofs to compose. Initial coverage plus this
forward invariance proves epsilon observation error for every run length.

The largest outward-rounded initial error bound is 0.05000000000000161; the largest one-step invariant bound is 0.10050000000000117, strictly below epsilon 0.101. The formal generation runtime was 0.140 seconds. The accepted run is
`runs/certified_continuous_realization_seed10701`; independent verification
and acceptance are recorded in
`validation/certified_continuous_realization_acceptance.json`. The method
is in `docs/CONTINUOUS_CLOSED_REALIZATION_METHOD.md`.

The lower and upper state counts do not coincide. This establishes an
approximate transition-closed realization and its size interval, not a
minimum-state realization. The proof applies to this frozen separable
benchmark. It does not establish a result for general coupled networks,
globally phase-crossing networks, or learned large-scale systems.


Version 0.32.0 release verification: 142 regression tests and 16
isolated-wheel certified generation/replay commands passed. Source, wheel,
and isolated-installed Python modules matched byte for byte. Wheel SHA-256:
0909a44c4ffffe3314403ae681c9a891633190e2d7859fb046e19552d3275168.
Evidence: validation/wheel_v32_run/status.json.
Version 0.34 follow-up: an exact-rational transition-aware lower certificate
rules out five states for the one-dimensional benchmark. Its current
infinite-horizon size interval is therefore 6–10; see
RESULTS_TRANSITION_AWARE_LOWER.md and
validation/certified_transition_lower_acceptance.json.
Version 0.35 follow-up: a stronger exact-rational multi-switch theorem
excludes six states as well, and a nine-center executable model improves
the upper bound to nine. The current scalar interval is 7–9; see
RESULTS_MULTISWITCH_LOWER.md, RESULTS_SHIFTED_REALIZATION.md, and
validation/certified_shifted_realization_acceptance.json.