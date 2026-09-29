# Invariant-slice lower bound for approximate causal realization

The frozen scalar ReLU benchmark has a seven-state exact-rational
transition lower bound at epsilon=0.101. Its two-dimensional separable
product previously had only a 25-state initial-output packing lower
bound and an 81-state executable upper bound. This experiment lifts
the scalar transition obstruction through four disjoint controlled
invariant slices. The certified interval is now

    28 <= K_N(0.101, all continuous action words) <= 81.

The checker extracts the actual rational affine maps from the serialized
one- and two-dimensional ReLU systems. It proves the 2D map is exactly
two independent copies of the scalar map, with identity observation and
full unit-domain invariance. It then regenerates the scalar seven-state
multi-switch certificate. Four second-coordinate fixed points have
pairwise output gaps greater than twice epsilon. Their exact fixed
actions are 0, 1/3, 2/3, and 1, all admissible.

For each fixed slice, all runs restricted to that second-coordinate
action implement the original scalar problem in the first coordinate.
Any realization needs at least seven abstract states reachable from
that slice. States reachable from different slices cannot be shared:
one abstract output cannot lie within epsilon of two distinct fixed
second-coordinate values separated by more than 2 epsilon. Therefore
every deterministic finite realization, regardless of encoder,
representative locations, or action-dependent transition table, has
at least 4*7=28 states.

The formal acceptance independently replays the historical scalar
lower certificate and the nine-by-nine 2D upper realization, then
generates and replays the new exact-rational slice certificate.
Evidence: `runs/invariant_slice_lower_v1/summary.json` and
`validation/invariant_slice_lower_acceptance.json`.
The theorem and scope are in `docs/INVARIANT_SLICE_LOWER_METHOD.md`.

| Bound | Previous | Current |
|---|---:|---:|
| Scalar minimum-state interval | 7–9 | 7–9 |
| Two-dimensional lower | 25 | 28 |
| Two-dimensional upper | 81 | 81 |
| Two-dimensional interval | 25–81 | 28–81 |

This is a strict improvement to a global minimum-state lower bound,
not proof that 28 is attainable or optimal. It relies on exact
coordinate separability and invariant fixed-point slices, so it does
not transfer automatically to the coupled nonlinear benchmark or
trained 128D models. The scalar eight-state question remains open;
numerical interval searches are not an impossibility certificate.

Release 0.43.0 is recorded in `validation/wheel_v43_run/status.json`.
All 192 regression tests passed. Source, wheel, and isolated-installed
modules matched byte for byte; five installed-package commands passed,
including fresh slice certification and historical lower/upper replays.
Wheel SHA-256:
`4908fed8528b78bffa0eb5f6863b9eee93b7c9fad79c8d1a6baaa6f066d6823a`.
