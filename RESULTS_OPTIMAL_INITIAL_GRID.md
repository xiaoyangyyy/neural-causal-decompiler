# Exact initial-grid synthesis improves eight nonlinear realizations

On the same eight frozen trained phase-crossing ReLU rings, choosing the
initial grid reduces the executable finite realization from 459 states to
**297 at 8D** and **324 at 32/64/128D**, at rational tolerance `17/100`.
The reductions are 35.3% and 29.4%, respectively, without changing a model,
the recurrent simulation relation, or the continuous-action domain.

| State dimension | Seeds | Initial states | Recurrent states | General minimum interval |
|---:|---|---:|---:|---|
| 8 | 6101, 6102 | 81 | 216 | [81,297] |
| 32 | 6101, 6102 | 108 | 216 | [81,324] |
| 64 | 6101, 6102 | 108 | 216 | [81,324] |
| 128 | 6101, 6102 | 108 | 216 | [81,324] |

The new result also proves the exact minimum initial coordinate grid
**within the fixed recurrent sensitivity proof class**: four observed
axes force at least 81 cells, and the sole grid below 108 is checked
exactly. At dimensions 32 and above it fails the handoff inequality on
coordinate zero; a 108-cell refinement succeeds. This inequality failure
excludes a proof candidate, rather than witnessing an actual unsafe
trajectory. The hidden feedback coordinate needs only one initial bin.

Exact certificates cover the full unit initial and action cubes and all
finite horizons. Every model's actual state/control phase-crossing
witnesses, original 81-state lower, old 459-state upper, new upper and
initial-grid optimality certificate are replayed. Runtime traces check
the initial selector, output, and transitions including action-bin edges.

These models use a structured fixed hidden feature dictionary and
learned output coefficients on synthetic data. The recurrent enclosure
remains conservative. Neither global causal-quotient minimality nor the
original end-to-end decompilation requirements are established.

Evidence: [method](docs/OPTIMAL_INITIAL_GRID_METHOD.md),
[eight-case acceptance](validation/optimal_initial_grid_acceptance.json),
[certificates](runs/optimal_initial_grid_v1), and
[previous nonlinear results](RESULTS_NONLINEAR_TWO_STAGE.md).
Replay with `python -m scripts.acceptance_optimal_initial_grid --verify`.


An exploratory follow-up extracts five exact action-phase polygons from the
actual 128D transition weights. At three initial abstract centers,
coordinatewise action boxes admit 32 candidate successor cells each,
while joint polygon clipping admits six each. Vertex evaluations replay
the original network exactly. This is evidence for a subsequent nonlinear
shared-action closure method, not a certified closed graph or a stronger
finite-machine upper. See [probe](validation/nonlinear_action_region_probe.json)
and [replay script](validation/probe_nonlinear_action_regions.py).


Release 0.56.0 passed 213 regression tests and isolated installed-wheel
replay over all eight cases. All 115 installed Python modules matched
source bytes. The checker binds the exact acceptance and JUnit file hashes
and replays original lower/two-stage proofs, new optimal-initial-grid
proofs, phase witnesses and executable traces. Wheel SHA-256:
`c21b1e0931699b099a315930eef988dfea1e93926188d85273c58368e1e0bc01`. Evidence: validation/wheel_v56_run/status.json and
validation/pytest_v56.xml.
