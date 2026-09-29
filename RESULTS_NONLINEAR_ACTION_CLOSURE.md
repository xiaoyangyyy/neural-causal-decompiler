# Eight certified nonlinear networks compile to standalone finite programs

The same eight frozen trained phase-crossing ReLU networks now have
standalone certified realizations with **132 / 166 / 162 / 163 states**
at dimensions **8 / 32 / 64 / 128**, respectively, for both training seeds.
Tolerance remains exactly `17/100`. The source weights and recurrent
simulation relation are unchanged.

| Dimension | Seeds | Initial | Recurrent | Previous upper | New general minimum interval |
|---:|---|---:|---:|---:|---|
| 8 | 6101, 6102 | 81 | 51 | 297 | [81,132] |
| 32 | 6101, 6102 | 108 | 58 | 324 | [81,166] |
| 64 | 6101, 6102 | 108 | 54 | 324 | [81,162] |
| 128 | 6101, 6102 | 108 | 55 | 324 | [81,163] |

Exact action geometry replaces independent coordinate action boxes with
five phase polygons extracted from the actual learned network. Joint
polygon/cell intersection checks every initial source and closes the
entire recurrent graph. The independent general lower remains 81.
For seed 6101 at 128D, the checker examines 2,936 initial candidate edges
and retains 617; it examines 1,256 recurrent candidates and retains 311.
All 55 retained recurrent states are also initial successors, and their
successors stay within the same set.

Each portable program uses **one shared action template**, discrete state
rows and exact source offsets. Execution takes only the program JSON;
it needs no source network or training data. Export verification rebuilds
all templates and rows from the certified network. Dedicated tests compare
exact execution with the original finite machine and prohibit calls to
network execution or compilation helpers during standalone execution.

The certificates cover the full unit initial and continuous action cubes
and all finite horizons. The graph remains conservative at closed-cell
boundaries and because it covers more actions than the quantized runtime
alphabet. These are structured shallow networks trained on synthetic
data. Neither global causal minimality, a globally shortest program nor
the original end-to-end neural-to-SCM requirements is established.

Evidence: [method](docs/NONLINEAR_ACTION_CLOSURE_METHOD.md),
[eight-case acceptance](validation/nonlinear_action_closure_acceptance.json),
[certificates and programs](runs/nonlinear_action_closure_v1), and
[previous results](RESULTS_OPTIMAL_INITIAL_GRID.md).
Replay: `python -m scripts.acceptance_nonlinear_action_closure --verify`.


An exploratory next step checks exact state-phase extrema for the same
128D seed-6101 network. It finds that an 81-cell initial grid can satisfy
all 128 handoff inequalities with exact minimum slack about 0.00790395,
although the earlier absolute-weight sensitivity predicate excluded it.
This probe is not integrated into an accepted graph/program certificate;
the bounds in the table remain the authoritative results. Evidence:
[phase-aware probe](validation/phase_handoff_probe.json) and
[next proof obligations](docs/PHASE_HANDOFF_PLAN.md).


Release 0.57.0 passed 216 regression tests and isolated installed-wheel
replay of all eight complete graph certificates and standalone programs.
All 116 installed Python modules matched source bytes. Exact phase
witnesses, original lower/initial proofs, joint action graphs, factored
program templates and installed standalone traces passed. The status
binds the acceptance, program and JUnit hashes. Evidence:
validation/wheel_v57_run/status.json and validation/pytest_v57.xml.
Wheel SHA-256: `cdf993c5775dd5058aa0a29aebb85b6baa4a28cd9b4de8ee3cfb6b6a48eadacb`.


The exploratory 81-cell handoff above is now certified for all eight
models in the [0.58 results](RESULTS_PHASE_INITIAL_HANDOFF.md). That stage
also adds an independent whole-machine lower 82. This document retains
the original 0.57 checkpoint results.
