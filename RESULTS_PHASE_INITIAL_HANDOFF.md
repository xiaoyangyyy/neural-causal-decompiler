# Minimum initial labels and a strictly larger whole-machine lower

Release 0.58 adds two results on the same eight frozen phase-crossing
trained ReLU networks. Exact signed phase extrema certify 81 initialization
labels at every dimension, and packing proves that count minimal. A new
continuum-coverage/common-successor theorem excludes every deterministic
81-state machine with a state-only output map, improving the independent
whole-machine lower from 81 to 82. Tolerance is exactly 17/100.

| Dimension | Seeds | Initial labels (minimum) | Recurrent states | Previous upper | New whole-machine interval |
|---:|---|---:|---:|---:|---|
| 8 | 6101, 6102 | 81 | 51 | 132 | [82,132] |
| 32 | 6101, 6102 | 81 | 58 | 166 | [82,139] |
| 64 | 6101, 6102 | 81 | 54 | 162 | [82,135] |
| 128 | 6101, 6102 | 81 | 55 | 163 | [82,136] |

Neural weights, training data and the recurrent simulation relation have
not changed. The old absolute-weight handoff lost signed cancellations;
exact phase polygon extrema recover enough slack for the 81-cell initial
grid. Seed 6101 at 128D checks 156 state pair-cells and has minimum
handoff slack about 0.007903950876. Every new complete graph and exported
program has one shared normalized action template, plus state offsets.
Programs execute using their JSON without the source neural network.

The new lower is a property of arbitrary deterministic state-only-output
machines, rather than a grid search minimum. If only 81 states existed,
initial continuum coverage would force decoder coordinates into three
narrow bands. Two concrete initial states must use the same label, yet
under one common action their exact next observations demand a successor
decoder in the gap between those bands. All eight frozen models have an
exact witness. This separates initialization complexity from transition
complexity without claiming the whole-machine minimum is already known.

The initial proof requires shallow state/action-separable networks with
at most two state support coordinates per output. The lower proof does
not require that transition architecture, but the tested family remains
structured shallow networks trained on synthetic traffic-inspired data.
Certificates interpret serialized binary64 coefficients as exact rational
weights and cover the mathematical function, not extra inference rounding.
Upper certificates cover full initial/action cubes and every finite
horizon. The lower already follows from times zero and one.

Global state/code minimality, general deep-network/intervention families,
uniqueness and original R4/R5/R8/R9/R10 remain open. This release is a
verified advance toward the full objective, not its completion.

Evidence:
[initial proof method](docs/PHASE_INITIAL_HANDOFF_METHOD.md),
[whole-machine lower proof](docs/TRANSITION_CONSISTENCY_LOWER_METHOD.md),
[initial/upper acceptance](validation/phase_initial_handoff_acceptance.json),
[lower acceptance](validation/transition_consistency_lower_acceptance.json),
[portable programs](runs/phase_initial_handoff_v1), and
[lower certificates](runs/transition_consistency_lower_v1).

Replay:

```powershell
python -m scripts.acceptance_phase_initial_handoff --verify
python -m scripts.acceptance_transition_consistency_lower --verify
python -m pytest --junitxml=validation/pytest_v58.xml
```

The phase-initial acceptance retains [81,U] with its original packing
lower. The separate lower acceptance binds those exact same models and
programs to the stronger [82,U] result; this preserves prior evidence.


Release 0.58.0 passed **221 regression tests**, with zero failures, errors
or skips. Isolated installed-wheel replay passed all eight cases,
including original lower/initial proofs, legacy graphs/programs, exact
phase-aware handoffs, minimum initial labels, the new transition-consistency
lower certificates, complete new graphs and standalone execution.
All **118 installed Python modules matched source bytes**. The status
binds both acceptance files, the JUnit report, program/lower certificate
hashes and the wheel. Evidence:
[wheel replay](validation/wheel_v58_run/status.json),
[complete tests](validation/pytest_v58.xml), and
[checker](validation/check_wheel_v58.py).
Wheel SHA-256: `c4f4118b7e8fed1e9e7d4f7c7ca6433e03efe54113093d5dc6b3050d2b0f1908`.

The next implementation target is the
[joint mixed-state/control phase proof](docs/NEXT_MIXED_RELU_PLAN.md).
The original full objective remains incomplete.
