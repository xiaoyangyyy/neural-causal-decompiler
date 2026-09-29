# Learned local ReLU phases with certified global realization

The previous nonlinear release fixed a dictionary of hinge directions and
knot locations, then trained only the output layer. This study makes both
local hidden ReLU units per state coordinate trainable: each unit learns
four input-direction weights (local state, upstream state, demand, signal),
a bias that determines its phase boundary, and an output readout. Direct
linear weights and the output bias are learned jointly. The sparse locality
pattern and the identity observation of the first four coordinates remain
structural priors. There are 17d trainable parameters for d state
coordinates. The teacher remains the same synthetic queue ring; this is
not unconstrained representation learning or measured traffic recovery.

Two new data seeds (7201, 7202) and four dimensions (8,32,64,128) are
fixed in 'scripts/acceptance_learned_local_global.py'. For each case,
2,048 training, 512 selection, and 1,024 test state-action samples are
disjoint. Full-batch Adam runs for 600 epochs with a selection-only best
checkpoint. The test split and 64 independent ten-step intervention
rollouts are evaluated after checkpoint choice. The protocol requires
actual hidden-direction and hidden-bias movement, both state and control
nonlinear phase witnesses, held-out quality gates, and a frozen-network
global certificate. These gates were written before the formal seeds ran.

The exact-rational verifier reads the final serialized ReLU weights, not
the queue simulator. It proves the complete unit state/action cube maps
into the unit state cube; the relation covers every initial state; and
every implicit abstract transition preserves the relation for every
continuous action. The observation error is at most 0.17 at every time,
including arbitrarily long action sequences. The abstract state bins are
(6,5,5,5,1,...,1,3), so the finite upper model has 2,250 states
independent of the four tested dimensions. Exact initial-output packing
separately proves at least 81 states. Exact second-difference witnesses
show that the frozen transition is non-affine along both a state and a
control direction. The minimum state count is not closed.

| Seed | Dimensions | Test RMSE range | Largest test error | Largest 10-step rollout error | Certified interval |
|---|---|---:|---:|---:|---:|
| 7201 | 8/32/64/128 | 4.37e-5 to 5.30e-5 | 3.92e-4 | 4.63e-4 | [81,2250] each |
| 7202 | 8/32/64/128 | 4.60e-5 to 5.49e-5 | 3.63e-4 | 3.76e-4 | [81,2250] each |

All eight formal gates pass. Mean hidden-direction movement is 0.0509
to 0.0583; mean hidden-bias movement is 0.0413 to 0.0435. The
smallest exact coordinatewise transition-closure margin exceeds 0.0044.
Every state-phase second difference exceeds 0.0017 in magnitude and
every control-phase second difference exceeds 0.00085 in magnitude.
Evidence is in 'runs/learned_local_global_v1/summary.json' and
'validation/learned_local_global_acceptance.json'.

A matched-data comparator reruns the previous fixed-dictionary learner
on the same train, selection, and test inputs. Its held-out RMSE is
1.32 to 1.71 times that of the learned-hidden model across all eight
cases. This is a post-hoc descriptive diagnostic, not a frozen success
gate. It regenerates from 'validation/learned_local_dictionary_comparator.json'.

The result shows that learning local phase boundaries can improve
held-out fidelity while retaining a complete unbounded-horizon
certificate. It does not identify arbitrary causal variables, remove the
known local influence pattern, establish a new universal compression
theorem, prove exact minimality, or validate an external traffic system.
The broader R4/R5/R8/R9/R10 goals remain incomplete.



Release 0.38.0 is recorded in 'validation/wheel_v38_run/status.json'.
All 167 regression tests passed. Source, wheel, and isolated-installed
Python modules matched byte for byte. Four installed-package commands
passed, including fresh local-phase generation/replay and formal 128D
full retraining/replay. Wheel SHA-256:
07db257e1306c4e7cb9ec9e704fb27ed016fb0618f753f26e88ca178b4a6a67d.

