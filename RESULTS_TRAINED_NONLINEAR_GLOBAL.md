# Trained nonlinear recurrent realization with control phase crossings

This experiment addresses the gap left by the frozen trained affine models.
It trains an actual ReLU transition network on a traffic-inspired ring of
queue states. The two normalized, continuously admissible controls are
demand and signal. Congestion depends on an upstream-minus-local queue
threshold; an additional hinge depends on demand-minus-signal. Both
nonlinearities are active inside the declared full unit state/action cubes.

The simulator is synthetic, not a measured traffic system. Its hinge
thresholds (0.075 and 0.125) are absent from the frozen network's fixed
dictionary (-0.1, 0, 0.1, 0.2). The network learns its output-layer
coefficients by ridge regression. Its hidden feature layer is structured
and fixed; this experiment does not show that unconstrained representation
learning can discover the causal variables or phase boundaries.

## Frozen data and model protocol

The formal script 'scripts/acceptance_trained_nonlinear_global.py' fixes
two independent seeds (6101, 6102), four dimensions (8, 32, 64, 128),
2,048 training samples, 512 model-selection samples, and 1,024 untouched
one-step test samples per case. It explicitly checks that the three input
splits have no common rows. Three ridge penalties are compared using only
selection error; the chosen network is then evaluated once on test data.
Sixty-four disjoint intervention rollouts of length 10 test multistep
fidelity. The formal acceptance thresholds, architecture, candidate
abstraction, and data-seed protocol are frozen in that script.

The frozen ReLU network, not the simulator formula, is the ground-truth
system for certification. Its serialized binary64 weights are interpreted
as exact rationals. The verifier checks the image of the complete unit
state/action cube, coordinatewise global ReLU sensitivity, initial-cell
coverage, and inductive closure of every implicit abstract transition.
The executable abstraction uses bins (6,5,5,5,1,...,1,3), relation radii
(0.17,0.17,0.17,0.17,1.01,...,1.01,0.4), and 128 action bins per control
coordinate. Every model receives an unbounded-horizon output-error
certificate at epsilon 0.17. An 81-point initial-output packing gives the
separate lower bound. Both state and control nonlinearity are checked by
nonzero exact-rational second-difference witnesses of the frozen network.

| Seed | Dimensions | Test RMSE range | Largest test absolute error | Largest rollout state error | Certified state interval |
|---|---|---:|---:|---:|---:|
| 6101 | 8/32/64/128 | 7.50e-5 to 7.57e-5 | 5.83e-4 | 5.69e-4 | [81, 2250] each |
| 6102 | 8/32/64/128 | 7.38e-5 to 7.74e-5 | 6.07e-4 | 5.17e-4 | [81, 2250] each |

All eight cases pass the fixed gates. The smallest exact transition-closure
slack across the eight certificates is greater than 0.0017. The state and
control second differences have magnitudes above 0.0021 and 0.00067,
respectively, so these are genuine phase-crossing frozen functions.
Complete retraining, model-selection, test metric, network-weight,
certificate, and summary replay passed. Evidence:
'runs/trained_nonlinear_global_v1/summary.json' and
'validation/trained_nonlinear_global_acceptance.json'.

The dimension-independent 2,250-state upper bound follows from this
particular sparse and contractive influence graph. Most interior state
coordinates use one abstract value; the four directly observed coordinates
and the upstream coordinate feeding the first output retain resolution.
It does not prove a universal dimension-independent abstraction theorem.
The lower and upper bounds do not match. The simulator is synthetic,
the feature dictionary is structured, and the work does not complete
end-to-end causal decompilation, natural traffic-system recovery, or the
original R4/R5/R8/R9/R10 requirements. Query efficiency against passive
traces, random interventions, and activation clustering remains untested.



A matched-data affine diagnostic uses the same train, selection, and test
inputs and the same local variables (own queue, upstream queue, demand,
signal), but removes hinge features. Its held-out RMSE is at least 47.3 times
the frozen nonlinear network's RMSE in every case. The diagnostic was added
after the formal success gates were fixed and is descriptive, not an
additional preregistered acceptance criterion. Its eight cases regenerate
and replay from 'validation/trained_nonlinear_affine_ablation.json'.



Release 0.37.0 is recorded in 'validation/wheel_v37_run/status.json'.
All 165 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte. Four installed-package commands passed:
fresh nonlinear generation/replay, formal 128D nonlinear replay, and the
earlier weighted-affine replay. Wheel SHA-256:
563d7d34dfcce68210c8ee2d0fedfff08a24082b0e0476e4eea10f277b9bb805.

