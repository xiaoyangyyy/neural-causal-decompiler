# Succinct global realization of frozen trained ReLU dynamics

The frozen 8/32/64/128-dimensional continuous networks from
'runs/certified_continuous_scale_seed4701' now have exact-rational
infinite-horizon simulation certificates over every initial state in
[0,1]^d and every continuous action sequence in [0,1]^u.
The models were already trained and serialized in the earlier scaling run;
this experiment does not refit or select their weights.

The abstract state is a vector of 14 grid indices. An action is quantized to
one of 64 midpoints per coordinate. The executable transition evaluates the
frozen network at the grid center and action midpoint using exact rational
arithmetic, then quantizes each resulting state coordinate. The output is the
frozen observation network evaluated at the state center. The machine is
succinct: its state count is 14^d, but its transition table is never
enumerated.

The checker interprets each serialized binary64 network weight as its exact
rational value. It independently recomputes these obligations from the frozen
model, not from the training target:

1. Exact interval propagation sends the full unit state-action cube into the
   unit state cube.
2. A grid cell is at most 1/(2*14) from its center, within the chosen
   relation radius delta=0.16.
3. For every ReLU layer, multiplication by the absolute weight matrix bounds
   coordinatewise perturbations. With state perturbation delta and action
   midpoint error 1/(2*64), the resulting transition bound plus final
   state rounding 1/(2*14) is at most delta.
4. The same weight-based bound limits all observation differences under the
   relation to 0.16 < epsilon=0.17.
5. The 81 initial points obtained from {0,0.5,1}^4 in the first four
   coordinates have pairwise observation distance 0.5>2*epsilon.
   Therefore every epsilon-accurate deterministic realization needs at
   least 81 states, even before transition constraints.

The first four obligations form an inductive simulation proof. The initial
relation holds, the transition preserves it for every action, and the
observation error holds at every time. Consequently the upper bound applies
to unbounded action sequences without sampling a horizon.

| Frozen model | State / action dimensions | Certified interval for minimum states | Largest transition error bound |
|---|---:|---:|---:|
| profile_000 | 8 / 2 | [81,14^8] | 0.156644 |
| profile_001 | 32 / 3 | [81,14^32] | 0.157550 |
| profile_002 | 64 / 4 | [81,14^64] | 0.158820 |
| profile_003 | 128 / 5 | [81,14^128] | 0.158618 |

Evidence: 'runs/compositional_scale_seed4701/summary.json' and each
profile's 'certificate.json'. 'verify_frozen_scale_models' recomputes the
rational obligations and rejects changed certificate fields or model hashes.
The source module is 'ncd/continuous_compositional_realization.py'.

This is a global trained-network upper certificate, unlike the earlier
fixed-state finite-horizon separation certificates. It is still a deliberately
weak upper bound: 14^128 abstract states would be infeasible to enumerate,
even though the transition is executable on demand. The 81-state lower bound
comes from only four observed coordinates and does not approach the upper
bound. These synthetic trained systems are stable affine dynamics encoded by
ReLU networks, with continuous but synthetic controls. The result does not
establish minimum realization, a nonlinear learned traffic model, or the
original broader causal decompilation requirements.

An exploratory eight-state search for the separate 1D frozen affine benchmark
is recorded in 'validation/explore_variable_eight_state.py' and
'validation/explore_finite_color_milp.py'. The former found no admissible
eight-interval construction. The latter found a finite sampled coloring using
only six colors for three discrete controls. Neither diagnostic is a global
eight-state construction or exclusion proof; the certified 1D interval remains
[7,9].



## Dimension-independent coordinate-weighted certificate

A second exact checker allows one state-bin count and one relation radius per
neural coordinate. It tests a candidate vector using the same full-cube
invariance and layerwise absolute-weight bounds, now coordinate by coordinate.
For the frozen models, the candidate uses 12 bins for state coordinate 0,
six bins for each of coordinates 1-3, one bin for every interior coordinate,
and 12 bins for the final coordinate. Relation radii are 0.17 for the five
resolved coordinates and 1.01 for the one-bin coordinates. Each continuous
control coordinate uses 128 action bins. The large hidden radius is valid:
a one-bin relation covers the whole unit interval, and weak inter-coordinate
coupling prevents that uncertainty from breaking the observed-coordinate
closure.

All four models pass exact recomputation and replay. Their certified upper
bound is now **12^2 * 6^3 = 31,104 states**, independent of the tested state
dimension. The lower bound remains 81. The 128D upper has therefore fallen
from 14^128 to 31,104 without changing the frozen network or epsilon.
The executable weighted machine computes transitions on demand; the complete
finite-state set is small enough to enumerate if desired, though the action
partition still grows as 128^u and was not enumerated.

Evidence: 'runs/weighted_compositional_scale_seed4701/summary.json' and
profile certificates. The checker re-evaluates every rational bound and the
frozen model hash. This result is specific to the sparse, contractive influence
pattern of the frozen trained affine models. It does not imply that arbitrary
128D recurrent networks admit a dimension-independent quotient or that
31,104 is minimal.



The exact-replay acceptance record is
'validation/compositional_trained_acceptance.json'. The complete project
regression suite passes 163 tests after adding four compositional checks.



Release 0.36.0 is recorded in 'validation/wheel_v36_run/status.json'.
All 163 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte; five installed-package compositional generation
and replay commands passed. Wheel SHA-256:
9c42ac17aa0f8b210670b93b354d1a3302999d885e819d4d8b6e68be1755dd1e.

