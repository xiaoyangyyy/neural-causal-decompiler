# Exact scalar minimum via interval-chain exclusion

For the frozen scalar ReLU system with exactly affine dynamics
`x_next = 0.5*x + float64(0.4)*a`, identity output, all initial states
and continuous actions in `[0,1]`, and `epsilon=float64(0.101)`, the
minimum deterministic intervention-labelled finite realization has **exactly
9 states**. The frozen separable two-dimensional product now has the
certified minimum-state interval **36?81**; its exact minimum remains open.

The lower proof converts every possible finite realization into a family of
interval hulls. It allows arbitrary abstract outputs, encoders, and
action-dependent deterministic transitions. For each source interval, all
increasing target subchains are considered, including those that skip
intermediate intervals. Every eight-state branch is closed by a rational
Farkas identity. The 144,246-byte compressed proof stream has 40,545
branch records: 40,387 closed branches and 158 internal branches. The
independent verifier checks every identity with exact fractions and
requires complete branch coverage. A smaller realization could be padded
by duplicating a state interval and its target choices, so excluding eight
also excludes all smaller state counts.

The matching upper construction uses nine output centers at
`0.1, 0.2, ..., 0.9` and exact relation radius `201/2000`. An exact
rational checker verifies initial coverage, output error, and an action
subcover for every source across the entire continuous action interval.
This gives an executable inductive realization for every finite action
word without floating-point interval rounding.

The prior invariant-slice theorem finds four fixed second-coordinate
values whose outputs are pairwise separated by more than `2*epsilon`.
Each slice is a copy of the scalar system, so each needs nine states;
states cannot be shared between slices. This raises the 2D lower bound
from 28 to 36. The replayed nine-by-nine upper construction still has
81 states.

Evidence: [method](docs/EXACT_INTERVAL_CHAIN_METHOD.md),
[proof stream](validation/exact_interval_lower_8.json.gz),
[acceptance](validation/exact_interval_lower_acceptance.json), and
[2D slice certificate](runs/exact_interval_lower_v1/certificate.json).
Reproduce with

```powershell
python -m scripts.acceptance_exact_interval_lower
```

This closes the scalar minimum only for the frozen globally affine ReLU
benchmark at the declared tolerance and domains. It does not prove the
2D optimum, a general nonlinear minimum-realization theorem, or the
minimum finite realization of the trained high-dimensional networks.

Release 0.44.0: all 195 regression tests passed. The isolated wheel
installation replayed the eight-state proof, exact nine-state upper,
invariant-slice lift, and historical 81-state upper; all 104 installed
Python modules matched source byte for byte. Evidence:
[wheel status](validation/wheel_v44_run/status.json). Wheel SHA-256:
`bed88cf2d7f2c4b69478842e3542b923ee81f7022f830a8d24ecaa8c73440519`.
