# All-horizon behavioral quotient of frozen affine neural systems

This experiment returns to the core Neural Minimal Realization question:
which internal states are indistinguishable under *every* admissible
future action sequence? For a frozen ReLU system proven globally affine
and invariant on the full unit cube, it computes that equivalence relation
exactly rather than choosing an interpretability feature or a grid.

The certified theorem is simple but consequential. With
x_next=Ax+Ba+b and y=Cx+c, equal future action words cancel from the
difference between two runs. Thus x and x' are equivalent exactly when
C A^t(x-x')=0 for every t. The checker constructs the invariant
observability row space and its canonical rational basis T; fibres
Tx=Tx' are exactly the behavioral classes. It verifies the induced
quotient equations T A=A_bar T and C=C_bar T. The reported dimension is
the affine dimension of this exact quotient, not the number of states
in a finite epsilon realization.

| Frozen or controlled system | State dimension | Observed outputs | Exact quotient dimension | Certificate |
|---|---:|---:|---:|---|
| Trained affine ring, profile 000 | 8 | 4 | 8 | Cyclic triangular delayed observation |
| Trained affine ring, profile 001 | 32 | 4 | 32 | Same |
| Trained affine ring, profile 002 | 64 | 4 | 64 | Same |
| Trained affine ring, profile 003 | 128 | 4 | 128 | Same |
| Profile 003 with one effective feedback edge removed | 128 | 4 | 4 | Closed coordinate projection |
| Constructed mixed sum | 2 | 1 | 1 | Exact row-space closure |
| Constructed constant observation | 2 | 1 | 0 | Constant response |
| Nonlinear narrow-tent control | 1 | 1 | unresolved | Phase changes on cube |

The 128D result is a negative answer to naive exact compression.
Coordinate 4 is invisible initially; it first affects an output after
124 steps under the same all-zero action word. The exact rational
response difference is nonzero, with descriptive magnitude
2.821186441973509e-161. The frozen transition's cyclic predecessor
connections form a triangular delayed-observation proof that every
state direction eventually matters. Therefore no two distinct states
in the full cube have identical responses under all future words.

The controlled 128D ablation deletes only effective coefficient
A[0,127]. A compensating constant hidden unit preserves every other
effective transition coefficient and offset *exactly*, as checked
against the original frozen network. The first four coordinates then
form a closed, directly observed subsystem; all other 124 coordinates
can be quotiented out. This is a counterfactual architecture control,
not a claim that the trained network naturally contained a four-state
or four-coordinate exact quotient.

The small mixed example proves the method is not restricted to
coordinate selection: its exact classes are determined by x0+x1.
The constant-output example has a single behavioral class. The
phase-changing ReLU control is deliberately unresolved; no affine
theorem is applied to it.

The same original profile 003 also has a separately replayed
*approximate finite-state* certificate at epsilon 0.17:
81 <= K_N(0.17, all continuous action words) <= 40,824.
This does not conflict with an exact quotient of affine dimension 128.
The exact witness above is roughly 160 orders of magnitude below
the allowed error, and exact dimension is a different mathematical
quantity from finite epsilon state count.

All eight cases regenerate and replay in
`runs/affine_quotient_global_v1` and
`validation/affine_quotient_acceptance.json`. The underlying proof,
scope and algorithms are in
`docs/AFFINE_BEHAVIORAL_QUOTIENT_METHOD.md`.

This is a canonical exact quotient only for globally affine frozen
systems whose state cube is invariant. It does not solve the nonlinear
phase-changing case, recover latent variables from arbitrary 20M/80M
parameter networks, or close the approximate upper/lower gap.
Original R4/R5/R8/R9/R10 remain open.

Release 0.42.0 is recorded in `validation/wheel_v42_run/status.json`.
All 189 regression tests passed. Source, wheel and isolated-installed
modules matched byte for byte; eight installed-package commands passed,
including fresh quotient generation, the 128D frozen and ablated replays,
and the prior approximate-grid replay. Wheel SHA-256:
`679dccc5a82e95123389e53074d875de4cc62642bea8ebf582db0d01899fda3e`.
