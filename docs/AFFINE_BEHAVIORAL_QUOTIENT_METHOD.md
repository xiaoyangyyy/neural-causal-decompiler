# All-horizon behavioral quotient of globally affine ReLU networks

The original Neural Minimal Realization question asks when two internal
states give the same observable response under every admissible future
action sequence. This experiment answers that question exactly for
frozen ReLU systems whose transition and observation can be proven
affine on the complete unit state/action cube.

The checker symbolically propagates rational affine forms through each
serialized ReLU layer. For every hidden preactivation it calculates its
exact minimum and maximum on the full unit cube. A fixed nonnegative
or nonpositive sign permits an exact affine branch. A changing sign
returns unresolved. It also proves the extracted transition maps the
entire unit cube back into itself, so the same affine formula applies
at every future time.

Write the certified maps as

    x_next = A x + B a + b,      y = C x + c.

For two states given the same action word, the state difference after
t steps is A^t (x-x'), so their observation difference is
C A^t (x-x'). Hence future-response equivalence is exactly the
kernel of the observability row space

    W = span(rows(C), rows(CA), ..., rows(CA^(d-1))).

The row space is invariant under A once closed; a canonical rational
row basis T defines the quotient coordinate z=Tx. The checker verifies
T A = A_bar T and C = C_bar T exactly. These identities make the
quotient dynamics and observation well-defined for every continuous
action word and all horizons. Fibres of T are precisely the behavioral
equivalence classes. The quotient's affine dimension is rank(T);
this is not a claim about a minimum finite number of states.

For the frozen local affine networks, a special proof avoids rational
matrix powers up to 128 dimensions. Each A row depends only on its own
coordinate and its cyclic predecessor, with every predecessor
coefficient nonzero; one observation row is a nonzero multiple of
coordinate 0. The rows e0, e0A, ..., e0A^(d-1) acquire a new nonzero
coordinate in cyclic order. Their triangular minor has a nonzero
diagonal, proving full rank d. The certificate stores every exact
cyclic edge and an explicit delayed difference: perturbing state
coordinate 4 changes the observed response at horizon d-4 by the
product of the edges along its unique shortest path. A separate
Boolean support propagation checks that no output can depend on that
coordinate at an earlier horizon; the certificate records the first
distinction horizon when both checks agree.

The 128D frozen profile has four immediately observed coordinates,
yet its exact quotient dimension is 128. Coordinate 4 first becomes
visible after 124 steps. The stored rational response difference is
nonzero and has descriptive magnitude about 2.82e-161. This is an
exact distinction, however small. The same frozen model also has a
separately checked finite-state realization bound of 81 to 40,824
at uniform error 0.17. Exact behavioral equivalence and approximate
finite-state complexity are different quantities.

A controlled counterfactual replaces the wrap feedback connection
with a constant hidden unit. Exact coefficient comparison proves
that only effective A[0,127] changes; all other transition
coefficients, controls, and offsets remain identical. The first four
coordinates then form a closed subsystem and are directly observed,
so the exact quotient dimension falls from 128 to 4. This is a
structural sensitivity experiment, not retraining evidence.

For small general affine systems the checker computes the canonical
rational row space directly. One constructed two-state system has the
non-coordinate quotient z=x0+x1, of dimension 1. A constant-output
control has dimension 0. A phase-changing narrow-tent ReLU system
returns unresolved. General row-space closure is capped at dimension
12; large cases require the cyclic or closed-coordinate proof.
The method does not yet solve nonlinear phase-changing systems,
latent-state recovery from arbitrary networks, or minimum finite-state
complexity at epsilon>0.

Run:

    python -m ncd.affine_observability MODEL.json OUTPUT_DIR
    python -m ncd.affine_observability MODEL.json OUTPUT_DIR --verify

Formal study: `python -m scripts.acceptance_affine_quotient`.
Evidence: `runs/affine_quotient_global_v1` and
`validation/affine_quotient_acceptance.json`.
