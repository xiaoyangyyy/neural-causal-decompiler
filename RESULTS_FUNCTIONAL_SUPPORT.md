# Exact function-level support for frozen ReLU transitions

Version 0.40 left two valid support pairs unresolved: a narrow tent hid
between every behavior-query level, and two nonzero network paths cancelled
to a constant. This version adds a bounded exact activation-region checker.
Within each feasible full-dimensional ReLU region it computes the output's
rational affine coefficient for each input. A nonzero coefficient yields
an exact in-region intervention witness; zero coefficients in every feasible
region prove coordinate independence on the whole cube by continuity.

The formal study covers 27 fixed systems. The 20 prior trained networks
span 8, 32, 64, and 128 state dimensions. Seven controls include the
coupled phase-crossing system, two narrow tents, a three-input coupled
narrow tent, duplicate-path cancellation, and a distinct four-hinge
identity cancellation. All 27 certificates regenerate and replay.

| Family | Cases | Behavior queries | Present | Absent | Unresolved |
|---|---:|---:|---:|---:|---:|
| Learned local phase, two frozen seeds | 8 | 2,400 | 1,856 | 42,208 | 0 |
| Fixed-dictionary nonlinear, two frozen seeds | 8 | 2,400 | 1,856 | 42,208 | 0 |
| Trained affine, four frozen profiles | 4 | 1,230 | 1,472 | 21,104 | 0 |
| Seven coupled/adversarial controls | 7 | 85 | 13 | 11 | 0 |
| **Total** | **27** | **6,115** | **5,197** | **105,531** | **0** |

The region checker contributed six exact witnesses: one for the
single-coordinate narrow tent, two for the oblique two-input tent, and
three for a coupled three-input tent. None of these six edges appeared
in the fixed five-level behavior proposal. It also proved two extra
nonedges despite nonzero-weight paths: duplicate-path cancellation and
a constant expressed by four distinct ReLU hinges. The earlier 0.40
certificate had two unresolved controls; both now resolve. The new
oblique and hinge controls check that the improvement is not limited
to a single duplicate-neuron pattern.

This is a function-level certificate, not a declaration that structural
paths equal causal edges. The behavior proposal reads only the step
oracle, but the added region witnesses and absence proofs read the
frozen weights. The checker works on exact rational interpretations
of the serialized binary-float parameters. Region enumeration is
bounded to three inputs and twelve hidden ReLUs, with explicit
constraint and region caps; cases outside the bound remain unresolved
if the earlier witness/zero-path rules cannot decide them. The 128D
trained networks do not require enumeration because their initial
support certificates are already complete.

The graph is a one-step, coordinate-specific counterfactual support
graph over the declared state/action cube. It does not recover latent
variables, prove a minimum causal quotient, or validate the synthetic
networks on external systems. General large-network function-level
absence and original R4/R5/R8/R9/R10 requirements remain open.

Proof and limits: `docs/FUNCTIONAL_SUPPORT_METHOD.md`. Evidence:
`runs/functional_support_global_v1/summary.json` and
`validation/functional_support_acceptance.json`. Replay with
`python -m scripts.acceptance_functional_support`.

Release 0.41.0 is recorded in `validation/wheel_v41_run/status.json`.
All 180 regression tests passed, source/wheel/isolated-install modules
matched byte for byte, and eight installed-package commands passed.
Wheel SHA-256:
`6e7127fc673003debadbe2ff54511be25d940fd0c521d814253562ab8ac0747b`.
