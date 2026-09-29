# Continuous initial-region certification method

## Domain and response metric

Let L and R be axis-aligned initial-state boxes and let A be the complete
finite-horizon continuous action-word box. For an initial state x and action
word a, Y(x,a) is the observed neural response trace. The response distance is
the maximum over time of the L-infinity observation distance.

The verifier can return two conclusive statuses:

- certified-within-epsilon proves that every x in L, every y in R, and every
  action word a in A have response distance at most 2*epsilon.
- robustly-separated stores one action word a* and proves that every x in L and
  every y in R have response distance greater than 2*epsilon under a*.

The robust separation statement is stronger than finding a witness for one
pair of center points. If neither statement closes, the result is unresolved.

## Product-domain branch certificate

The root domain is the Cartesian product L x R x A. Each tree leaf stores a
box over both initial states and the full action sequence. A split may divide a
left-state coordinate, a right-state coordinate, or an action coordinate. The
leaf upper bound encloses every cross-region response distance in that box.
The global upper bound is the maximum leaf upper bound.

A robust lower witness is always evaluated against the complete original
initial boxes. Initial-state subdivision can therefore never turn
region-specific witnesses into an invalid global separation claim. The
certificate stores the witness action word and the numerical method used to
justify its lower bound.

## Relational and fallback bounds

When both executions have the same certified stable ReLU phase, the verifier
propagates their difference and cancels shared controls and affine biases.
This supports both an upper enclosure and a robust lower witness. If any paired
phase cannot be certified, the affected calculation falls back to
outward-rounded independent interval propagation. Every leaf and witness
records its actual method.

The independent verifier checks the system hash, exact product-domain
coverage, every split, every leaf bound, the complete-domain witness lower
bound, aggregate status, artifact hashes, deterministic model retraining, and
deterministic certificate regeneration.

## Formal profiles

| Profile | State dim | Action dim | Horizon | Per-coordinate radius | Leaf budget |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 2 | 3 | 0.001 | 8 |
| 1 | 32 | 3 | 5 | 0.001 | 8 |
| 2 | 64 | 4 | 10 | 0.001 | 8 |

The learned networks are the systems under verification. Their generating
stable affine dynamics are used only for training targets.

## Claim boundary

These certificates establish local box-to-box response statements for fixed
finite horizons and complete bounded action boxes. They do not construct a
finite quotient for an uncountable state domain, cover arbitrary unions or
curved regions, prove global minimality, or address widespread ReLU phase
crossings inside the initial boxes.
