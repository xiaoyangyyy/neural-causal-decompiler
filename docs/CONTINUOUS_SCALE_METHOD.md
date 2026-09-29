# Continuous certification scaling method

## Objective

This stage measures whether certified neural-state distinguishability can scale
past the one-dimensional horizon-one benchmark. The declared profiles increase
continuous latent dimension, control dimension, horizon, and neural parameter
count together:

| Profile | State dim | Action dim | Horizon |
|---:|---:|---:|---:|
| 0 | 8 | 2 | 3 |
| 1 | 32 | 3 | 5 |
| 2 | 64 | 4 | 10 |
| 3 | 128 | 5 | 20 |

Each transition network is fitted from continuous state/control samples. The
trained network itself is the system under verification. The generating affine
dynamics are used only to produce training targets.

## Stable relational bound

Independent interval propagation loses the fact that the two compared neural
executions receive the same control word. The relational verifier propagates an
interval for their difference together with separate state enclosures. At an
affine layer, shared controls and biases cancel from the difference.

For every hidden ReLU, the verifier requires both executions to be certified in
the active phase over the complete box, or both to be certified in the inactive
phase. The difference then passes through the same linear phase exactly. If this
condition cannot be proved, the certificate leaf falls back to independent
interval propagation. Thus stability improves tightness but is never assumed
without verification.

All scalar affine operations use the existing outward-rounding interval
implementation. A feasible center control word supplies the lower bound. The
stored certificate records which bound method justified every leaf, and the
independent verifier recomputes it.

## Comparisons

Every profile contains a near pair and a far pair. The relational method and
independent IBP receive the same complete action box. Independent IBP receives
eight branch leaves in the formal run, while the relational method uses one
unsplit leaf. Random continuous control words are also evaluated, but are
reported only as lower evidence and never as an absence certificate.

## Claim boundary

The experiment studies learned ReLU representations of stable affine dynamics.
The ReLU networks are high-dimensional, but all hidden activations are
certifiably stable over the declared domains. The result establishes scaling in
dimension and horizon for that regime. It does not establish comparable scaling
when many ReLUs cross phase boundaries, for arbitrary continuous initial-state
regions, or for nonlinear traffic data.
