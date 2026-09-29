# Exact phase-aware initial handoff

This proof replaces the absolute-weight initial sensitivity predicate for
supported frozen networks. It changes neither neural weights nor the fixed
weighted recurrent relation. It proves a minimum initialization label count
of 81, then compiles the corresponding complete closed finite programs.

## Model and arithmetic

Observation must equal the first four state coordinates, with zero bias,
on the full unit state cube. Tolerance is exactly 17/100. The recurrent
certificate is independently replayed. The transition has two controls
and one hidden ReLU layer. Every scalar output may involve at most two
state coordinates. Every relevant hidden unit must be state-only or
action-only; mixed units and unsupported architectures are rejected.
These properties are extracted from actual nonzero network weights.

Serialized binary64 weights are interpreted as exact dyadic rationals.
The certificate concerns the resulting mathematical neural function;
it does not additionally certify floating-point inference roundoff.
No teacher equations, training targets or fitted local surrogate are used.

## Signed phase extrema and the handoff

The decomposition of each output is F_i(x,u)=G_i(x_a,x_b)+A_i(u).
Use three initial bins on observed axes, one on all hidden axes. For each
state pair-cell, split its rectangle at all actual crossing hidden
preactivation lines. Rational closed half-plane clipping covers the
rectangle. Verify every selected ReLU sign at every polygon vertex,
construct the signed affine output form, then take vertex extrema.
If c is the cell midpoint, the exact bound is

    E_i = max_cells max(|min G_i - G_i(c)|, |max G_i - G_i(c)|).

Signed output sums preserve cancellations that the older absolute-weight
predicate discards. On action phase polygons compute

    L_i = max_regions (|partial_0 A_i| + |partial_1 A_i|).

The piecewise-affine function is continuous, so this bounds its global
L-infinity Lipschitz constant on the convex action square. At action
midpoint quantization with m=128, and recurrent coordinate bins n_i,
verify for every coordinate

    E_i + L_i/(2m) + 1/(2n_i) <= r_i.

The first term controls every concrete initial point in its cell, the
second controls continuous action quantization, and the third controls
recurrent midpoint quantization. The verified recurrent relation then
propagates error at every finite horizon. At time zero observation error
is at most 1/6 < 17/100.

Certificates contain all pair-cell extrema, phase hashes, signed control
bounds, coordinate inequalities and slacks. Replay reconstructs the entire
certificate. Boundary cells are closed, so quantization ties are covered.
At 128D there are only 156 pair-cells to check because supports are sparse;
this is not an enumeration of the full 128-dimensional initial cube.

## Minimum initial labels and complete programs

The 81 initial states use three bins on each of four observed axes. The
81 initial packing points {0,1/2,1}^4, with remaining axes zero, have
pairwise output distance at least 1/2 > 2*(17/100). Any deterministic
state-only-output machine accurate at time zero needs distinct labels
for these points. The certified grid attains 81 initial labels, so the
minimum initialization label count is exactly 81, even among arbitrary
selectors and partitions. This says nothing by itself about minimum total
machine states.

Explicit certificate-schema dispatch preserves the old sensitivity proof.
The action-polygon closure enumerates all new initial centers and the
entire recurrent fixed point. Standalone program verification reconstructs
every state row, offset and normalized action template. Execution uses
only program JSON, without the source network. The new machines contain
81 initial plus 51/58/54/55 recurrent states at 8/32/64/128 dimensions.

The old 108-cell optimum remains valid within its fixed absolute-weight
proof predicate. Its failed inequality was a failure of that proof, not
an impossibility result for an 81-label initialization.

Replay: `python -m scripts.acceptance_phase_initial_handoff --verify`.
See [results](../RESULTS_PHASE_INITIAL_HANDOFF.md) and the complementary
[whole-machine lower theorem](TRANSITION_CONSISTENCY_LOWER_METHOD.md).
