# Exact functional support from ReLU activation regions

Version 0.40 certified one-step coordinate support when behavior queries found
every structurally possible edge. Two valid counterexamples remained:
a narrow ReLU tent hid between all five query levels, while two nonzero
paths cancelled to a constant. Version 0.41 adds an exact, bounded
function-level checker for these cases.

For a frozen ReLU transition F on the full unit state/action cube, a support
edge j -> i exists iff two inputs differing only in coordinate j give
different F_i values. The behavioral proposal still sees only the callable
step oracle. The new region checker sees the frozen serialized weights
and can add exact edge witnesses or prove additional absences. Consequently,
a region-discovered edge is *not* claimed to be behavior-only discovery.

The checker expresses every hidden preactivation as an affine rational
function of the original inputs after fixing preceding ReLU signs.
It branches on strictly positive and strictly negative signs. A
preactivation identically zero on a current cell contributes zero without
branching. The unit cube interior and all sign constraints are strict
linear inequalities over exact rationals. Fourier-Motzkin elimination
determines feasibility and reconstructs a rational interior point.
Infeasible sign combinations are discarded.

On each feasible activation cell, every output is affine. If its
coefficient for input j is nonzero, the checker moves a rational distance
along j while staying inside that cell and re-evaluates both points on
the exact frozen network. This produces a nonzero intervention witness.
If the coefficient is zero in every feasible full-dimensional cell,
the output is independent of j on the entire closed cube: the cells
cover a dense subset of the interior, and the continuous piecewise-affine
network extends the equality across cell boundaries and cube boundaries.

This rule can prove absence even when nonzero weight paths exist. It also
finds narrow dependencies that a finite query grid misses. A test with
four distinct hinge units uses
ReLU(x-.25)-ReLU(.25-x)-ReLU(x-.75)+ReLU(.75-x) = .5
on the full cube. Its graph contains paths, but its state input is
functionally absent; exact region enumeration verifies that all feasible
cell slopes are zero.

The algorithm is exponential in general. It currently attempts region
enumeration only when the transition has at most three inputs and twelve
hidden ReLU units, with explicit region and Fourier-Motzkin constraint
caps. Exceeding a cap returns unresolved. Larger frozen networks from
the formal suite already have complete certificates from exact query
witnesses and structural zero paths, so no region enumeration is
required there. This is not a scalable general network-equivalence solver.

The certificate stores exact witnesses, function-level absence pairs,
region counts, resource limits, the frozen model digest, and unresolved
pairs. Verification regenerates the query proposal and recomputes all
exact region classifications from the frozen model. The proposal
uses float64 responses; the formal certificate interprets serialized
float coefficients as exact rationals.

Run:

    python -m ncd.functional_support path/to/system.json output/dir
    python -m ncd.functional_support path/to/system.json output/dir --verify

Formal study: `python -m scripts.acceptance_functional_support`.
Evidence: `runs/functional_support_global_v1` and
`validation/functional_support_acceptance.json`.

The broader research goal remains open. These graphs use declared
coordinates and one-step interventions, not learned latent variables or
a globally minimal causal computation quotient. Large dense networks
with missing query witnesses still need a stronger proof-producing
method, likely compositional region reasoning, exact solver certificates,
or other tractable sufficient invariants.
