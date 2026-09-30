# Exact obstruction for the implemented unary mechanism grammar

Two frozen one-parent Tanh mechanisms from seed 8301, three-node test-id
worlds have an independently verified finite-grammar fidelity obstruction.
For both, the implemented mechanism_library((0,)) offers the functions
1, x, x^2, sin(x), cos(x), and tanh(x). The theorem allows **every real
linear combination of all six**. This is more permissive than the
current structured eight-term search, which admits at most one unary
operator per parent.

| Frozen network | Parent box | Raw grid-error lower bound | Training scale | Normalized lower bound |
|---|---:|---:|---:|---:|
| World 0, node 1; correct inferred graph | [-2,2] | 0.021151184166321 | 0.635165393353 | **0.033300277987** |
| World 5, node 1; incorrect inferred graph | **[-1,1]** | 0.026157459637773 | 0.804526329041 | **0.032512869615** |

Both bounds exceed the 0.01 normalized frozen-network fidelity gate.
Graph correctness is recorded because it matters to causal recovery;
the grammar theorem itself concerns approximation of the frozen neural
function and does not use the true graph.

The neural semantics interpret every binary32 checkpoint parameter as
an exact rational and evaluate affine and Tanh layers over the reals.
Extra hardware inference rounding is not covered. The first network
has SHA-256
6b232a551fd02516c0281ecb30f3596196f9223e747d4105c5c04608ed64e3f3;
the second has SHA-256
3f24bd64320498505c27aa78ce7bde43b260c7f6d7f34d773da9af302d26d420.
The first theorem's seven rational points are
-2, -1, 0, 1/2, 1, 3/2, 2. The second theorem uses
-1, -3/4, -1/2, 0, 1/2, 3/4, 1, entirely inside the historical
[-1,1] box. These are parent-input queries, not all frozen do values.

Here is the proof common to both. Let B be the 7-by-6 matrix of exact
basis values and y the seven exact neural outputs. A rational midpoint
matrix B0 rounds each rigorously enclosed entry to denominator 2^48.
The first six rows of B0 have an exact rational inverse. The certificate
gives rational dual weights lambda with l1 norm 1 and
lambda-transpose B0 = 0 exactly. Outward intervals for sin, cos, Tanh,
and the neural outputs use checked Taylor or exponential remainders.

Let t = ||Bc-y||_infinity for any real coefficient vector c. Let E
bound B-B0, Y bound the first six neural outputs, and D be the stable
anchor inverse bound

    D = ||B0_anchor^-1||_infinity /
        (1 - ||B0_anchor^-1||_infinity ||E_anchor||_infinity).

Then ||c||_infinity <= D(Y+t), while the dual gives

    |lambda-transpose y| <= t + ||E||_infinity D(Y+t).

The rational lower enclosure for |lambda-transpose y| therefore yields
an **unconditional** lower bound on t. Since the seven points lie in
the declared box, this also bounds the supremum error over that box.
The two exact certificates and independent receipts are under validation:

- frozen_unary_grammar_obstruction_v1.json and its _verification_v1.json receipt;
- frozen_unary_grammar_obstruction_unit_box_v1.json and its
  _unit_box_verification_v1.json receipt.

The read-only verifier reopens each checkpoint, checks the actual
mechanism_library source and interval kernel hashes, independently
re-evaluates all seven network and basis values, checks dual
annihilation and anchor invertibility, and recomputes the rational
lower bound. The scoped proof records explicitly link these
certificates to R4.program_fidelity and R10.mechanism without changing
their unresolved original status.

The conclusion covers every real coefficient vector in the **current
six-function one-parent grammar** for the two named checkpoints.
It says nothing about a larger or nested program language, other
frozen networks, true SCM correctness, noise recovery, or extra
device rounding. The original all-instance R4/R10 targets remain open.
