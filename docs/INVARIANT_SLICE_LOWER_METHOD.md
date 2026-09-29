# Invariant-slice lift for finite-state interventional lower bounds

The frozen scalar neural system already has an exact-rational transition
lower certificate: at uniform output error epsilon=0.101, no deterministic
realization with fewer than seven states can cover every initial state and
every continuous action word. The two-dimensional frozen benchmark is two
separable copies of that scalar system, but its prior lower bound used
only initial output packing and was 25. This result lifts the scalar
transition obstruction to four disjoint invariant slices and proves
a 28-state lower bound.

Let the frozen scalar dynamics be x'=lambda*x+beta*a+offset, y=x,
with 0<=lambda<1. The checker extracts exact affine coefficients from
the serialized ReLU networks and proves that the 2D transition is
exactly two copies with identity observation. A second-coordinate
value c is fixed by the admissible action

    a_c = ((1-lambda)*c-offset)/beta.

The valid fixed-point interval is the intersection of [0,1] with
[offset/(1-lambda), (offset+beta)/(1-lambda)]. For this network it
contains four points c_1,...,c_4 with pairwise distance greater than
2*epsilon. The certificate constructs the points and actions as exact
rationals, checks action admissibility, and replays the scalar
multi-switch lower certificate.

Consider *any* m-state deterministic realization with error at most
epsilon for the complete 2D state cube, all continuous action words,
and all horizons. For a chosen slice c_i, restrict initial states to
(x,c_i) with x ranging over [0,1], and restrict second-coordinate
actions to a_c_i. The second coordinate stays c_i forever. All abstract
states that appear on runs from this slice, with their first output
coordinate, form a finite realization of the scalar system. The
certified scalar theorem forces at least seven such states.

No abstract state can appear on two distinct slices: its fixed second
output would have to be within epsilon of both c_i and c_j, which is
impossible when |c_i-c_j|>2*epsilon. The four reachable abstract-state
sets are disjoint. Therefore m>=4*7=28, independent of encoder,
state outputs, action-dependent transition table, or candidate grid.

The upper certificate is unchanged: a replayed executable
81-state realization achieves the same epsilon for every initial
state and continuous action sequence. Thus the certified interval is

    28 <= K_N(0.101, all continuous action words) <= 81.

This argument uses invariant slices, not the false assumption that
the approximate state complexity of a product automatically equals
the product of the component complexities. It applies only when
exact separability, identity observation, admissible fixed actions,
and the scalar lower theorem all hold. It does not establish the
exact 2D minimum or close the scalar 7–9 interval.

Run:

    python -m ncd.invariant_slice_lower SCALAR.json PRODUCT.json OUTPUT
    python -m ncd.invariant_slice_lower SCALAR.json PRODUCT.json OUTPUT --verify

Formal evidence and independent historical replays:
`runs/invariant_slice_lower_v1` and
`validation/invariant_slice_lower_acceptance.json`.
