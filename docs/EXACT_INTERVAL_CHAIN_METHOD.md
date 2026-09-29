# Exact interval-chain lower bound for the frozen scalar ReLU system

The frozen scalar network is exactly affine on the full unit state-action box:
`F(x,a) = lambda*x + beta*a`, `O(x)=x`, with `lambda=1/2` and
`beta` equal to the exact binary64 value serialized as `0.4`. The tolerance
is the exact binary64 value serialized as `0.101`. Initial states and actions
range over all real numbers in `[0,1]`; the horizon is unbounded.

For any deterministic finite-state realization, collect every concrete state
that can be paired with abstract state `i` on any valid run, and let
`I_i=[L_i,U_i]` be its closed interval hull. Observation error gives
`U_i-L_i <= 2*epsilon`. The initial selector gives a cover of `[0,1]`.
The scalar affine transition is monotone, so hulls preserve the inductive
transition requirement:

    for every source i and action a, some target j has
    [lambda*L_i+beta*a, lambda*U_i+beta*a] subset [L_j,U_j].

An interval contained in another may be deleted: its initial assignments
and incoming transitions can be redirected to the containing interval,
while all remaining source transitions keep their existing targets. Thus
one may sort the intervals by nondecreasing lower *and* upper endpoints.
Their initial cover requires `L_0=0`, `U_(m-1)=1`, and
`L_(j+1)<=U_j` for all adjacent pairs.

For source `i`, target `j` accepts precisely an interval of actions

    A_ij = [(L_j-lambda*L_i)/beta,
            (U_j-lambda*U_i)/beta] intersect [0,1].

Because target lower and upper endpoints are ordered, an interval subcover
of `[0,1]` can be chosen as an increasing chain `j_0 < ... < j_r`.
The first target accepts action 0, the last accepts action 1, and every
successive action interval overlaps. These are *linear* inequalities in
the unknown interval endpoints. For an edge `j<k`, the overlap inequality is

    L_k - U_j + lambda*(U_i-L_i) <= 0.

The proof enumerates **all** nonempty increasing target chains for each
source, including chains that skip intermediate targets. This matters:
requiring every intervening target to participate is not a valid necessary
condition. A numerical mixed-integer solver is used only as a search guide.
Each closed branch carries a sparse nonnegative rational Farkas vector `y`
with `A^T y=0` and `b^T y<0` for that branch's exact inequalities `Ax<=b`.
`ncd.exact_interval_lower.verify_interval_chain_exclusion` reconstructs
every branch and checks those identities using Python rational arithmetic;
it does not call an LP or MILP solver.

The complete compressed proof stream is stored in `validation/`. The
acceptance script independently replays that stream, the existing exact
seven-state lower theorem, the executable nine-state scalar upper model,
and the two-dimensional invariant-slice lift. An exclusion at `m=8` also rules out every smaller realization: any
feasible interval family with fewer than eight states can be padded by
duplicating an interval and its target choices, while the chain may skip
the duplicate. Thus an eight-state exclusion and the existing nine-state
construction close the scalar minimum at exactly nine. Four
second-coordinate fixed-point slices separated by more than `2*epsilon`
then force at least `4*9=36` states for the separable two-dimensional model;
the existing upper construction has 81 states.

The theorem is specific to this frozen scalar affine system and its exactly
separable two-dimensional product. It does not establish minimum state
counts for coupled nonlinear or trained high-dimensional networks.
