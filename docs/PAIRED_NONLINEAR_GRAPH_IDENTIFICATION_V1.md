# Local nonlinear graph identification from paired node interventions

This is a conditional R9 proof kernel. It extends the archived exact linear
paired-do identity to differentiable nonlinear acyclic SCMs at one baseline
state. The estimator reads response vectors, not graph or equation metadata.
It does **not** establish graph recovery from the project's ordinary
independently sampled interventions, the frozen Discoverer, or the full
R0-R13 objective.

Let the observed-coordinate SCM be

    X_j = f_j(X_pa(j)) + U_j,

with an acyclic graph and one fixed exogenous vector u. Write x for its
baseline realization and J_{ji}=partial f_j/partial x_i evaluated at x,
using zero for nonparents. For each i, set X_i to x_i+t while retaining the
same u for every other equation. Let Y^(i)(t) be the full response. Then
Y^(i)(0)=x. Acyclicity makes every parent of i unchanged by do(X_i), so
differentiating the structural recursion at t=0 yields

    (I-J) dY^(i)/dt(0) = e_i.

Consequently, the local total-effect matrix T with these derivative columns
satisfies T=(I-J)^(-1), hence J=I-T^(-1). This is an elementary conditional
SCM calculation, not a claim of a new general identification theory.

A finite rational do step h>0 and recorded responses produce T_hat with
entries [Y_j^(i)(h)-Y_j^(i)(0)]/h. Assume each response coordinate has second
derivative bounded by M on [0,h] and each recorded coordinate has absolute
error at most delta. If all baseline/do measurements refer to the **same**
exogenous realization, then

    ||T_hat-T||_inf <= e := n(M h/2 + 2 delta/h).

The infinity norm here is the maximum absolute row sum. Let
K=||T_hat^(-1)||_inf, computed from the response data. If K e<1, the
Neumann-series inverse perturbation identity gives

    ||(I-T_hat^(-1))-J||_inf <= eta := K^2 e/(1-K e).

Assume every nonzero off-diagonal entry of J at this state has magnitude at
least gamma and eta<gamma/2. Thresholding the recovered direct-effect
estimate at gamma/2 then yields exactly the support of J. Identifying the
**structural graph** also requires every true edge to have nonzero derivative
at the chosen state. For example, X_1=X_0^2+U_1 has a real edge but its
local derivative vanishes at X_0=0. Several base states can expose different
edges, provided their bounds and visibility conditions are checked.

The executable estimator is `ncd/paired_nonlinear_graph.py`. Its status is
explicitly conditional on pairing, curvature and visibility; it refuses to
return a graph when the numerical separation inequalities do not close. It
uses exact rational arithmetic for supplied responses, not device floats.

## Independently checked nonlinear case

The development SCM has U=0 and equations

    X0=U0
    X1=3/5 X0+1/10 X0^2+U1
    X2=2/5 X0+1/2 X1+1/20 X1^2+U2.

Using h=1/16, M=1/5, delta=0 and gamma=2/5, the independent checker
recomputes every do response, the true local Jacobian, the finite-difference
matrix and its rational inverse. On [0,h] the largest nontrivial curvature
bound it derives is 17699/128000 < 1/5. Its direct-effect error bound is
728610190906563/10354418753536000 < gamma/2=1/5, and it checks the
actual direct-effect discrepancy is below that bound. Thus the recovered
edges 0->1, 0->2 and 1->2 are exact for this SCM and state.

The generator and certificate are
`validation/paired_nonlinear_graph_case_v1.py` and
`validation/paired_nonlinear_graph_case_v1.json`. The independent checker
`validation/verify_paired_nonlinear_graph_case_v1.py` uses its own rational
matrix inversion, verifies source hashes and the complete certificate, and
writes `validation/paired_nonlinear_graph_case_verification_v1.json`.
The 0.63.0.dev1 wheel is installed into an isolated target;
`validation/check_paired_nonlinear_installed_v1.py` checks wheel member,
installed module and source bytes before replaying the same result. Its
receipt is `validation/paired_nonlinear_installed_replay_v1.json`.
The hash-bound scoped ledger entry is
`validation/paired_nonlinear_graph_proof_record_v1.json`;
`validation/record_paired_nonlinear_graph_v1.py --verify` recomputes it.

A two-root counterexample in the tests shows why pairing cannot be dropped:
with true empty graph and baseline (0,0), do(X0=h) at a *different* noise
state with U1=h/2 and do(X1=h) at U0=0 falsely produces a 0->1 response.
The estimator correctly labels pairing as an assumption; it cannot infer
whether an external experiment actually held noise fixed. Neither the
mathematical proof nor the case certifies independent intervention samples,
finite-sample statistical recovery, neural-program fidelity, the exogenous
noise law, or floating-point execution. Original R9 and the 38-atom ledger
remain unresolved at their broader scope.