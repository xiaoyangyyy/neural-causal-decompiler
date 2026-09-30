# Archived SCM intervention obstruction, version 1

This is an exact counterexample for **one fixed historical symbolic SCM
candidate**, using the oracle metadata of its archived three-node
linear-Gaussian world. It is a development-world result and does not refute
all candidate SCMs or close any original R0–R13 atom.

The true graph and archived candidate graph both have edges 0→1, 0→2 and
2→1. The true node-1 mechanism is linear in parents 0 and 2; the candidate
node-1 equation contains a nonzero x0*x2 term. The candidate's separately
fitted *ideal* Gaussian noise law at node 1 has a fixed mean. Both true and
candidate noise have finite first moments.

Under the simultaneous hard intervention do(x0=t, x2=t), exact binary64
rational arithmetic gives the candidate-minus-true mean at node 1:

    Δ(t) =
      (-1451201429595957/9007199254740992) t²
      + (213391650496175/9007199254740992) t
      - 771785150891951/144115188075855872.

The coefficient of t² is nonzero. Thus the supremum of the absolute local
mechanism error on all real parent values is infinite. For each t, the
coordinate projection x↦x1 is 1-Lipschitz in the joint l1 metric, so the
Kantorovich dual lower bound gives

    W1_l1(true do(t,t), candidate ideal-law do(t,t)) ≥ |Δ(t)|.

In particular, at t=1, this lower bound is exactly

    20576741616488463 / 144115188075855872
    = 0.14277982696492597... .

Therefore the fixed candidate cannot satisfy a *uniform all-real*
intervention-distribution guarantee with any finite constant, regardless
of the Gaussian variance mismatch. The t=1 witness exceeds 0.01 in the
world's raw unit observed coordinates. This does not assert that the
original protocol's separately defined normalized error metric or bounded
intervention family has failed.

The primary certificate is
`validation/fixed_scm_intervention_obstruction_v1.json`. Its generator
expands the archived node-1 AST into an exact rational polynomial. A
separate verifier, `validation/check_fixed_scm_intervention_obstruction_v1.py`,
does not import that generator and instead checks structural degree and
recovers polynomial coefficients from exact evaluations at t=-1,0,1.
Its sealed receipt is
`validation/fixed_scm_intervention_obstruction_verification_v1.json`.
Both check the original archive-manifest SHA-256, all archived payloads,
source hashes, graphs, world metadata, candidate scope and certificate
fields. Tests reject modified numerical claims, broadened conclusion
flags and rehashed archive mutations.

This proof applies to the separately fitted *ideal Gaussian candidate
law*. The archived explicit SCM's empirical residual sampler is a
different law, and its independence is unproved. The true equations and
true Gaussian noise come from benchmark oracle metadata. The historical
world reuses one world across fit, selection and evaluation, so it is
not confirmation evidence. It remains possible that a different
candidate or a restricted input/intervention domain admits a valid
bound. The original R10 mechanism and intervention-distribution atoms
remain unresolved.
