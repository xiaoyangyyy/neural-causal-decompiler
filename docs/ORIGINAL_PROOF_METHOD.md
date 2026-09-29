# Original causal decompilation: proof contracts and current boundaries

Version 0.59 implements the first original-project proof milestone. It is not
completion of R0-R13. The original ledger contains 38 atomic claims; two
universal observational recovery claims are strictly refuted, and 36 remain open.
Existing implementation, successful tests and scoped certificates do not close
stronger original claims.

## Objects and semantics

Keep the true SCM, frozen discovery network, extracted discovery program and
conditional neural mechanism as four separate objects. Proofs interpret saved
binary coefficients as exact rational weights and mathematical operations.
Additional CPU/GPU inference rounding is outside the current certificates.
A protected division is not ordinary division. CDIR protection thresholds are
interpreted from their actual binary64 values.

`ncd prove --config validation/original_proof_milestone_protocol.json` produces
a proof bundle from a frozen job protocol. `--resume` accepts only the same
configuration, requirements snapshot and source identities. Every job binds
its declared domain, tolerance, frozen training scale, checkpoint and candidate.
`ncd verify-proof runs/original_proof_milestone_v1` independently replays all
mathematical conclusions and verifies checkpoint-to-export correspondence.
`ncd audit-requirements runs/original_proof_milestone_v1 --require-closed`
returns exit code 1 while original claims remain unresolved. Integrity replay
success is separate from whole-project completion.

## Observational impossibility

Let a=1/2. Forward SCM: X=U, Y=aX+V, with independent centered Gaussian
variances 1 and 3/4. Reverse SCM: Y=U', X=aY+V', with the same independent
Gaussian variances. Both have mean zero and covariance [[1,1/2],[1/2,1]],
which is positive definite. A centered Gaussian law is fixed by this covariance,
so the entire observational laws and all n-fold sample laws coincide.
Under do(X=1), E[Y] is 1/2 in the forward model and zero in the reverse.

For any estimator, including randomized estimators and access to the exact
population observational law, the probabilities of the two different correct
answers sum to at most one. Therefore one direction error is at least 1/2.
This is a universal two-model obstruction, not a failed fitted candidate.
Additional identifying assumptions, informative interventions and equivalence
class recovery are separate tasks. Unknown/CPDAG output is not a failure of
identification when direction is not identifiable.

## Deterministic neural/program verification

The interval engine uses rational outward rounding with an 80-bit denominator.
Exponential enclosures use range-reduced Taylor series and geometric tail bounds;
logarithms use the positive atanh series and its checked tail. Tanh follows from
exp; square root bounds use integer square roots; trigonometric bounds use
Taylor's derivative remainder and global Lipschitz bounds. Arithmetic rounding
only widens intervals. No floating optimization status supplies a proof.

Frozen Tanh mechanisms are compared against the actual historical CDIR
expressions over declared closed boxes. A proof tree stores every split and
both closed children, including shared boundaries. Verification recomputes
bounds and checks complete coverage. A strict point-error enclosure proves a
candidate counterexample. Uncovered cells, unsupported operators or exhausted
budgets remain unresolved. All mechanism errors are normalized by the frozen
network output scale; the default threshold is exactly 1/100.

The historical raw-data Discoverer is also exported, including sample
normalization, clipping, both encoder/head stacks, swapped input and logit
symmetrization. A lazy interpreter derives the historical Rule's decision from
certified raw CDIR statistics. Current certified statistics include moments,
protected normalization and centered-kernel dependence with order-statistic
bandwidth enclosures. Cross-fit regression uses certified sorting, interval normalization and
Gaussian elimination with nonzero-pivot certificates; residual-kernel paths
reuse the same dependence enclosure. Ambiguous sorting, bandwidth selection
or pivots remain unresolved, never success.
Attention has a sound interval primitive, but full graph-attention export and
its raw statistical front end are not yet integrated. This remains an R9 gap.

## CEGIS and internal intervention scope

The bounded candidate generator queries only the frozen mechanism. Its grammar
uses total-degree <=4 polynomials, at most two parent coordinates and at most
eight fitted terms. Each strictly checked counterexample feeds the next fit.
The verifier reconstructs candidates from the same neural queries and witnesses
and rechecks every round. Candidate failure does not prove this grammar, or
all program languages, impossible. No global MDL minimum is claimed.

The linear interchange checker verifies the exact read/write diagram for every
declared compatible mask and all source/base states in their boxes. It includes
unpatched coordinate leakage and signed cancellation. A separate source mode
handles independent source worlds for each intervened variable; cancellation
between different source variables is forbidden. This interface proof does
not validate nonlinear neural continuation or all possible learned mappings.
Full multi-step circuit-to-rule abstraction remains an original R5 obligation.

## SCM and statistical guarantees

For a correct common DAG and coupled exogenous laws, let eta_j bound local
mechanism error, nu_j bound E|U_j-V_j|, and L_ij bound coordinate sensitivity.
The conditional certificate checks, in topological order,

    e_j = eta_j + nu_j + sum_i L_ij e_i,

with e_j=0 for coordinates clamped to identical intervention values. The sum of
coordinate errors bounds joint Wasserstein-1 under the l1 metric. The certificate
proves the conditional induction arithmetic; supplied parent correctness, local
domain coverage, Lipschitz bounds, product noise laws and coupling distances
remain separate premises. Empirical independent residuals are not evidence that
true noises are independent or that intervention distributions are recovered.

Hoeffding certificates use independent bounded world units and Bonferroni
family correction, with rational log/square-root enclosures. They are conditional
on fixed hypotheses, independent sampling and declared mixture weights.
Repeated intervention pairs or edges from a single world are not independent
units. Unbounded NMSE/MAE metrics do not automatically receive Hoeffding bounds.

## Population identification and finite-sample recovery

The [Peters et al. additive-noise theory](https://jmlr.org/papers/volume15/peters14a/peters14a.pdf)
uses restricted additive-noise models and identification assumptions. Its
population result and oracle regression/independence argument are existing
foundations, not a project novelty or a finite-sample certificate for a trained
network. The linear Gaussian family above is an explicit excluded obstruction.
Acyclicity and causal sufficiency alone do not restore its direction.

A finite-sample recovery contract needs a correctly specified model family,
independent sampling, a positive separation from alternatives, noise/tail
control, and uniform error control of the regression and dependence scores.
Smoothness, support coverage and complexity restrictions are needed for the
chosen nonparametric score estimator. Near-zero edge effects or nearly
reversible models cannot silently receive a uniform sample-size guarantee.

For a simple conditional example, freeze K candidate hypotheses and bounded
per-row losses in [-B,B] before evaluating n independent rows. Suppose the true
candidate is present and every alternative has population loss at least gamma
above it. Hoeffding and a union bound give

    Pr(max_g |empirical_loss(g)-population_loss(g)| >= gamma/2)
      <= 2 K exp(-n gamma^2/(8 B^2)).

Thus a strictly larger n than (8 B^2/gamma^2) log(2K/delta) gives recovery
probability at least 1-delta under those premises. With nuisance-score errors bounded by r, require the total estimation
error r+t to be strictly less than gamma/2 and allocate confidence to both
events. For example, r < gamma/4 and t < gamma/4 suffice; the sampling
bound then uses n > (32 B^2/gamma^2) log(2K/delta_s), with sampling
failure budget delta_s and nuisance failure budget delta_r satisfying
delta_s+delta_r <= delta. This argument proves neither bounded losses
nor a positive margin for the present learned pipeline. These premises remain
unresolved, so 96 discovery samples do not receive an invented recovery bound.
Held-out world-level confidence bounds measure benchmark performance rather
than individual-dataset causal identifiability.

## Explicit abstraction and intervention commutation

Following the existing [causal abstraction framework](https://arxiv.org/abs/2106.02997),
freeze a state map tau, an input map alpha and an allowed intervention map omega.
For neural execution N and program execution P, exact deterministic abstraction
on declared domain D and intervention family I requires

    tau(N_i(x)) = P_omega(i)(alpha(x))    for every x in D and i in I.

Numeric abstraction replaces equality by a declared metric bound epsilon;
classification requires identical labels. The map omega must be defined for
every admitted compatible intervention combination. For interchange sources,
quantify over the entire admitted tuple of independently chosen source states,
not just a common source that happens to cancel leakage.

For multiple steps require local relations

    tau_(t+1)(N_(t,i)(h)) = P_(t,omega(i))(tau_t(h))

on every reachable state and executed branch, with compatible input/output maps.
Induction composes exact relations. Approximate local errors eta_t compose as
E_(t+1) <= L_t E_t + eta_t only when the program transition has a proved L_t
bound on the covered states; discrete guard agreement is a separate obligation.
Unvisited branch variables cannot count as intervention coverage.

A statistical abstraction instead states, for an explicitly frozen sampling law
mu over base/source/intervention tuples, that Pr_mu(commutation error <= epsilon)
>= 1-delta. Independent-world measurements and familywise bounds can support
this statement conditional on the sampling premises. They do not imply the
universal deterministic relation or correctness of the real-world causal SCM.
