# Exact joint-Gaussian coupling for conditional SCM error bounds

The existing ncd.scm_error_certificate propagates supplied local mechanism
errors and supplied noise distances along a DAG. It explicitly leaves the
noise coupling unverified. This v1 certificate verifies a continuous joint
noise coupling for declared affine Gaussian laws; it does not infer either
law from data.

Let Z have independent standard-normal coordinates and define true and
candidate exogenous vectors U = mu + A Z and V = nu + B Z. A and B may have
off-diagonal effects. U and V can therefore each have cross-node
correlation. Their covariance matrices are respectively A A-transpose and
B B-transpose, recomputed exactly from the submitted rational matrices.
Using the same Z constructs a valid coupling of the two declared joint laws.

For each node j, the verifier requires a nonnegative rational s_j whose
square is at least the sum of squared entries in row j of A-B. The triangle
inequality and Cauchy-Schwarz give

    E|U_j - V_j| <= |mu_j - nu_j| + s_j = d_j.

No product assumption on U or V is needed. This proves a noise-side
coordinate distance bound for the declared laws. The law identities
themselves are declarations, not recovered facts about an unknown SCM.

Suppose both SCMs have the same correct DAG, additive unit-gain noise and
identical do values. At every common parent state in a domain containing
both coupled executions, suppose the true structural function f_j and
candidate function g_j differ by at most eta_j. Suppose g_j has
coordinatewise Lipschitz constants L_ij on that domain. In topological
order, a directly intervened node has paired error e_j = 0; otherwise

    e_j = eta_j + d_j + sum_{i parent of j} L_ij e_i.

The construction is an upper bound on expected coordinate error. Because
the paired full outcome vectors are a coupling of the two intervention
distributions, their joint Wasserstein-1 distance under l1 cost is at most
the sum of these e_j. Correlation among exogenous coordinates does not
invalidate the pointwise triangle inequalities or the shared coupling.

ncd.gaussian_joint_scm_certificate uses exact rational arithmetic to check
the DAG, matrix dimensions, covariance entries, Gaussian row norm bounds,
intervention masks and recurrence. Its verifier independently recomputes the
entire certificate and rejects changed scientific conclusions. A two-node
example has true covariance [[1,1],[1,2]] and candidate covariance identity;
the certified noise coordinate bounds are [0,1]. With local errors
[1/10,1/20] and candidate parent Lipschitz bound 2, the joint intervention
distribution bound is 27/20 without intervention and 21/20 after do(node 0
= 2). Ten adversarial and exact-arithmetic tests pass.

The graph correctness, local mechanism errors, Lipschitz bounds, domain
coverage, and whether the declared Gaussian laws are the actual true and
recovered noises remain independent unproved premises. This is a
conditional continuous-distribution theorem and a verified noise coupling
for its declared family. It does not close R10.noise,
R10.intervention_distribution, or any original R0-R13 atomic claim.

The portable example is
validation/gaussian_joint_scm_certificate_v1.json. The separate standard
library checker
validation/check_gaussian_joint_scm_certificate_v1.py recomputes the
covariances, row norm inequalities, noise transport costs, topological
order and intervention recurrence without importing the producer module.
Its sealed receipt is validation/gaussian_joint_scm_verification_v1.json;
the certificate, producer source and checker source SHA-256 values are
recorded there. Both the producer's internal verifier and the separate
checker replay successfully. Twenty targeted tests pass, including
changes to the covariance, coupling bound, local intervention error,
theorem assumptions and original-claim scope.

The independent checker also replays generated rational certificates for
three-, five- and eight-node DAGs with correlated noise and simultaneous
interventions at the first and last nodes. These are mathematical scale
checks of the theorem kernel, not trained-network or independent-world
confirmation results.
