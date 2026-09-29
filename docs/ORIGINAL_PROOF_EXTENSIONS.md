# Original-proof extensions: scope, counterexamples and replay

This supplement adds evidence without changing the 38 original atomic claims.
The original ledger still contains two refuted boundary claims and 36 unresolved
claims. The five new conclusions below are scoped records, not original closure.
Core 0.59 and the frozen 300-world protocol remain unchanged. Extensions are
packaged separately as `ncd-proof-extensions` 0.1.0.

## Actual statistical semantics correction

`ncd.discovery_fidelity_proof.raw_feature` in 0.59 used CDIR operations for
historical Rules, although the execution pipeline uses `statistics.extract_one`.
They differ in standardization floors, variance ratios, regression basis,
quantile selection, response scaling and ridge regularization. General Rule
certificates made through that old macro backend must not be assumed valid.
Historical artifacts are retained; a source/semantics audit is saved in
`runs/actual_statistics_rule_revalidation_v1/audit.json`.

The delivered historical local-box certificate visits only feature 7, HSIC.
It has been re-proved using the correct actual-statistics frontend and retains
its original local conclusion. No unvisited feature is counted as covered.
The replacement frontend treats ambiguous sorting, rank-deficient conditioning,
unsupported operators and broad interval cells as unresolved. Mathematical
certificates do not certify device rounding.

## Exact feature-family impossibility for a historical frozen network

Let A have 16 base rows. For X=+s_x, Y has four copies each of +5s_y and -5s_y.
For X=-s_x, Y has one copy each of +/-s_y and +/-7s_y, and two copies each of
+/-5s_y. Repeat each base row six times (96 rows). Let B reflect only X.
The certified instance uses s_x=1/8 and s_y=1/4. A and B are not row permutations.

Both conditional Y distributions have zero mean and second moment 25s_y^2,
since 1^2+7^2=2*5^2. Therefore correlations, skewness and both mixed third moments
vanish; variance ratio and fourth moments agree. RBF HSIC is invariant under
coordinate reflection, including its bandwidth thresholds and denominator floor.

Every distinct row has even multiplicity. Each sorted parity cross-fit fold
contains exactly half of every row multiplicity, in both regression directions
and in both datasets. Training means are zero and variances of the standardized
columns are one. At the actual five quantiles, both predictors have centers
[-1,-1,0,1,1]; the outer quantiles lie on plateaux, so the binary64 quantile
constants do not affect this identity. Under predictor reflection, the 14 basis
columns transform by an orthogonal signed permutation: odd functions change
sign, even functions remain fixed, and RBF columns exchange opposite centers.
The bias column remains fixed and the positive diagonal ridge commutes with this
transformation. The strictly positive definite ridge system has a unique solution.
Predictions of Y are unchanged; reflection of the response X negates predictions
and residuals. Squared residual errors and squared-distance kernels agree.
Consequently all 14 mathematical features agree exactly.

The actual frozen checkpoint is
`runs/all_seed91_final/bivariate/models/full/discoverer.pt`, SHA-256
`84a313ccbeb7f120686547bee9b010a6ef5f686b506ebf9f9b1fa4d6234c89e8`.
Exact coefficient export and rational Tanh/log enclosures certify strict network
labels 1 and 2. Mathematical population means, standard deviations, encoder
averages and squared averages are invariant under identical row replication;
the checker validates the template and evaluates the 16-row base, certifying
the full 96-row input rather than querying an approximate surrogate.

Any deterministic function of these 14 features must return one common label.
Thus no member of that entire family matches this network on both datasets,
regardless of operator grammar, constants or program length. This is a
family-level refutation, not just the failure of one extracted Rule. It does not
exclude raw-data CDIR programs, statistically approximate recovery, or floating
point functions that exploit last-bit feature differences. The observed binary64
feature discrepancy (~1.1e-16) is diagnostic and is explicitly outside the proof.

## Population identification versus finite-sample separation

Compare independent standard Gaussian roots with X0~N(0,1), X1=beta*X0+U1,
where U1~N(0,1) independently. Extra nodes are independent roots. For beta!=0
the population laws differ. Their per-row KL(B||A) is beta^2/2. With n iid rows,
Pinsker and two-point testing give

    TV(A^n,B^n) <= min(1, sqrt(n*beta^2/4))
    inf_estimators max_graph_error >= (1-TV)/2.

At n=96 and beta=1/100 the certified error lower bound exceeds 0.4755. A one
percent uniform recovery promise on this two-model family is impossible.
Benchmarks excluding this coefficient and informative interventions are not
refuted. This is distinct from equal-law observational nonidentifiability.
Separation, controlled nuisance errors, noise tails and a bounded candidate
complexity remain premises requiring evidence for the learned estimator.
The inequalities are established statistical theory, not project innovations:
[John Duchi's notes, sections 2.2.4 and 2.3.1](https://web.stanford.edu/class/stats311/lecture-notes.pdf).
Restricted population assumptions follow the scope of
[Peters et al.](https://jmlr.org/papers/v15/peters14a.html); they do not provide
an automatic finite-sample guarantee for this particular neural pipeline.

## Decoder boundary and observational candidate families

Two exact dyadic, reverse-symmetric probability tensors refute universal node
permutation equivariance of the legacy `decode_graph`: first-argmax direction
ties and confidence ties during cycle projection depend on node indices.
These certify decoder counterexamples. Reachability by a particular trained
network, true causal correctness and the entire original R2 claim remain separate.

`graph_family` preserves all maximizer classes (including interval uncertainty),
retains both orientations of class 3 and returns the lazy family of compatible
acyclic graphs. A forced cycle is an explicit empty-family conflict. Its witness
is only a membership witness. Neither a certified CPDAG nor true-graph coverage
is claimed. Membership is permutation equivariant; a canonical selected witness
need not be equivariant. Historical selected DAGs remain comparison results.

## Composed SCM mechanism and noise error

For correct parents and common do-values, suppose local mechanism errors are
bounded by epsilon_i, parent Lipschitz constants by L_ij, and an explicit joint
noise coupling bounds E|U_i-V_i| by delta_i (unit noise gain). Topological
induction gives

    d_i <= epsilon_i + delta_i + sum_j L_ij*d_j,
    d_i = 0 for a node intervened on with the same fixed value.

The coupling bounds every coordinate W1 and joint W1 with l1 cost. Local noise
marginals alone do not justify a compatible joint coupling, especially with
correlated exogenous noise. Parent errors are not repaired by an oracle graph.

The executable certificate is deliberately scoped to two specified linear SCMs
with identical known coefficients, explicit independent Gaussian product noise
laws, and different intercepts/noise parameters. For common independent Z_i,
U_i=mu_i+sigma_i*Z_i and V_i=m_i+s_i*Z_i; Cauchy-Schwarz gives E|Z_i|<=1, so
delta_i<=|mu_i-m_i|+|sigma_i-s_i|. The checker verifies acyclicity and coefficient
equality, rejects unspecified/correlated noise, and enumerates every intervention
target subset. Bounds hold for all real intervention values. This establishes a
foundational propagation lemma; it does not certify the actual learned SCM's
parent sets, nonlinear uniform errors or fitted residual noise independence.
Intervention commutation remains the causal-abstraction contract described by
[Geiger et al.](https://arxiv.org/abs/2106.02997), rather than an innovation claim.

## Replay

From the project root:

```powershell
python -m proof_extensions prove --config validation/original_proof_extensions_protocol.json
python -m proof_extensions verify-bundle runs/original_proof_extensions_bundle_v1/manifest.json
```

The bundle binds configuration, certificates, source files, checkpoints and the
unchanged original ledger. Single certificates can be checked with
`python -m proof_extensions verify <certificate.json>`.
The checker independently recalculates every critical condition; no search
result or numerical probe is accepted as proof. The root installation has 23
focused tests covering actual statistics, transcendental enclosures, graph
forward semantics, branch/rank boundaries, tampering, candidate sets and SCM
noise/parent/intervention errors. Original 249-test core release evidence is
preserved separately. Independent installed replay is recorded separately.

The full plan is still open: complete raw discovery-domain partitioning,
nontrivial full-domain mechanism recovery, shorter-program enumeration, actual
multi-step internal maps, causal graph identification, nonlinear SCM noise
couplings and original-scope OOD guarantees remain unresolved. A scoped success
or refutation never substitutes for those original quantifiers.
