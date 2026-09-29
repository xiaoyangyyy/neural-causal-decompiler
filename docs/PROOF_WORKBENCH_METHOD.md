# Internal-information and exact-semantics proof workbench

Four additional scoped records are independently verified from the installed
`ncd-proof-workbench` 0.1.0 wheel. Nine source and installed tests pass. Original
closure remains two refuted claims and 36 unresolved claims. The bounded-MDL
example is a verifier demonstration, not a theorem of minimum recovery for the
actual full discovery network. The root nonuniqueness example does bind an actual
historical neural checkpoint and covers its full declared input box.

## Post-normalization information loss

For the actual historical teacher in `runs/oblique_seed1193`, translation of one
input column changes its raw mean but leaves centered rows and population
standard deviations exactly equal. The encoder inputs, pooled first/second
moments, log scales, and both swapped computations agree. Therefore the entire
post-normalization trace agrees for the translated datasets. This covers all
three existing cut sites, not just one fitted linear probe.

The actual raw-program mean group is
`group/9c4989da4a99dfc7`; its frozen probe variance is exactly the binary64 value
of 0.29304352025615177. The checker reconstructs the occurrence group from the
bound Rule and verifies that every member is mean(raw X). All these occurrences
are executed in the eager raw feature stage, before the Rule branches. No unused
branch variable is counted as successful intervention coverage.

For two admissible 96-row datasets differing by a unit column translation, every
function of the identical post-normalization state must make at least one raw
mean error of 1/2. Divide by the frozen fit standard deviation to obtain the
certificate's normalized lower bound. This excludes the entire family of
post-normalization readout functions on that two-input domain; it is not the
failure of one trained mapping. It does not exclude access to preprocessing
mean nodes or raw inputs, recovery of other variables, or alternative programs
which omit a redundant mean computation.

There is also a statistical obstruction. Take iid rows of the explicit Gaussian
SCM X=U0, Y=X/2+U1 with independent standard Gaussian noises. For any coordinate
of the sample mean vector and any centered row coordinate,

    Cov(mean(X), X_r-mean(X)) = 1/n - 1/n = 0.

The corresponding cross-coordinate covariances cancel in the same way. All
these variables are jointly Gaussian, hence the sample mean vector is independent
of the complete centered data matrix. Every cut state is a function of that
matrix; retaining log standard deviations does not restore the mean. The X mean
is N(0,1/n). For any predicted value conditional on a cut state, the probability
of lying within epsilon*fit_std is maximized at zero. Independent randomization
cannot increase that bound. Thus

    P(normalized raw-mean error <= epsilon)
        <= erf(epsilon * sqrt(n*fit_variance/2)),
    normalized MSE >= 1/(n*fit_variance).

At n=96 and epsilon=0.01 the success bound is approximately 0.042299784, far below
a 99 percent success claim. These are analytic bounds under a specified law,
not confidence intervals estimated by treating repeated interventions as
independent worlds. Device rounding is excluded: minute floating point changes
are not assumed exactly translation invariant for arbitrary bit-sensitive maps.
Full original R5 remains open because it allows other intermediates and mappings.

## Strict counterexample to legacy decimal conversion

Legacy `ncd.cdir.to_sympy` converts a float with Rational(str(value)). The fidelity
backend instead uses the exact rational value of the binary64 constant. These
semantics disagree. The legacy checker calls this expression zero:

    ((constant(0.1)+constant(0.2)-constant(0.3))*constant(2**60))*x0.

Using exact binary64 constant values, the difference in the parentheses is
1/2**55 and the expression is 32*x0. At x0=1 it differs from zero by exactly 32.
Only the purported exact polynomial subset is used; protected division and
transcendentals are absent. This refutes soundness of that concrete checker,
not decidability of every program class. Mathematical exact arithmetic and
device arithmetic are explicitly separate.

The replacement `canonical_polynomial` expands add/sub/mul/square using reduced
rational coefficients and sorted exponent tuples. Each input constant comes from
`Node.from_dict(...).value`, preserving the actual binary64 value. It proves
mathematical all-real function equality in this subset. Operators outside it are
rejected. A polynomial's unique coefficient representation does not prove equal
intervention-bearing internal computations. The pending core fix is saved in
`validation/pending_core_cdir_binary_semantics.patch`; the running confirmation
still pins the original 0.59 source and the fix has not been applied there.

## Finite MDL and uniqueness

The grammar freezes variable count, a finite set of exactly representable
constants, operator set and maximum unit AST-node cost. Enumeration considers
all ordered ASTs, including nested squares, and hashes every syntax/normal-form
pair. The independent checker regenerates the enumerated prefix and determines
whether it exhausted the frozen bound. Budget exhaustion stays unresolved even
if a faithful candidate was found; only its current upper bound is reported.

For the frozen two-variable grammar with constant 0, add/sub/mul/square and cost
at most 3, all 36 expressions are checked. The exact target x0+x1 has minimum
cost 3 and two optimal syntaxes, x0+x1 and x1+x0. Their canonical polynomial
representation agrees. Syntax uniqueness and function equality are separate.
This small exact-target example verifies the backend; it is not minimum
recovery of a historical discovery network.

For the actual historical zero-parent mechanism 1, the checker exports its
frozen Tanh network and constructs two distinct binary64 constants within half
the declared normalized tolerance of its output. Independent interval checks
prove both constants faithful throughout [-1,1]^3 at epsilon=0.01. Each has cost
one in the frozen grammar; no zero-cost expression exists. They are shortest and
are different mathematical functions. Therefore approximate fidelity alone does
not imply a unique shortest semantic realization. This does not refute exact
minimal realization or close original R12.

## Replay and history

```powershell
python -m proof_workbench prove --config validation/original_proof_workbench_protocol_v1.json
python -m proof_workbench verify-bundle runs/original_proof_workbench_bundle_v1/manifest.json
```

Every job binds its inputs and target scope; the bundle binds source, protocol,
certificates and the original ledger. `validation/proof_workbench_installed_v1`
contains independent installed evidence and explicit forgery rejection.
`validation/source_snapshot_059` retains byte copies of the original core and
proof packages plus a junction to retained experiment artifacts. Its original
protocol source check passes. This preserves an old-version replay route when
the main source is repaired; historical successes and failures remain recorded.

The next useful positive R5 proof is a composition certificate for the actual
neural preprocessing mean/std/normalization computations, including compatible
independent-source interventions, degenerate inputs and clipping boundaries.
It must trace the existing network rather than insert a surrogate. The actual
nonlinear continuation and full discovery program remain separate obligations.
