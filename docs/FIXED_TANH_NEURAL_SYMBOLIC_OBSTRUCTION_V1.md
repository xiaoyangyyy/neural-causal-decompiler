# Fixed Tanh network versus symbolic program, version 1

This certificate compares one archived symbolic child mechanism directly with
its **frozen neural teacher**. The true SCM equation and exogenous law are
not needed for the mathematical mismatch. It is a candidate-specific,
same-world development result, not a refutation of every program or a
closure of the original R10.mechanism atom.

The frozen child-1 checkpoint records parents (0,2), width 48, finite
binary32 parameters, positive input standard deviations and a positive
output training scale. Its pinned source implements

    affine(2,48) → tanh → affine(48,48) → tanh → affine(48,1),
    then output = yscale × network + ymean.

Under *real arithmetic* with the stored binary32 numbers treated as
exact rationals and real tanh, every last hidden coordinate has magnitude
at most 1. Thus the network output has a global absolute bound

    B = |ymean| + yscale ×
        (|last_bias| + sum_i |last_weight_i|)
      = 85170555198899581 / 36028797018963968
      ≈ 2.3639577850481537.

The archived symbolic child equation contains a nonzero x0*x2 term.
Along x0=x2=t, independent exact AST calculations show a nonzero
quadratic coefficient. Therefore the absolute network-symbolic error is
unbounded on the all-real parent domain, because the neural output is
bounded and the symbolic output grows quadratically.

At the finite parent setting x0=x2=2, the symbolic value is exactly

    -138715334957337881 / 36028797018963968
    ≈ -3.8501239684556845.

The reverse triangle inequality gives a network-symbolic absolute error
of at least

    |symbolic| - B =
    13386194939609575 / 9007199254740992
    ≈ 1.4861661834075308.

The saved frozen training output scale is exactly 5850101/8388608
(about 0.6973863840), so the normalized error is strictly at least
13386194939609575/6281498118324224, about 2.1310513332. This
exceeds the 1/100 training-scale tolerance for this exact parent
setting under the stated real semantics.

The primary certificate and checker are
`validation/fixed_tanh_neural_symbolic_obstruction_v1.json` and
`validation/fixed_tanh_neural_symbolic_obstruction_v1.py`. The
independent verifier
`validation/check_fixed_tanh_neural_symbolic_obstruction_v1.py`
does not import the generator: it verifies the pinned original bundle
and source hashes, loads the frozen checkpoint, checks the Tanh
architecture, recomputes the output bound with a separate rational
conversion, and reconstructs the symbolic quadratic from values at
-1, 0 and 1. Its sealed receipt is
`validation/fixed_tanh_neural_symbolic_obstruction_verification_v1.json`.
Targeted tests reject changed numerical claims, scope inflation and
rehashed archived model mutations.

The proof covers the real-valued mathematical network corresponding to
the saved binary32 weights. It does not prove a roundoff enclosure for
arbitrary PyTorch float32 execution. The historical evaluation
interventions did not establish that (2,2) belongs to the original
declared intervention family, so this finite witness does not by itself
close the original R10 claim. The all-real obstruction applies only
where that all-real domain is actually claimed. Different symbolic
programs and bounded domains may admit valid fidelity certificates.
