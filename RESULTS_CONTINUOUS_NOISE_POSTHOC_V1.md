# Continuous noise post-hoc diagnostic, version 1

This is a development diagnostic on one historical confirmation world after its
results were already known. It is **not** an independent 8101/8102 confirmation
experiment, a 99% statistical guarantee, or closure of original R10. The
original atomic ledger remains 0 proved, 2 refuted, 36 unresolved.

The read-only input is seed_8101_n3_test_id_0, a three-node linear-Gaussian
world. The candidate uses the existing observational-graph branch and frozen
neural checkpoints; no oracle graph or true equation is used in selection.
Its noise fit uses residuals against the actual executable symbolic equations.
The true world is read only by the evaluator. Separate pseudorandom samples
of 512 rows each use seeds 9100 (fit), 9101 (family selection) and 9102
(evaluation). They come from the **same world**, so there is one world and no independent
confirmation replication for population inference.

The estimated source graph has zero edge errors on this instance. All three
selected marginal laws are Gaussian. Values below are empirical quantities in
observed-coordinate units, not confidence bounds.

| Intervention | Continuous mean marginal W1 | Empirical mean marginal W1 | Continuous joint energy | Empirical joint energy |
|---|---:|---:|---:|---:|
| None | 0.0410 | 0.0442 | 0.0041 | 0.0050 |
| do(X0=-1) | 0.2261 | 0.1791 | 0.2261 | 0.1676 |
| do(X0=1) | 0.0955 | 0.1409 | 0.0647 | 0.1016 |
| do(X1=1) | 0.0215 | 0.0295 | 0.0028 | 0.0043 |
| do(X0=1,X1=-1) | 0.0751 | 0.0872 | 0.0751 | 0.1095 |

In do(X0=-1), the local symbolic mechanism MAE at true parent values is
0.1577 at node 1 and 0.2646 at node 2. In do(X0=1), it is 0.1053 and
0.2634. These are evaluator-only diagnostics; the fitted candidate never
reads the true equations or exogenous draws. They show why changing only the
noise law cannot repair intervention fidelity on this instance. The
continuous candidate is better on four of five listed empirical distances and
worse on one; there is no uncertainty analysis or independent-world
replication, so this is not a general improvement claim.

The historical finite-residual SCM has exact population TV distance 1 from
the continuous true SCM under interventions leaving a free node, as separately
proved in RESULTS_SCM_POPULATION_PROOF_V1.md. A finite-sample W1 or energy
comparison can favor that empirical model without contradicting the TV proof.
The new continuous representation removes the finite-support obstruction as
a representation choice, but joint noise independence, true noise-law
distance, graph correctness in other worlds, and intervention distribution
recovery remain unproved.

Portable input and result: validation/continuous_noise_posthoc_package_v1/.
Its manifest binds the frozen world, three neural checkpoints, executable
symbolic SCM, result, and the relevant source files. The result SHA-256 is
e2b7fced0beaab6b01bdb4182a752e08826706795f97f0831b195985b42929f6.
In the current environment the package reproduced the result byte for byte.
The replay checker rejects a tampered checkpoint. Run:

    validation/wheel_v65_env/Scripts/python.exe -I -B validation/verify_continuous_noise_posthoc_v1.py validation/continuous_noise_posthoc_package_v1

The checker establishes local integrity and deterministic replay, not an
independent causal or statistical proof. The next R10 step must improve the
symbolic mechanism under intervention shifts and evaluate the learned joint
noise law on newly frozen, independent worlds, with a valid distribution
distance guarantee.