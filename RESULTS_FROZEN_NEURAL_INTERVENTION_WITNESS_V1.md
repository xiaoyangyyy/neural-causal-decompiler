# Frozen neural mechanism: certified intervention fidelity obstruction

This is a strict result for one frozen three-node neural mechanism and true
world. It does not prove that every neural network fails, refute every
candidate SCM, or close original R10. The original atomic ledger remains
0 proved, 2 refuted, 36 unresolved.

The archived world and checkpoint are in
validation/continuous_noise_posthoc_package_v1/. The true node-2 mechanism
has only parent X0, observed scale one, and deterministic response c*X0,
where the binary64 value loaded from its JSON coefficient is approximately
c=1.0054647945184059. The frozen learned node-2 mechanism is the actual
two-Tanh, width-48 NeuralMechanism checkpoint; its input and output
normalizers are included. The checkpoint SHA-256 is
81802ba722f173f35594d02f738e78ba8975284a12099af25bbb5b1a73f146eb.
The true world SHA-256 is
0ad38c1b7844caca150f6c38e90b4ccd08d8a3ee9421ddba944b3d9757a192cf.

For do(X0=-1) and do(X0=1), an independent interval evaluator extracts every
binary32 weight as an exact rational. It propagates intervals through both
affine/Tanh layers and the frozen target normalizer. For each nonnegative
Tanh exponent argument, it bounds exp(z) by the degree-64 Taylor sum and a
geometric majorant of the remaining tail. Negative arguments use oddness.
All intermediate affine and activation intervals are rounded outward to
a 10^-9 rational grid. The checker rejects a changed checkpoint, world,
forward-source hash, layer shape, or unsupported exponent range.

| Quantity | Certified enclosure, displayed approximately |
|---|---:|
| Neural output at X0=-1 | [-0.680513831, -0.680513825] |
| Neural output at X0=1 | [0.673936648, 0.673936655] |
| Neural paired response contrast | [1.354450473, 1.354450486] |
| True minus neural contrast | [0.6564791030, 0.6564791161] |

The true paired contrast is about 2.0109295890, so the error is strictly
greater than 13/20 = 0.65. Even pointwise, the difference between frozen
neural and true deterministic responses is strictly greater than 0.32 at
each intervention value. Dividing by the checkpoint's frozen target
training standard deviation gives strict normalized lower bounds 0.654977
at X0=-1 and 0.668234 at X0=1.

Consequently, on either point no program P can be within 0.01 normalized
output error of both this frozen neural mechanism and the true mechanism:
the triangle inequality would bound their difference by 0.02, contrary to
the certified lower bound above 0.6. This is a scoped incompatibility
between neural fidelity and true causal correctness for this checkpoint.
It does not imply that another trained network, intervention-aware
training, or a different original program family has the same failure.

The accompanying symbolic SCM has a separate strict contrast error above
0.528; see RESULTS_LINEAR_TANH_INTERVENTION_WITNESS_V1.md. Its symbolic
expression is numerically closer to the true intervention contrast than
the neural teacher, but neither satisfies the original causal goal on
this instance. Changing only the noise law cannot correct a deterministic
paired contrast. The next causal recovery attempt must address frozen
mechanism OOD fidelity before treating symbolic extraction or noise
calibration as sufficient.

The JSON certificate stores exact rational interval endpoints and
identities. Source tests cover rational enclosure, outward rounding,
weight tampering and certificate tampering. Verify with:

    validation/wheel_v65_env/Scripts/python.exe -I -B validation/frozen_neural_intervention_witness_v1.py verify --certificate validation/frozen_neural_intervention_witness_v1.json

This theorem uses ideal real arithmetic with the exact loaded binary64
world coefficient and exact checkpoint binary32 weights. It does not claim
bitwise PyTorch or GPU device semantics, finite-sample generalization, or
universal impossibility for other frozen networks.