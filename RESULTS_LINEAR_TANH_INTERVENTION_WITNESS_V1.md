# Fixed recovered SCM: certified intervention-contrast counterexample

This is a strict counterexample for one frozen recovered explicit SCM, not a
refutation of all candidate programs or closure of original R10. The original
38-claim ledger remains 0 proved, 2 refuted, 36 unresolved.

The world and recovered SCM are the exact archived files in
validation/continuous_noise_posthoc_package_v1/. The true three-node world has
observed scale one, node 2 has only parent 0, and its deterministic mechanism
is c*x0 with serialized decimal coefficient
c = 1.0054647945184059. The recovered explicit SCM has the same declared
node-2 parent but executes a + b*tanh(x0), where
a = 0.0005967716691673165 and b = 0.9735387410177686.

Compare paired surgical interventions do(X0=-1) and do(X0=1). In each SCM,
hold its own node-2 exogenous value fixed across the two interventions. The
additive noise and constant offset cancel *pathwise*. The true node-2 response
contrast is 2*c; the recovered SCM contrast is 2*b*tanh(1). This argument is
independent of the selected additive noise family or its fitted parameters.
It requires the same exogenous law and coupling across interventions, as in
the declared SCM semantics. It does not compare the true and recovered noise
draws to each other.

The checker interprets serialized decimal coefficients as exact rationals.
It bounds exp(2) with the rational Taylor sum through degree 16. The first
omitted term is 2^17/17!, and every later term ratio is at most 2/18, so the
remaining tail is at most that first term divided by (1-2/18). Because
tanh(1)=(exp(2)-1)/(exp(2)+1) and that transform is increasing, it yields a
strict rational interval for tanh(1). All subsequent arithmetic is exact
Fraction arithmetic. The resulting decimal displays below are only for
readability; the JSON certificate stores the full rational bounds:

- True contrast: about 2.0109295890368117.
- Recovered contrast: strictly inside
  [1.4828828314884100, 1.4828828315113494].
- True minus recovered contrast: strictly inside
  [0.5280467575254625, 0.5280467575484018].

The lower endpoint is greater than 1/2 by exact rational comparison. Thus
the fixed recovered SCM fails the scoped claim that its node-2 paired
response contrast is within absolute error 1/2 on these two interventions.
The certificate binds the archived world SHA-256
0ad38c1b7844caca150f6c38e90b4ccd08d8a3ee9421ddba944b3d9757a192cf
and explicit SCM SHA-256
f58a47e7db0128aeaf50c810da951e624a26af96fabee7909f00def842f1f691.

The independent checker reads both input JSON files, validates the two
equation shapes, graph parents, scale and Gaussian true-noise premise, and
recalculates every rational bound. Source tests reject a changed equation
and a tampered certificate. Reproduce with:

    validation/wheel_v65_env/Scripts/python.exe -I -B validation/linear_tanh_intervention_witness_v1.py verify --certificate validation/linear_tanh_intervention_witness_v1.json

The theorem is about ideal real-valued semantics of the serialized decimal
coefficients. It is not a proof of exact floating-device execution or a
claim about an unknown neural mechanism outside these frozen files. A future
R10 method must recover the correct intervention-scale mechanism, or report
that such fidelity cannot be certified; changing only its noise law cannot
repair this fixed candidate's response contrast.