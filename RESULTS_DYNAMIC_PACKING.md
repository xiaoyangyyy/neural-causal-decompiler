# One-step packing improves the trained affine 128D lower bound

For the frozen globally affine 128-dimensional neural system from the
continuous-scale experiment, the certified finite-state interval at
`epsilon=binary64(0.17)` is now **135?40,824**, improving the previous
81-state lower bound while retaining the independently replayed upper
certificate. The claim covers every unit-domain initial state, every
continuous unit-domain action word, and all finite horizons.

The new lower certificate contains 135 rational initial states. A common
all-zero action is enough to distinguish all 9,045 pairs: 8,991 pairs
separate immediately, while 54 require one transition. The exact minimum
pairwise output-trace distance is about 0.345, exceeding the exact
`2*epsilon` threshold by about 0.005. The verifier reads the frozen
network, proves globally fixed ReLU phases for a fast exact affine
calculation, and checks every pair with rational arithmetic. A direct
exact ReLU evaluator remains available for nonlinear witness sets.

The same 135-point proof passes on all four frozen affine ring models of
dimension 8, 32, 64 and 128. The decisive extra distinction comes from
a feedback edge into the first observed coordinate. Deleting that edge
makes this point set fail; the same candidate also fails on two separately
trained nonlinear 128D models. These failures concern this witness set,
not the true minimum state counts of the control models.

Evidence: [method](docs/DYNAMIC_PACKING_METHOD.md),
[acceptance](validation/dynamic_packing_acceptance.json),
[128D packing certificate](runs/dynamic_packing_v1/affine_d128/certificate.json),
and the [historical upper certificate](validation/automatic_affine128_dev/certificate.json).
Reproduce with

```powershell
python -m scripts.acceptance_dynamic_packing --verify
```

The exact behavioral quotient of this model still has dimension 128.
The new 135-state lower bound is for finite observation tolerance and
one-step distinguishability, so it does not equate quotient dimension
with finite-state count. The large upper/lower gap and the nonlinear
trained models remain open.

Release 0.46.0 passed all 202 regression tests. The isolated wheel
replayed all four affine packing certificates, the historical same-model
upper, three counterfactual controls, and the nonlinear exact-evaluation
fallback; all 106 installed modules matched source bytes. Evidence:
[wheel status](validation/wheel_v46_run/status.json). Wheel SHA-256:
`72af6d415bcc9875161a934a1ab45757b41fa042b637ff05ecac39cff863a80b`.
