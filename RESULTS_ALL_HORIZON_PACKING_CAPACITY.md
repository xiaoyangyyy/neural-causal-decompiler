# The affine 128D ring has exact all-horizon packing capacity 162

The frozen trained affine 128D network now has a certified
finite-realization interval **[162,27,216]** at
`epsilon=binary64(0.17)`. A six-pair rational construction improves
the previous lower bound 135 to 162. Its exact verifier checks all
13,041 witness pairs under one zero action.

A separate exact upper theorem proves that 162 is the largest
**pairwise output-trace packing** possible for any unit-domain initial
states, any common continuous action word, and any finite horizon.
The proof partitions the four directly observed initial coordinates
into 81 cells and shows each cell can hold at most two one-step
packing points. An exact forward-invariant difference bound then
shows that a pair unseparated at times 0 and 1 can never separate
later, even though the network's exact quotient has dimension 128.

The result replays on frozen affine rings of dimension 8, 32, 64 and
128. Removing the effective feedback edge makes this particular
six-pair construction fail. These controls identify what the witness
uses; they do not bound the control models' optimum sizes.

Evidence: [method](docs/ALL_HORIZON_PACKING_CAPACITY_METHOD.md),
[acceptance](validation/one_step_capacity_acceptance.json),
[128D packing certificate](runs/one_step_capacity_v1/affine_d128/packing_certificate.json),
and [128D capacity certificate](runs/one_step_capacity_v1/affine_d128/capacity_certificate.json).
Reproduce with:

```powershell
python -m scripts.acceptance_one_step_capacity --verify
```

The exact capacity concerns **pairwise common-action trace packing**.
It does not settle the globally smallest deterministic finite-state
realization. Any further lower bound above 162 must use a stronger
obstruction, such as transition consistency, rather than only
pairwise trace separation. The independently replayed 27,216-state
upper remains valid for the same 128D model.


Release 0.49.0 passed all 206 regression tests. An isolated installed
wheel replayed all four frozen affine dimensions, the matching
packing/capacity certificates, the same-model 27,216-state upper,
and the feedback-edge ablation control. All 109 installed Python
modules matched source bytes. Evidence:
[wheel status](validation/wheel_v49_run/status.json) and
[Junit report](validation/pytest_v49.xml). Wheel SHA-256:
`99c82c2321981f25fd0408a03b5f3278d8bc984d8541196e3600f00eadd56829`.
