# All-horizon behavioral cover and packing match at 162

For the frozen trained affine 128D ReLU ring at
`epsilon=binary64(0.17)`, the minimum number of **concrete initial
trajectory representatives** needed to approximate every initial
state under every common continuous action word and at every time
is **exactly 162**.

An explicit `3^4*2` representative set uses observed-coordinate
centers `1/6,1/2,5/6`, feedback-coordinate centers `1/4,3/4`,
and `1/2` on the remaining 123 coordinates. Exact rational
inequalities bound the initial, first-step and invariant future
errors. The prior 162-point all-horizon packing is a matching lower
bound for any behavioral cover. All four frozen affine dimensions
8, 32, 64 and 128 replay.

Evidence: [method](docs/ALL_HORIZON_BEHAVIORAL_COVER_METHOD.md),
[acceptance](validation/behavioral_cover_acceptance.json),
and [128D exact certificate](runs/behavioral_cover_v1/affine_d128/certificate.json).
Reproduce with:

```powershell
python -m scripts.acceptance_behavioral_cover --verify
```

This closes a passive behavioral complexity measure, **not** the
minimum deterministic finite-state realization. The representative
trajectories do not supply a transition-closed 162-state machine.
The general finite-machine interval remains **[162,27,216]**.
The gap is now specifically about enforcing one deterministic
transition for all concrete states represented by the same abstract
state under each action, rather than about finding more pairwise
distinguishable trajectories.


Release 0.50.0 passed all 207 regression tests. An isolated installed
wheel replayed the matching packing and covering proofs on all four
frozen affine dimensions, plus the same-model 27,216-state machine
upper. All 110 installed Python modules matched source bytes.
Evidence: [wheel status](validation/wheel_v50_run/status.json) and
[Junit report](validation/pytest_v50.xml). Wheel SHA-256:
`3c23d6a327d8b16c76da14a5a1b4a0c5bfc5302f7cb35c586c0c1be592699506`.
