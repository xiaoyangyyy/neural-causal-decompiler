# Integer-grid refinement reduces the trained affine 128D upper bound

For the same frozen trained affine 128-dimensional ReLU ring and
`epsilon=binary64(0.17)`, the certified minimum finite-state interval
is now **[135, 27,216]**. The 135-state lower certificate is unchanged.
A new exact rational construction lowers the prior 40,824-state upper
bound by 13,608 states, exactly one third. It covers all unit-cube
initial states, every continuous unit-cube action sequence, and every
finite horizon.

The search derived an absolute influence matrix from the frozen network,
enumerated bounded integer bin choices, and proposed state-grid radii.
An independent exact-rational checker then verified the full-domain
invariant and observation/transition inequalities from the serialized
network. Numerical scoring is not part of the proof.

| Construction | Bins on coordinates `(0,1,2,3,127)` | Action bins per coordinate | Abstract states |
|---|---|---:|---:|
| Historical baseline | `(18,6,7,6,9)` | 128 | 40,824 |
| State-only refinement | `(16,6,6,6,8)` | 128 | 27,648 |
| State/action refinement | `(14,6,6,6,9)` | 512 | 27,216 |

The remaining 123 state coordinates use one bin each. The action
partition is implicit and does not enumerate `512^5` cells.
The upper certificate uses exact rational tolerance `17/100`; this
is slightly stricter than the lower certificate's exact binary64
interpretation of `0.17`, so both bounds apply at the latter tolerance.
The acceptance script replays the model, historical upper, lower packing,
and both new upper certificates together.

Evidence: [method](docs/INTEGER_GRID_REFINEMENT_METHOD.md),
[acceptance](validation/integer_grid_refinement_acceptance.json),
[state/action proposal](runs/integer_grid_refinement_v1/state_action/proposal.json),
and [exact upper certificate](runs/integer_grid_refinement_v1/state_action/certificate.json).
Reproduce with:

```powershell
python -m scripts.acceptance_integer_grid_refinement --verify
```

The bounded grid search does not prove that 27,216 is the smallest
coordinate-grid construction or the globally smallest finite
realization. The separately trained nonlinear 128D networks are not
covered by this upper certificate.


Release 0.47.0 passed all 203 regression tests. An isolated installation
of the built wheel replayed the historical upper, dynamic lower, and both
new upper certificates; all 107 installed Python modules matched source
bytes. Evidence: [wheel status](validation/wheel_v47_run/status.json) and
[Junit report](validation/pytest_v47.xml). Wheel SHA-256:
`858d2f9f74651c03de3c5643a79bc09519cdc7073480d7a1e92ad0017855369b`.
