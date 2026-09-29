# Reachability-aware two-stage machine lowers the 128D upper to 4,723

An executable, exact-certified finite realization for the frozen
trained affine 128D ReLU ring now has **4,723 states** at
`epsilon=17/100`, compared with the previous 27,216-state upper.
The reduction is approximately **82.65%**. With the independent
162-state lower certificate at binary64 `epsilon=0.17`, the
same-model minimum-state interval is **[162,4,723]**.

The machine starts in one of 243 coarse states covering the entire
unit initial-state cube. Every action moves it into a tagged
recurrent phase. The previously certified weighted grid contributes
only 4,480 cells that intersect the proved transition-image box.
Exact-rational inequalities verify initial output error, the
initial-to-recurrent handoff, closure of the recurrent index set,
and the recurrent simulation for every continuous action sequence
and all time. The handoff's narrowest coordinate slack is about
0.000476. The implementation exposes actual initial, output and
transition functions; its action partition is implicit.

| Part | Abstract states |
|---|---:|
| Full-cube initial grid | 243 |
| Transition-image-box recurrent grid | 4,480 |
| **Total** | **4,723** |

Evidence: [method](docs/REACHABLE_TWO_STAGE_METHOD.md),
[acceptance](validation/reachable_two_stage_acceptance.json),
and [exact 128D certificate](runs/reachable_two_stage_v1/affine_d128/certificate.json).
Reproduce with:

```powershell
python -m scripts.acceptance_reachable_two_stage --verify
```

This construction crosses the previous uniform-grid optimality
boundary by using separate initial and recurrent state sets and
pruning unreachable recurrent indices. It does not prove that
4,723 is globally minimal. The trained nonlinear 128D networks
require their own certificates.


Release 0.51.0 passed all 208 regression tests. An isolated
installation of the built wheel replayed the matching 162-state
lower evidence, the recurrent exact upper, the 4,723-state two-stage
certificate, and executable action traces. All 111 installed Python
modules matched source bytes. Evidence:
[wheel status](validation/wheel_v51_run/status.json) and
[Junit report](validation/pytest_v51.xml). Wheel SHA-256:
`ef32785ee9238486f55ac2ec6d75d4cd07f0167d7992fe8d73bd1177c8bad461`.
