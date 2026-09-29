# The 27,216-state grid is optimal within its exact proof class

For the frozen trained affine 128D network, **27,216 states is the
smallest count certified by any uniform coordinate-grid abstraction
satisfying the existing exact weighted-verifier inequalities** at
`epsilon=17/100`. This covers all integer bin allocations across all
128 state axes, all positive integer action-bin counts, and all positive
relation radii in that verifier. It upgrades the 0.47 bounded search
result from a good candidate to an exact optimum **within this class**.

The lower proof uses a ten-step finite Neumann sum of exact sensitivity
matrices. Three directly observed axes each need at least six bins.
If any other axis beyond `(0,1,2,3,127)` is refined, a two-axis AM-GM
bound requires at least `63*2*6^3=27,216` states. If none is refined,
125 exact integer cases exclude every `n_0*n_127<=125`; the weakest
case `(15,8)` still misses the tolerance by approximately 0.00103343.
The 0.47 exact upper certificate attains `126*6^3=27,216` states.

Evidence: [method](docs/WEIGHTED_GRID_OPTIMALITY_METHOD.md),
[exact class-optimality certificate](runs/weighted_grid_optimality_v1/certificate.json),
and [acceptance](validation/weighted_grid_optimality_acceptance.json).
Reproduce with:

```powershell
python -m scripts.acceptance_weighted_grid_optimality --verify
```

The general finite-state minimum remains in **[135,27,216]**. The
lower proof here applies only to this weighted uniform-grid certificate
family. It neither rules out smaller non-grid realizations nor proves a
minimum causal quotient or solves the separately trained nonlinear
128D models.


Release 0.48.0 passed all 205 regression tests. An isolated installation
of the built wheel replayed the frozen model, 135-state lower proof,
27,216-state upper proof, and this class-optimality certificate. All
108 installed Python modules matched source bytes. Evidence:
[wheel status](validation/wheel_v48_run/status.json) and
[Junit report](validation/pytest_v48.xml). Wheel SHA-256:
`f7ae7adc341ae82b16e3893f7e5d33b7bdfcb01f65ec7a1a69daacd165431101`.
