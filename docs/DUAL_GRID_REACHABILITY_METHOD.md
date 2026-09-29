# Joint state/action grid choice with exact reachable-graph closure

A uniform coordinate-grid certificate gives only an outer bound on the eventual executable graph. Its total number of full-cube cells, or even the coordinatewise transition-image rectangle, need not rank grid choices by the size of their *closed abstract successor graph*. The frozen trained affine 128D ReLU ring provides a concrete example.

The v0.53 machine uses recurrent bins `(16,6,6,6,8)` on active coordinates `(0,1,2,3,127)` and 128 implicit action bins. Its certified coordinatewise transition-image rectangle has 4,480 states, while exact shared-action pairwise support closure retains 899 recurrent states. An independently available weighted-grid certificate instead uses recurrent bins `(14,6,6,6,9)` and 512 implicit action bins. Its coordinatewise rectangle contains **4,992** states, yet its exact pairwise action-closure contains only **798**. Thus the larger rectangle yields the smaller actual certified finite machine. The action bins change the error budget and deterministic action quantization; they do not multiply the abstract state count.

The checker first replays the original state/action joint-grid weighted certificate on the serialized network. It builds and replays a new two-stage certificate: 243 initial states cover the full unit initial cube; exact rational output and transition sensitivity inequalities hand off to the joint recurrent relation under every continuous unit action. It then proves the network transition globally affine by exact ReLU phase bounds. For each active output pair, 60 rational support inequalities on shared action variables test necessary conditions for a proposed target cell, as detailed in [the pairwise method](CORRELATED_REACHABILITY_METHOD.md). Strictly disjoint closed intervals exclude an edge; all other edges remain. This is conservative even if some retained edges are infeasible.

All 243 initial sources produce 6,480 candidate edges, of which 2,051 survive the pairwise filter and lead to 780 distinct recurrent cells. Repeatedly expanding retained recurrent states produces the least closed abstract graph: 14,696 recurrent candidate edges were checked, 4,901 were retained, and 798 recurrent cells remain. The checker recomputes the full ordered set from frozen network weights and verifies every cell lies in the independently certified 4,992-state outer rectangle. The executable `dual_grid_initial`, `dual_grid_output`, and `dual_grid_step` accept only these states.

The previous exact weighted recurrent simulation and the new exact initial handoff establish observation error at most `17/100` for every initial point in the state unit cube, every continuous unit-cube action word and every finite horizon. The machine contains `243+798=1,041` states. The independent 162-state packing certificate uses binary64 `0.17`, slightly looser than rational `17/100`, and gives the same-model general minimum interval **[162,1041]**.

This result is a certified upper bound, not the global minimum. The 798 states form a conservative abstract closure, and the pairwise support filter is not a complete test of five-dimensional action feasibility. Nor does this certify the trained nonlinear 128D networks.

Replay:

```powershell
python -m scripts.acceptance_dual_grid_reachability --verify
```

Evidence: `runs/dual_grid_reachability_v1/affine_d128/two_stage_certificate.json`, `runs/dual_grid_reachability_v1/affine_d128/closure_certificate.json`, `validation/dual_grid_reachability_acceptance.json`, and the existing state/action weighted and packing certificates linked by SHA-256.
