# Automatic certified grid synthesis from frozen neural influence

Previous infinite-horizon realizations supplied state-bin counts and
coordinate relation radii by hand. This experiment removes that choice
from the frozen-network workflow. The synthesizer reads only a serialized
ReLU transition and observation network, the unit state/action domains,
epsilon, and a continuous-action partition size. It does not read the
queue simulator, ring-neighbor indices, or a handwritten state-bin vector.

For a ReLU network with weight matrices W_l, multiplication of the
absolute matrices gives a global coordinatewise Lipschitz bound. Partition
the transition bound into state M_x and control M_a; let M_o be the
observation bound. For candidate bins n and action bins A, the least
radius suggested by the linear sufficient condition is

    r = (I - M_x)^(-1) (M_a 1/(2A) + 1/(2n)).

The search uses this equation only when the absolute-influence dynamics
are contractive. Its output sensitivity is
M_o (I - M_x)^(-1). Starting from one bin per coordinate, it increases
the coordinate with the largest reduction in violated output error per
increase in log state count, then prunes redundant increments. A reserved
2% output margin and a small radius inflation absorb numerical proposal
error. The heuristic does not prove an optimal grid.

**The floating-point search is untrusted.** A separate exact-rational
checker re-evaluates the final serialized neural network, the complete
unit-domain transition invariant, initial coverage, observation error,
and every implicit abstract transition. A failed proof returns
unresolved. A passed proof applies to all continuous action sequences
and all time horizons.

The frozen study 'scripts/acceptance_automatic_grid_global.py' covers
21 models: eight learned-local-phase networks, eight fixed-dictionary
trained nonlinear networks, four earlier trained affine networks, and
the constructed coupled phase-crossing 2D benchmark. All 21 proposals
and exact certificates regenerate and replay.

| Frozen family | Cases | Prior supplied upper | Automatic certified upper | Lower |
|---|---:|---:|---:|---:|
| Learned local phase, 8/32/64/128D | 8 | 2250 | 1000-1125 | 81 |
| Fixed-dictionary nonlinear, 8/32/64/128D | 8 | 2250 | 1125-1375 | 81 |
| Trained affine, 8/32/64/128D | 4 | 31,104 coordinate-weighted | 38,556-40,824 | 81 |
| Coupled phase-crossing 2D | 1 | 100 generic grid | 64 | 25 |

The affine cases are an important negative comparison: automatic search
is not uniformly better than manual design. It still discovers a finite,
dimension-controlled upper model without a supplied coordinate pattern.
For the coupled system, the 25-state lower bound is independently
recomputed from all 25 exact initial-output packing points; its own
automatic certificate checks a simpler nine-point packing. The 64-state
upper is over the actual coupled ReLU network, including its active
internal phase boundary.

Evidence: 'runs/automatic_grid_global_v1/summary.json' and
'validation/automatic_grid_global_acceptance.json'. Each case stores
the proposed bin/radius vector, the exact certificate, the frozen model
hash, and a regeneration record.

This is a generic, checkable method to propose finite causal
realizations from network weights. It does not certify that the greedy
grid is minimum among grids, that the finite realization is globally
minimum among all machines, or that the network's causal variables and
sparse support were discovered from behavior. The current trained
models and benchmark remain synthetic; externally grounded systems and
the original broad R4/R5/R8/R9/R10 objectives are still open.



Release 0.39.0 is recorded in 'validation/wheel_v39_run/status.json'.
All 170 regression tests passed. Source, wheel, and isolated-installed Python
modules matched byte for byte; five installed-package commands passed,
including fresh coupled-grid synthesis/replay and formal learned and affine
replays. Wheel SHA-256:
91a32e50d87303cb683d05f44cc576d6ed3c04128f7a5892727a5388eb958a9e.

