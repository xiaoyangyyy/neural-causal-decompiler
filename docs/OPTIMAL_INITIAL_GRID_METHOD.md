# Exact minimum initial coordinate grid for a fixed recurrent relation

The eight frozen trained phase-crossing ReLU rings use tolerance
`epsilon = 17/100`, four directly observed state coordinates, and the
previous certified recurrent relation with 128 action bins. The recurrent
image enclosure retains 216 cells. The new constructor chooses the
initial grid instead of assigning three bins to the hidden feedback axis.

## What is certified

Let `b_j` be positive integer initial coordinate bins, `n_j` the frozen
recurrent bins, and `r_j` its certified simulation radii. Write `L_F` and
`L_H` for the exact rational absolute-weight sensitivity bounds computed
from the actual ReLU networks. The proof predicate is

\[
 L_H((1/(2b_j))_j)\leq\epsilon,\qquad
 L_F((1/(2b_j))_j,(1/(2m))_a)_i+1/(2n_i)\leq r_i.
\]

The first inequality covers the full initial cube. The second includes
both continuous-action quantization and recurrent state quantization;
it hands every initial point and every action into the fixed recurrent
simulation relation. Its prior proof covers every subsequent finite
horizon. Since every abstract center and action center lies in the unit
cube, every successor lies in the previously certified transition-image
box and its quantized state belongs to the retained 216-cell set.

The runtime exposes the initial selector, observation, and transition
functions. Initial and recurrent states carry different tags. There are
no hidden continuous variables in this finite machine. Action symbols
are quantized implicitly; their count is not added to the state count.
All proof arithmetic is rational. Smoke traces check execution, while
the inequalities and induction establish the full-cube, all-horizon claim.

## A small exact optimality theorem

The verifier extracts the observation network's exact affine map and
requires it to be the first four coordinates with zero bias. Each
observed axis therefore needs at least three initial bins: two bins have
midpoint error `1/4 > 17/100`. Hence every admitted coordinate grid has at
least `3^4 = 81` cells.

Any grid with fewer than 108 cells must have exactly three bins on those
four axes and one bin on every hidden axis. Raising an observed bin count
from three to four already makes 108 cells; raising a hidden bin count
from one to two makes at least 162. Thus the only candidate below 108 is
the 81-cell baseline. The constructor checks that baseline and the four
possible 108-cell single-observed-axis refinements exactly.

For both 8D seeds, the baseline passes, proving a minimum of 81 within
this proof class. For both seeds at 32/64/128D, its sensitivity handoff
fails on coordinate zero, and a 108-cell refinement passes. This proves
a minimum of 108 within the same class. All selected grids give the
hidden feedback axis one bin.

A failed upper-bound inequality is an exclusion from this proof
predicate, **not** a concrete counterexample or proof that an 81-state
behavioral machine is impossible. The optimality statement fixes the
recurrent relation, action resolution, equal-width coordinate cells,
midpoint representatives, and exact sensitivity proof predicate. It is
not optimality among all partitions, relations, or finite machines.

## Evidence and replay

`ncd/initial_grid_synthesis.py` recomputes the observation map, exact
bounds, branch exclusion, minimal candidate, and recurrent enclosure.
Every certificate binds the model and recurrent certificate by canonical
SHA-256. `scripts/acceptance_optimal_initial_grid.py` additionally binds
file hashes to the previous eight-case acceptance, replays its lower and
two-stage upper proofs, and replays exact state/control phase witnesses.
It checks three traces and five continuous actions per model.

```powershell
python -m scripts.acceptance_optimal_initial_grid --verify
python -m pytest tests/test_initial_grid_synthesis.py
```

The general finite-machine minimum intervals become `[81,297]` at 8D
and `[81,324]` at 32/64/128D. The independent lower remains 81; there is
no new global minimum theorem. Models still have structured fixed hidden
features and fitted output coefficients on synthetic traffic-inspired
data. Original requirements R4/R5/R8/R9/R10 remain incomplete.
