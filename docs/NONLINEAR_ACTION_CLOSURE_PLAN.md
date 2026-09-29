# Next proof target: shared-action closure of actual nonlinear networks

The full project goal and original R4/R5/R8/R9/R10 remain open. This
bounded next step strengthens the certified nonlinear realization track;
it does not replace those end-to-end requirements.

## Evidence that motivates implementation

`validation/nonlinear_action_region_probe.json` binds the frozen seed-6101
128D model. Each of three initial centers yields four actual hidden ReLU
action phase boundaries, five exact polygons covering the action cube,
and six jointly feasible closed target cells instead of 32 coordinate-box
candidates. The probe checks the original network exactly at polygon
vertices. It is exploratory: no full recurrent closure has been computed.

## Implementation and proof obligations

1. Extract sparse first-layer affine preactivations and nonzero second-layer
   coefficients directly from the frozen network. Require two actions and
   one hidden layer. Hidden units irrelevant to the active recurrent axes
   can be omitted only after their output coefficients are checked zero.
   Avoid dense exact evaluation of all 128 output coordinates per vertex.
2. At each abstract state center, split the unit action square along every
   relevant phase-crossing hidden preactivation using rational half-plane
   clipping. Zero-area boundaries must stay covered by adjacent regions.
   Verify every polygon has a fixed valid phase, its affine output form
   equals the network algebra in that phase, and the splitting construction
   covers the action square. Do not substitute an affine network surrogate.
3. For each target recurrent cell, intersect each action polygon with all
   coordinate target-cell inequalities. Retain the cell if any closed
   intersection is nonempty, including point/segment boundary intersections.
   Closed cells may introduce extra edges at floor-quantization ties; this
   is a conservative relation. The runtime's 128-by-128 action midpoints
   are a subset of the action cube and therefore covered.
4. Enumerate all certified 81/108 initial centers. Compute their successors,
   then run a breadth-first least fixed point over recurrent centers.
   Every retained cell must lie inside the prior certified 216-cell image
   enclosure. Store ordered initial successor sets and recurrent graph.
5. Recompute the whole graph in the checker, verify closure, and bind the
   model, weighted relation, and optimal-initial-grid certificate hashes.
   Reuse the exact output/handoff and recurrent simulation inequalities.
   The machine upper is initial-state count plus retained recurrent count;
   the independent general lower remains 81.
6. Check the same eight frozen seed/dimension cases, including actual state
   and action phase witnesses, runtime traces, boundary behavior and tamper
   rejection. Report cases with no reduction too. Complete regression and
   isolated installed-wheel replay before declaring a new accepted upper.

The least fixed point is minimal only for this conservative graph. It is
not the exact concrete reachable set, a minimum over all possible finite
machines, or an end-to-end neural-to-SCM recovery result.


Implementation update: the complete eight-case graph and portable-program
acceptance are implemented in 0.57. See NONLINEAR_ACTION_CLOSURE_METHOD.md
and ../RESULTS_NONLINEAR_ACTION_CLOSURE.md for the actual scope and
verification evidence. The original full project goal remains open.
