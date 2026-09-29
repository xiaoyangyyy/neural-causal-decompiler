# Next proof obligations: jointly coupled state/control ReLU phases

The full original objective remains active. Version 0.58 proves minimum
initial labels and a whole-machine successor obstruction on eight frozen
structured systems. It does not finish R4/R5/R8/R9/R10 or establish a
universal neural-to-SCM inverse.

## Concrete missing capability

The phase-initial handoff currently rejects any relevant hidden unit with
both state and control weights. Thus merely increasing state dimension
would preserve a real structural limitation. The next accepted experiment
must contain nonzero mixed state/control hidden units and train hidden
parameters as well as output coefficients; a fixed hand-designed feature
dictionary must not be presented as general end-to-end neural learning.

## Proposed exact relation proof

For a local scalar output depending on at most two state coordinates and
two controls, certify the joint shared-control difference

    E_i = max_(x in initial pair-cell, u in unit square)
          |F_i(x,u) - F_i(c,u)|.

Both network copies receive the same continuous u. Work in at most four
continuous variables; retain all signed affine sums. Split the domain
into actual ReLU phases for both copies, including mixed units. For each
leaf establish a nonempty rational polytope, all selected phase signs,
and exact extrema of its affine difference. Empty leaves need a checked
infeasibility witness. Stable phases can be pruned using sound bounds.
The checked closed children must cover each parent domain.

Independently bound the signed control gradient at abstract centers,
then verify

    E_i + L_i/(2*action_bins) + 1/(2*recurrent_bins_i) <= radius_i.

Reuse the weighted recurrent relation only after replaying it for the new
frozen model. Do not infer safety from sample trajectories. If the exact
bound does not close, report unresolved and keep the concrete witness or
remaining cover, rather than silently reverting to a separated network.

## Training and acceptance requirements

1. Freeze all training architecture, seeds, data splits and hyperparameters
   before the final run. Synthetic targets may generate data but cannot
   supply extracted mechanisms or phase equations to the checker.
2. Train actual hidden weights with explicit local support constraints;
   report that support constraint as a remaining restriction. Retain both
   selected seeds and every declared dimension, including failed proofs.
3. Store the neural checkpoint, exact mixed-unit and phase-crossing
   witnesses, recurrent relation, joint initial handoff, complete action
   graph and standalone program. Compilers must derive all structure from
   the frozen checkpoint.
4. Rebuild action partitions per abstract source; mixed units can change
   action thresholds and remove the one-template property. Export whatever
   template count is proved rather than assuming one.
5. Test interior and degenerate phase boundaries, feasible and infeasible
   polytope leaves, altered neural coefficients, certificate forgery and
   models where a proposed handoff really fails.
6. Replay the full graph and all upper/lower bindings in an isolated
   installed package after regression passes. Preserve older schemas.

## Independent global-minimum work

The 82-state lower uses a decoder-band lemma specific to a hypothetical
81-state machine. Extra labels can change continuum coverage, so the same
lemma must not be reused to claim 83 or a larger lower. A stronger lower
needs constraints that explicitly account for extra labels and common
successors. Any solver-based infeasibility result needs an independently
replayable certificate, not only a solver status or sampled infeasibility.

Reducing a computed conservative graph or proving a grid minimum remains
an upper/proof-class result. The whole-machine state/code minimum and
original neural circuit-to-rule/SCM recovery need separate evidence.
