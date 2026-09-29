# Collateral-preserving tangent intervention protocol

## Question

Version 0.18 exacted active read-coordinate exchange inside a natural PCA tangent
but left inactive read coordinates unconstrained. Targeted numerical error
improved in all six confirmation cases, while one collateral comparison narrowly
failed. This experiment tests whether the tangent's remaining degrees of freedom
can suppress inactive-coordinate leakage without sacrificing the active exchange.

## Frozen method

Use the same frozen teachers, 54 variables, three sites, linear probes,
rank-one biorthogonal mappings, 95%-variance PCA tangent, pair generation, and
budgets as version 0.18. Compare:

1. ordinary biorthogonal writes;
2. minimum-norm natural-PCA writes from version 0.18;
3. collateral-preserving natural-PCA writes; and
4. an equal-rank random-subspace control.

For each mask, the new method first computes the minimum-norm tangent solution
that exactly matches active read-coordinate changes. In the null space of the
active constraints it then minimizes squared changes to every inactive
biorthogonal read coordinate plus a fixed `1e-3` coefficient on null-space
movement. This is a closed-form ridge solution fitted without test labels.
Report active residual, inactive read-coordinate RMS leakage, displacement
norm, numerical target/collateral NMSE, and behavioral interchange.

No final-test site, variable, threshold, penalty, or method selection is allowed.

## Data and confirmation

- Development seed: 3592.
- Confirmation seeds: 3593 with joint seed 191 and 3594 with joint seed 192.
- Each confirmation uses 1,024 fit worlds, 8,192 test worlds, 384 training
  pairs, 2,048 test pairs, 96 samples per world, 120 mapping updates, every
  compatible single/two-variable mask, all 54 groups, and all three sites.
- Complete regeneration and mapping-refitting replay is required.

## Frozen acceptance rule

The hypothesis passes only if:

1. both seeds replay completely with full site/group/mask coverage;
2. active-coordinate maximum residual is at most `1e-6` at every site;
3. inactive-coordinate RMS leakage is lower than the minimum-norm PCA baseline
   at every site in both seeds;
4. in each seed, weighted collateral NMSE is lower than the minimum-norm PCA
   baseline at at least two sites;
5. at those collateral-winning sites, weighted targeted NMSE is no higher;
6. in each seed, targeted NMSE remains lower than ordinary biorthogonal writes
   at at least two sites; and
7. all sites and groups are reported without post-hoc filtering.

Passing would establish a finite empirical improvement in intervention
selectivity. It would not establish exact circuit identity, unique abstraction,
causal truth, or general decompilation.
