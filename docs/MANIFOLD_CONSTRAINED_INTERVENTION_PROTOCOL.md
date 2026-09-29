# Manifold-constrained internal intervention protocol

## Question

Version 0.17 showed that a more flexible quadratic probe improves natural-state
measurement but becomes substantially worse after hidden-state patching. This
experiment tests the narrower hypothesis that rank-one interventions fail
because their write vectors leave the locally supported hidden-state manifold.
It does not treat probe readability as mechanism evidence.

## Frozen method

- Use the same frozen teachers, symbolic executor, 54 scalar trace groups,
  three neural sites, pair construction, and rank-one linear biorthogonal
  mapping objective as version 0.6.
- Fit a linear probe and the biorthogonal read/write mapping only on
  `alignment_fit` worlds and fixed single-variable training pairs.
- Fit a PCA basis to the natural `alignment_fit` hidden states at each site.
  Retain the smallest rank explaining at least 95% of centered variance.
- For each held-out patch, begin at the unmodified base state and permit a
  displacement only in the fitted PCA span. Choose the minimum-norm tangent
  displacement whose active biorthogonal read coordinates equal the
  source-minus-base coordinates. Use a Moore-Penrose inverse with fixed
  tolerance `1e-10`.
- Compare the ordinary biorthogonal write, the constrained natural-PCA write,
  and a seeded Haar-random subspace with exactly the same rank. The random
  control uses the same active-coordinate constraint solver.
- Report constraint residuals, displacement norms, PCA rank and explained
  variance, numerical target/collateral NMSE, and behavioral interchange.

No final-test site, group, rank, variance threshold, or method selection is
allowed.

## Data and confirmation

- Development integration seed: 3392.
- Confirmation seeds: 3393 using frozen joint seed 191, and 3394 using frozen
  joint seed 192.
- Per confirmation seed: 1,024 fit worlds, 8,192 test worlds, 384 fixed train
  pairs, 2,048 fixed test pairs, 96 samples per world, all 1,411 compatible
  single/two-group masks, three sites, and 120 mapping updates.
- World identities are disjoint across fit and test. Confirmation runs require
  complete regeneration and mapping-refitting replay.

## Frozen acceptance rule

The hypothesis passes only if all of the following hold:

1. both confirmation runs pass complete replay and cover every fixed site,
   group, and compatible mask;
2. active-coordinate maximum absolute residual is at most `1e-6` for every
   natural-PCA site;
3. in each seed, natural-PCA weighted targeted NMSE is lower than ordinary
   biorthogonal targeted NMSE at at least two of three sites;
4. at those winning sites, natural-PCA weighted collateral NMSE is no higher
   than ordinary biorthogonal collateral NMSE;
5. in each seed, natural-PCA targeted NMSE is lower than the equal-rank random
   subspace control at at least two sites; and
6. all sites and groups are reported without post-hoc filtering.

Passing would support a finite, empirical manifold-constrained intervention
construction. It would not prove exact circuit identity, unique causal
abstraction, causal truth, or general neural-to-SCM decompilation.
