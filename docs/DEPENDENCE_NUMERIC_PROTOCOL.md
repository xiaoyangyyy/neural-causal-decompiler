# Dependence-kernel mechanism protocol

Frozen after quick seed 892 and before formal seeds 893/894. Seed 893 uses the
joint-191 frozen teacher/program; seed 894 uses joint-192. Seed 892 is integration
only.

The raw CDIR executor expands each of the three dependence occurrences (`dep_xy`,
`resdep_xy`, `resdep_yx`) into six scalar steps: two median-distance kernel
bandwidths, two centered-kernel energies, the cross-kernel numerator, and the
normalization denominator. This yields 18 occurrence-aware structural groups.
Their unmodified execution must reproduce the declared CDIR features exactly.

Causal dependencies are fixed before pairing: each bandwidth precedes its matching
energy, numerator, and denominator; each energy precedes the denominator. A pair
containing an ancestor and descendant is excluded. Other singleton and pair masks
are eligible. No neural output or truth label filters pair acceptance.

Each formal run uses 1,024 fit worlds and 4,096 separate test worlds with 96 samples.
Fit uses 384 non-reused singleton pairs; test uses 1,024 non-reused compatible
singleton/pair groups. All three neural cuts, rank-one orthogonal blocks, 120 updates,
numeric weight 0.5, behavior-only, shuffled-target and random controls are reported.
No test-based site or method selection is allowed.

Primary evidence is per-step natural, targeted, no-intervention and collateral NMSE,
plus behavior accuracy on informative pairs. This is a finite intervention audit of
frozen teachers. It does not expose cross-fit regression internals, prove unique
neural coordinates, or establish a universal causal principle.
