# Causal feature-guided program synthesis protocol

Status: frozen before any confirmation or development evaluation artifact is
generated.

## Question

Can candidate-independent causal support for the 14 fixed discovery features
change bounded program search and improve final-test program-to-teacher
fidelity?

The intervention target is the frozen neural teacher's internal
representation. Ground-truth SCM labels are used only for separately reported
accuracy and never for program selection.

## Feature guidance

Before candidate search, execute the 14 discovery-feature CDIR roots on
alignment-fit and validation worlds. At each of the three fixed neural cuts:

1. fit a frozen linear readout for all 14 feature values;
2. fit an orthogonal numeric mapping on single-feature training interventions;
3. warm-start a biorthogonal read/write mapping from that orthogonal solution;
4. train both mappings with numerical loss only; no provisional symbolic
   program label or candidate-specific behavior enters this stage;
5. evaluate held-out single- and two-feature interventions, plus shuffled and
   random controls.

For each feature and site, support is positive only when the biorthogonal
target NMSE improves on no intervention and on both shuffled and random
controls. It is multiplied by natural-readout quality and collateral
preservation. Select the site using the mean corrected support on validation
worlds. Normalize the resulting 14 values to [0,1]. Site choice and
normalization use validation data only.

## Search and selection

Generate one common candidate pool from:

- the historical unaligned penalty grid;
- causal-aligned searches using the same penalties and a fixed alignment-weight
  grid 0.002, 0.01, 0.05.

All searches use extraction worlds and frozen teacher labels. Alignment costs
are additive over feature leaves in each predicate. Deduplicate candidates by
canonical program ID.

Select two programs on validation worlds:

- baseline: validation fidelity minus 0.001 times complexity, selected only
  among candidates reached by the historical unaligned searches;
- causal-guided: the same score plus 0.02 times the mean causal support of
  feature leaves used by the program, selected from the union of historical
  and causally aligned searches.

No final-test world is generated or accessed before both programs are fixed.
Report whether search enlarged the unique pool and whether selection changed.

## Seeds and partitions

- Development/integration seed: 1392 only.
- Confirmation A/B: seeds 1393 and 1394, using the frozen teachers associated
  with joint sources 191 and 192.
- Per confirmation seed: 1,024 extraction worlds, 1,024 alignment-fit worlds,
  3,072 validation worlds, and 1,024 untouched worlds in each of test_id,
  test_function, test_noise, test_scale, and test_intervention.
- All world identities are disjoint across roles.
- Mapping training uses 384 disjoint single-feature pairs. Validation uses 768
  disjoint pairs cycling all compatible single- and two-feature masks.

Historical seeds 393/394 and 1193/1194 are observation evidence and cannot be
used as confirmation for this method.

## Decision rule

The method is a replicated program-synthesis improvement only if:

1. causal guidance changes the selected program in both confirmation seeds;
2. guided program-to-teacher fidelity is no lower in any of the five final
   environments in either seed;
3. mean final-environment fidelity improves by at least 0.5 percentage points
   in both seeds;
4. the selected guided program uses at least one feature with positive
   control-corrected support;
5. all artifacts pass independent replay.

Ground-truth accuracy is reported but cannot substitute for teacher fidelity.
A failed rule is a negative result; no post-hoc environment or site subset
replaces it.

## Reproducibility

Store strict JSON/NPZ artifacts, model/program hashes, source snapshots, pair
pools, mappings, search traces, candidate IDs, predictions, and a SHA-256
manifest. The verifier regenerates every world and pair, refits readouts and
mappings, reruns all searches and selections, and recomputes final metrics.

