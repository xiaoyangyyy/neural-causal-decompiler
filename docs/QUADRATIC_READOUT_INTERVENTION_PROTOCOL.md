# Quadratic-readout internal intervention protocol

Frozen after the verified version 0.6 error-floor audit and before new confirmation worlds are generated.

## Motivation and question

The version 0.6 linear probes have natural-state weighted NMSE of 0.54-0.89 across seeds/sites, before any intervention. This measurement error contaminates the 2.40-4.30 targeted intervention NMSE. The experiment asks whether an explicit diagonal-quadratic readout reveals substantially more accurate numerical interchange under the same rank-one biorthogonal intervention, without changing the teacher or symbolic targets.

A better probe remains a measurement instrument, not evidence of causal mechanism by itself.

## Readouts

- Linear comparator: the existing frozen ridge readout.
- Quadratic candidate: standardize hidden coordinates on alignment-fit worlds; expose each standardized coordinate and its square; standardize those `2d` features; fit the same per-variable ridge objective with masked executed examples.
- No cross-coordinate products are allowed. All means, scales, coefficients, target variances, and fit counts are serialized.
- Both readouts train only on alignment-fit natural states. Neither sees intervention-test pairs or confirmation metrics.

## Intervention controls

For each readout and each of representation, head_linear, and head_tanh:

1. fit the existing orthogonal numeric mapping on the same fixed single-variable training pairs;
2. warm-start the rank-one biorthogonal numeric mapping from that orthogonal mapping;
3. freeze the teacher and readout throughout mapping optimization;
4. evaluate on the identical held-out single/two-variable pair pool.

For the quadratic candidate also fit behavior-only, shuffled-target, and random-biorthogonal controls. Training steps, losses, conditioning penalty, rank, pair generation, and symbolic executor match version 0.6.

## Sources, budgets, and confirmation

- Seed 3193 uses frozen joint source 191; seed 3194 uses source 192.
- Per seed: 1,024 alignment-fit worlds, 8,192 alignment-test worlds, 384 fixed train pairs, 2,048 fixed test pairs, 96 observations per world, and all 1,411 compatible masks.
- All 54 raw-statistic, dependence-kernel, and cross-fit regression groups remain in scope.
- Full replay regenerates worlds, traces, readouts, pair pools, mappings, controls, predictions, metrics, hashes, and manifests.

## Decision rule

Call the quadratic readout a replicated R5 measurement improvement only if:

1. both runs pass complete replay and full mask coverage;
2. in each seed, quadratic natural-state weighted NMSE is lower than linear at at least two of three fixed sites;
3. in each seed, quadratic biorthogonal targeted NMSE is lower than linear biorthogonal at at least two sites;
4. mean collateral NMSE is no higher at those winning sites;
5. quadratic targeted NMSE beats its shuffled and random controls; and
6. all sites and groups are reported without test-time subset selection.

Passing would reduce a known measurement floor and strengthen empirical internal intervention evidence. It would not prove uniqueness, exact internal circuit identity, universal equivalence, causal truth, or complete neural-to-SCM decompilation.
