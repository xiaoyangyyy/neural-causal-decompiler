# Biorthogonal full-trace intervention protocol

Status: frozen before evaluation artifacts are generated.

## Question

Does a biorthogonal read/write intervention recover causally exchangeable
internal variables better than the existing jointly orthogonal rank-one
mapping when the frozen teacher may use overlapping directions?

This experiment tests empirical intervention fidelity. It does not establish a
unique decompilation, a universal causal theorem, or that a linear probe is a
mechanism.

## Frozen sources and seeds

- Development and integration only: seed 1092. Its results may change code but
  cannot support the confirmation claim.
- Confirmation A: seed 1093 with the frozen joint source for seed 191.
- Confirmation B: seed 1094 with the frozen joint source for seed 192.
- Worlds, pair pools, sites, methods, and metrics are generated for both
  confirmation seeds even if the first result is unfavorable.
- Seeds 793/794, 893/894, and 993/994 are historical evidence and are not reused
  for method selection or confirmation.

## Target scope

The executor enables both dependence and regression tracing and uses the union
of all versioned scalar groups:

- 8 raw feature-statistic groups;
- 18 dependence-kernel groups;
- 28 cross-fit regression groups.

All 54 groups are fixed before training. Training uses only single-group
interventions. Testing cycles over all 1,411 compatible single- and two-group masks, so all
two-group combinations are held out from mapping optimization.

The frozen world budgets are 1,024 alignment-fit worlds, 4,096 alignment-test
worlds, 384 train pairs, and 2,048 test pairs, with 96 observations per world.
No world or intervention pair is shared between fitting and testing.

## Mapping and controls

At each of representation, head_linear, and head_tanh cuts, fit:

1. biorthogonal numeric: learned write columns W and their exact dual
   R = W(W'W)^-1, using behavior plus numeric intervention loss. Optimization
   starts from the orthogonal numeric mapping fit on the same training pool;
   the returned checkpoint minimizes the full training-pool objective among
   step 0 and all optimization steps, without consulting test data;
2. orthogonal numeric: the existing jointly orthogonal rank-one mapping, fit on
   exactly the same pool and objective weights;
3. biorthogonal behavior-only;
4. biorthogonal shuffled-target;
5. random biorthogonal.

The teacher and numerical readout stay frozen. Returned biorthogonal mappings
must pass R'W = I within 2e-5. Report the write Gram condition number and maximum
absolute inter-column cosine. The random control uses the same dimensions and
dual construction.

## Fixed metrics and decision rule

Report every site and method, without choosing a site on test results:

- natural probe NMSE;
- targeted and collateral intervention NMSE, with no-intervention NMSE;
- overall and informative behavioral interchange accuracy;
- executed/informative pair counts;
- biorthogonality and conditioning diagnostics.

For each seed and site, aggregate targeted NMSE by weighting each group by its
number of scored targeted examples. The primary contrast is paired
biorthogonal-numeric minus orthogonal-numeric aggregate targeted NMSE. A lower
value favors the new mapping.

Call the new mapping a replicated improvement only if both confirmation seeds
show a lower primary NMSE at at least two of the three fixed sites, neither seed
has a greater mean collateral NMSE at those winning sites, and the result is
better than shuffled and random controls. Behavior-only is diagnostic and
cannot by itself satisfy the numerical criterion. Failure of this rule is
reported as a negative result; no post-hoc subset of groups or sites replaces
it.

## Reproducibility gate

Each run stores strict JSON/NPZ artifacts, source hashes, copied source modules,
and a SHA-256 manifest. An independent verifier regenerates worlds and pairs,
refits every mapping and readout, recomputes all metrics, and checks the
manifest. Confirmation is not interpreted until both runs pass replay.

## Protocol amendment 1: coverage-budget failure

This amendment was frozen after both 1093/1094 runs were replayed but before
their scientific metrics were inspected. Both runs accepted only 1,383 test
pairs because the disjoint sampler exhausted 4,096 worlds. That is fewer than
the 1,411 compatible masks, so those runs are reproducible engineering
artifacts but invalid confirmation runs.

The replacement confirmation seeds are 1193 and 1194, paired with frozen joint
sources 191 and 192 respectively. No mapping or metric rule changes. The only
budget change is 8,192 alignment-test worlds per seed; fit worlds remain 1,024,
train pairs remain 384, and requested test pairs remain 2,048. Since each
accepted mask has at most two active groups, 8,192 worlds leaves explicit room
for rejected execution-conditioned groups beyond the 6,144-world maximum for
2,048 immediately accepted pairs. A replacement run is valid only if it
accepts all 2,048 pairs and its test masks include all 1,411 compatible masks.
