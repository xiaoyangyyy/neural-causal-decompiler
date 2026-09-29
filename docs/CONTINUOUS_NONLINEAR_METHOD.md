# Phase-crossing continuous certification method

## Objective

This stage tests the continuous separation verifier after removing the stable-ReLU
assumption used by the 0.28 scaling experiment. The formal profiles increase
state dimension and finite control horizon together:

| Profile | State dim | Action dim | Horizon | Leaf budget | Near epsilon |
|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 2 | 1 | 80 | 0.03 |
| 1 | 32 | 2 | 2 | 64 | 0.08 |
| 2 | 64 | 3 | 3 | 24 | 0.12 |

The initial states are fixed points. Every certificate covers the complete
continuous action-word box [0,1]^(action_dim*horizon).

## Learned nonlinear systems

Each transition has a fixed first ReLU feature layer and a least-squares fitted
output layer. Three control-dependent gates have certified preactivation
intervals [-0.65,0.75], [-0.85,0.55], and [-0.85,0.35] on the declared
training domain. Every interval contains zero, so the benchmark cannot be
reduced to one globally stable affine phase. The fitted ReLU network is the
system being verified; the target construction is used only to generate
training pairs.

## Sound hybrid bound

For a branch leaf, the relational method is used only if both executions have
the same certified ReLU phase at every propagated gate. Shared controls and
biases then cancel in the difference recurrence. If any phase cannot be proved
equal and stable, that leaf is recomputed with outward-rounded independent IBP.
The certificate stores the actual method for every leaf, and the verifier
recomputes every bound.

The aggregate lower bound is the largest distance attained by a feasible center
word. The aggregate upper bound is the largest leaf upper bound. A pair is
separated when the lower bound exceeds 2*epsilon, within tolerance when the
upper bound is at most 2*epsilon, and otherwise unresolved.

## Two-by-two ablation

Four configurations separate the effects of the bound and the split heuristic:

| Name | Leaf bound | Split coordinate |
|---|---|---|
| hybrid_best | relational when stable, IBP fallback | smallest one-step child maximum |
| hybrid_widest | relational when stable, IBP fallback | widest coordinate |
| independent_best | independent IBP | smallest one-step child maximum |
| independent_widest | independent IBP | widest coordinate |

best-bound is a greedy one-step heuristic. It has no completeness or dominance
claim. An unresolved result means only that the fixed leaf budget did not close
the interval.

## Verification

The manifest hashes every artifact. Independent replay refits every model,
reconstructs all branch trees, checks exact domain coverage, recomputes every
leaf and witness bound, checks aggregate status, and compares the deterministic
profile JSON after removing timing fields.
