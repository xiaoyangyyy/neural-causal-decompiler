# Exact partial realization of the actual normalization prefix

This is a scoped positive R4/R5 result. It does not close the original universal
claims or recover the learned encoder/head. The unchanged historical teacher is
`runs/oblique_seed1193/teacher.pt` (SHA256
`37ce128dd6a5050075310c03b268e8089b2bc49d69670c4cb92a3f38e6519ef3`).
Actual `torch.fx.symbolic_trace` exports both normal and variable-swapped branches.
The verifier independently loads the checkpoint, re-exports the graph, checks its
hash, recognizes the primitive operations/arguments, and re-derives the cut edges.
Nothing is inserted into or retrained in the target model.

## State and local proof

For column j, let m_j=sum_r x_rj/N and v_j=sum_r(x_rj-m_j)^2/N. The program computes
s_j=sqrt(v_j), d_j=max(s_j,F), c_rj=x_rj-m_j, z_rj=c_rj/d_j, and
q_rj=max(-20,min(20,z_rj)). F is the exact rational value of the original host
binary64 constant 1e-5. The denominator is at least F, which is strictly above
the actual CDIR protected-division cutoff 1e-8; the verifier checks the exact
rational margin. Population variance is nonnegative. Thus all primitive
relations hold, including zero variance, equality s_j=F, and both clip boundaries.

CDIR supplies mean, population variance, square root, scalar conditional,
subtraction and division. The emitted program includes an explicit typed vector
clip intrinsic. Its three disjoint guards cover the entire line. A cancelling
absolute-value identity is not used as a floating runtime implementation.

## Fixed internal intervention map and composition

There are four named control states: mean_x, mean_y, clamped_std_x and
clamped_std_y. Each selected state takes its value from its own independently
selected valid source dataset; four sources need not agree. Both actual FX
occurrences are patched in the corresponding coordinate, reversing coordinates
in the swapped branch. This is the stated compatible intervention family.
Patching the mean does not recompute the natural raw population variance.

For each of the 16 masks, readout of the actual patched prefix states equals
execution of the explicit patched program. The relation includes all mean,
raw-standard-deviation, clamped-standard-deviation, centered, normalized and
clipped coordinates, including collateral changes in states that were not patched.
The branch trace records the selected conditional-child occurrence; an unselected
occurrence does not receive success coverage merely because a reused value was
computed elsewhere.

There are exactly four prefix escape edges: the clamped std and clipped tensor
in each branch. The std escape is needed for the existing log-scale input to the
head; dropping it would make the composition claim invalid. The verifier checks
that no other prefix state escapes. Equality at all cut edges, identical raw
input, and the same deterministic unchanged neural suffix imply identical final
mathematical logits. A common deterministic label rule gives identical labels,
including ties. This is a local-relation-to-whole-computation proof; the remaining
suffix is still neural.

## Exact runtime versus floating diagnostics

The theorem uses real operators with frozen stored parameter values and explicit
binary64 host constants. It is not a guarantee for every floating device execution.
The executable exact interpreter accepts rational N-by-2 inputs with N>=16,
including every finite binary64 dataset encoded as its exact rational value.
Means and variances are rational. Square-root and normalized values are represented
as n/sqrt(d), with rational n and positive rational d. Comparisons to rational
floors and clip bounds use signs and exact squared comparisons; floating output
is only an optional approximation. Non-rational real inputs are within the
mathematical theorem but are not serialized by the executable rational runtime.

Separate float64 diagnostics use the original frozen weight values, six frozen
fixtures and all 16 masks. They include zero variance, floor equality, clipping
edges, huge normalized inputs, independent source shifts and development seed
8100. Every intermediate coordinate and the final unchanged neural continuation
are checked. These 96 cases are diagnostics, not 96 independent scientific worlds
or a statistical confidence certificate.

## Commands and installation

```
python -m normalizer_proof prove --config validation/normalizer_proof_protocol_v1.json
python -m normalizer_proof verify runs/actual_normalizer_composition_v1/certificate.json
python -m normalizer_proof verify-bundle runs/actual_normalizer_composition_v1/manifest.json
python -m normalizer_proof execute --program runs/actual_normalizer_composition_v1/program.json --input input.json --output exact.json
```

Execution input has `data`, optional `sources` (four datasets), and optional `mask`
(four Booleans). Fraction strings such as `1/3` are accepted. The standalone
`ncd-normalizer-proof` entry point has the same commands. It remains separate from
the pinned core 0.59 while the 300-world confirmation replay is in progress.

`validation/check_normalizer_proof_v1.py` runs with `python -I` in the isolated
installed environment. It checks source/installed module-byte equality, replays
both proof and all 96 diagnostics, rejects weight/mask/closure forgeries and runs
12 meaningful boundary and intervention tests. Source and installed tests pass.
The original ledger remains 2 refuted and 36 unresolved. The earlier impossibility
of raw-mean recovery from *post*-normalization trace is unaffected: this positive
map explicitly reads/writes the actual *pre*-normalization mean nodes.
