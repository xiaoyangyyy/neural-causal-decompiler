# Biorthogonal full-trace intervention results

Protocol: docs/OBLIQUE_NUMERIC_PROTOCOL.md.

## Validity

- Development seed 1092 was used for implementation and stability diagnosis only.
- Seeds 1093/1094 are reproducible but invalid confirmation runs: the disjoint sampler accepted 1,383 pairs, fewer than the 1,411 compatible masks. Their scientific metrics were not used.
- Replacement seeds 1193/1194 each used 1,024 fit worlds, 8,192 test worlds, 384 train pairs, and 2,048 accepted test pairs.
- Both replacement runs contain all 1,411 compatible single/two-group masks and independently replayed all worlds, pairs, probes, mappings, audits, source hashes, and manifests.

## Primary result

The preregistered replicated-improvement rule passed. Weighted target NMSE and collateral NMSE are weighted by each group's scored examples.

| Seed | Site | Oblique target | Orthogonal target | Delta | Oblique collateral | Orthogonal collateral | Site rule |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1193 | head_linear | 4.2974 | 12.6471 | -8.3496 | 17.7983 | 24.4397 | pass |
| 1193 | head_tanh | 2.4024 | 3.8586 | -1.4562 | 15.3724 | 16.6386 | pass |
| 1193 | representation | 3.2392 | 5.3222 | -2.0829 | 16.0625 | 16.8194 | pass |
| 1194 | head_linear | 3.3959 | 5.1549 | -1.7590 | 13.7019 | 16.3943 | pass |
| 1194 | head_tanh | 2.9541 | 5.9312 | -2.9771 | 13.3500 | 14.9871 | pass |
| 1194 | representation | 3.5247 | 3.8747 | -0.3499 | 12.8875 | 13.8167 | pass |

Every seed passed at all three fixed sites; the rule required at least two. At each winning site, oblique numeric also had lower target NMSE than the shuffled-target and random-biorthogonal controls.

## Interpretation

The result rejects the earlier implementation assumption that each causal quantity must use the same mutually orthogonal read/write direction. A training-only warm start from the orthogonal solution followed by a dual read/write optimization consistently improves held-out numerical interchange across the union of 54 raw-statistic, dependence-kernel, and cross-fit regression groups.

The absolute target NMSE remains materially above zero (2.40 to 4.30), and collateral NMSE remains high (12.89 to 17.80). The experiment therefore establishes a replicated relative intervention improvement, not exact causal mechanism recovery. It also does not establish uniqueness, universal equivalence, or causal truth of the extracted program; targets are executions of the frozen symbolic program and behavior is measured against the frozen neural teacher.

Machine-readable decision: validation/oblique_numeric_acceptance.json.
