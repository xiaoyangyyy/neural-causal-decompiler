# Causal feature-guided synthesis results

Protocol: docs/CAUSAL_FEATURE_GUIDANCE_PROTOCOL.md.

## Validity

- Development seed 1392 was used only for implementation and diagnosis.
- Confirmation seeds 1393/1394 each used 10,240 disjoint worlds, 384 single-feature fit pairs, 768 validation pairs, all 105 compatible feature masks, three neural cuts, 12 fixed search settings, and five untouched final environments.
- Both runs were independently regenerated from the frozen teacher through mappings, search, selection, predictions, and metrics.

## Fixed decision rule

The replicated-improvement rule failed. Both seeds changed the selected program and used positively supported features, but only seed 1393 satisfied the final-test constraints.

| Seed | Selected site | Positive features | Baseline mean fidelity | Guided mean fidelity | Delta | All environments nondecreasing |
|---:|---|---:|---:|---:|---:|:---:|
| 1393 | head_tanh | 10 | 0.6855 | 0.7090 | +0.0234 | yes |
| 1394 | head_tanh | 12 | 0.6937 | 0.6793 | -0.0145 | no |

Per-environment teacher-fidelity changes:

| Seed | ID | Function | Noise | Scale | Intervention |
|---:|---:|---:|---:|---:|---:|
| 1393 | +0.0244 | +0.0322 | +0.0029 | +0.0293 | +0.0283 |
| 1394 | -0.0146 | -0.0332 | -0.0127 | -0.0049 | -0.0068 |

## Failure diagnosis

- The normalized 14-feature support rankings have Spearman correlation 0.300 across the two frozen teachers (two-sided p=0.297). Feature intervention evidence is therefore only partly stable.
- Seed 1393 improved all five environments (+0.29 to +3.22 percentage points). Seed 1394 degraded all five (-0.49 to -3.32 points).
- Both runs selected head_tanh and changed programs, so failure is not caused by a zero score or an inactive search path. It is a cross-teacher generalization failure of the feature-support prior.
- Ground-truth program accuracy moved with the selected programs but was diagnostic only; it did not enter selection or the decision rule.

## Interpretation

Candidate-independent causal feature interventions solve the circularity of the earlier candidate-specific score and demonstrably alter bounded search. They do not yet provide a stable selection principle. Future work must estimate support stability across fit-only subenvironments or teachers and use a robust lower-confidence objective before another untouched confirmation. Seeds 1393/1394 are now historical failure-analysis data and cannot be reused as confirmation.

Machine-readable evidence: validation/causal_guided_acceptance.json.
