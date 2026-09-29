# Conservative causal-guided synthesis results

Protocol: docs/CONSERVATIVE_CAUSAL_GUIDANCE_PROTOCOL.md.

Both confirmation runs passed full regeneration replay. The preregistered safe-improvement rule failed.

| Seed | Selection changed | Baseline mean fidelity | Guided mean fidelity | Delta | All environments nondecreasing |
|---:|:---:|---:|---:|---:|:---:|
| 1493 | yes | 0.6984 | 0.7012 | +0.0027 | no |
| 1494 | yes | 0.7145 | 0.7076 | -0.0068 | no |

| Seed | ID | Function | Noise | Scale | Intervention |
|---:|---:|---:|---:|---:|---:|
| 1493 | +0.0107 | +0.0049 | +0.0039 | -0.0010 | -0.0049 |
| 1494 | -0.0088 | -0.0010 | -0.0244 | +0.0000 | +0.0000 |

The validation noninferiority gate held by construction for both selected programs, but seed 1493 still decreased scale and intervention fidelity, while seed 1494 decreased ID, function, and noise fidelity. The equally weighted pooled mean change was -0.0021.

This rejects aggregate ID-validation noninferiority as a sufficient safety condition. The next selection design must create function/noise/scale/intervention perturbations inside the selection partition and require noninferiority across those synthetic validation environments before final tests are generated. Seeds 1493/1494 are historical failure-analysis data.

Machine-readable evidence: validation/conservative_guided_acceptance.json.
