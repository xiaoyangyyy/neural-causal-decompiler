# Structured symbolic mechanism search results

Protocol: `docs/STRUCTURED_MECHANISM_PROTOCOL.md`.

Both 30-world confirmation runs passed full replay. The preregistered rule **passed**.

| Seed | Baseline neural NMSE | Structured neural NMSE | Relative reduction | Baseline atoms | Structured atoms |
|---:|---:|---:|---:|---:|---:|
| 1793 | 0.029551 | 0.005412 | 81.7% | 2.308 | 0.886 |
| 1794 | 0.193794 | 0.006633 | 96.6% | 2.214 | 0.812 |

| Pooled diagnostic | Baseline | Structured | Change |
|---|---:|---:|---:|
| Neural-teacher NMSE | 0.111672 | 0.006023 | -94.6% |
| Ground-truth NMSE | 0.341882 | 0.228992 | -33.0% |
| Intervention-effect MAE | 7256.292319 | 0.030328 | -100.0% |
| Operator exact fraction | 0.504 | 0.668 | +0.164 |

The primary result concerns fidelity to frozen neural conditional mechanisms. Ground-truth and intervention metrics are diagnostics. The very large baseline intervention MAE is caused by redundant polynomial terms extrapolating outside their fitted range; the hierarchy constraint removes this instability in these runs.

The oracle DAG isolates equation extraction. These results strengthen R10 but do not solve R9, do not measure end-to-end inferred-graph SCM recovery, and do not prove causal identifiability or a universal theorem.

Machine-readable evidence: `validation/structured_mechanism_acceptance.json`.
