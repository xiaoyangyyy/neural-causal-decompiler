# OOD-guarded causal-guided synthesis results

Protocol: `docs/OOD_GUARDED_CAUSAL_PROTOCOL.md`.

Both confirmation runs passed full regeneration replay. The selector abstained
in both runs because the historical program was the only candidate satisfying
ordinary-validation and all-five-guard noninferiority. The preregistered
replicated-safe-improvement rule therefore failed.

| Seed | Candidates | Eligible | Selection changed | Mean baseline fidelity | Mean guarded fidelity | Delta |
|---:|---:|---:|:---:|---:|---:|---:|
| 1593 | 11 | 1 | no | 0.6936 | 0.6936 | 0.0000 |
| 1594 | 10 | 1 | no | 0.6922 | 0.6922 | 0.0000 |

| Seed | ID | Function | Noise | Scale | Intervention |
|---:|---:|---:|---:|---:|---:|
| 1593 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1594 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Each run used 1,024 extraction worlds, 1,024 mapping-fit worlds, 3,072 ordinary
validation worlds, five 512-world guard partitions, and five 1,024-world final
tests: 12,800 worlds in total. The guards used frozen-teacher predictions; truth
labels remained diagnostic only. Final tests were generated after selection
and were disjoint from extraction, fitting, validation, and guard data.

The result shows that the implemented gate can abstain rather than reproduce
the 0.8 OOD regressions. It does not show that causal feature evidence improves
program synthesis. Both runs failed the required program-change and 0.5
percentage-point gain conditions. Exact decompilation, general OOD safety, and
causal identification remain unproved.

Machine-readable evidence: `validation/ood_guided_acceptance.json`.