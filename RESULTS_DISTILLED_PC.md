# Teacher-distilled PC program results

Protocol: `docs/DISTILLED_PC_PROGRAM_PROTOCOL.md`.

Both 480-world confirmation runs passed complete replay. The preregistered behavioral-decompilation rule **failed**.

| Seed | Local exact fidelity | PC exact fidelity | Delta | Local active fidelity | PC active fidelity |
|---:|---:|---:|---:|---:|---:|
| 2193 | 20.10% | 10.52% | -9.58% | 50.07% | 23.74% |
| 2194 | 18.85% | 9.90% | -8.96% | 46.82% | 26.69% |

Pooled PC exact teacher fidelity was 10.21%, versus 19.48% for the frozen local tree (-9.27%). Pooled active-pair fidelity also fell.

As a separate truth diagnostic, PC exact-graph accuracy exceeded the local tree by +2.60%. Thus the explicit causal baseline can be more correct while being a worse decompilation of the network. This is evidence against claiming that these teachers implement the tested Fisher-PC algorithm.

Every PC execution stores CI tests, separating sets, the learned skeleton, the collider-oriented intermediate PDAG, and the final Meek-closed PDAG. The implementation capability is complete, but R9 behavioral decompilation remains incomplete.

Machine-readable evidence: `validation/distilled_pc_acceptance.json`.
