# Factorized skeleton-orientation graph programs

## Question and frozen protocol

Version 0.13 tests whether the four-class local graph tree is weakened by class imbalance and by combining edge presence with orientation. The replacement is an executable two-stage program: a binary skeleton tree followed by a three-class orientation tree. Both ordered pair views must agree after a fixed direction swap; conflicts become undirected and directed cycles pass through deterministic acyclic projection.

The protocol was frozen before confirmation in `docs/FACTORIZED_GRAPH_PROGRAM_PROTOCOL.md`. Training used only frozen teacher predictions from source extraction worlds. The `and`/`or` skeleton aggregator was selected only on source refinement worlds. Truth graphs were excluded from fitting and selection.

## Confirmation coverage

| Run | Frozen source | New worlds | Teacher/program comparisons | Replay |
|---|---:|---:|---:|---|
| 2393 | relational 493 | 480 | 960 | passed |
| 2394 | relational 494 | 480 | 960 | passed |
| Total | two independent sources | 960 | 1,920 | passed |

Each run covers 3, 5, and 8 nodes, five environments, both graph teachers, 32 worlds per node/environment cell, and 96 samples per world. All 30 mode/size/environment strata per run and their family strata are stored in each run's `summary.json`.

## Primary teacher-fidelity results

| Scope | Local exact graph | Factorized exact graph | Delta | Local active-pair | Factorized active-pair |
|---|---:|---:|---:|---:|---:|
| Seed 2393 | 19.79% | 20.63% | +0.83 pp | 46.65% | 49.27% |
| Seed 2394 | 18.02% | 18.23% | +0.21 pp | 43.66% | 44.61% |
| Pooled | 18.91% | 19.43% | +0.52 pp | 45.23% | 47.07% |

The per-mode nondecline condition failed in both runs. In seed 2393, the with-relations mode fell from 20.00% to 19.17% exact fidelity. In seed 2394, the without-relations mode fell from 20.00% to 19.79%.

## Truth diagnostic

Pooled exact truth accuracy rose from 8.75% for the local tree to 9.53% for the factorized program. The frozen teachers themselves reached 14.01%. These are diagnostics rather than synthesis objectives and do not alter the decision.

## Decision

The preregistered rule failed. Both runs replayed and pooled active-pair fidelity did not decline, but exact graph fidelity declined in one mode in each seed and the pooled +0.52 percentage-point gain was below the required +2 points.

This is a negative behavioral-decompilation result. It supports the narrower finding that separating skeleton and orientation improves pair-level agreement on these runs, but that improvement does not reliably compose into exact teacher graphs. It does not establish internal circuit identity, general graph recovery, causal identification, or end-to-end neural-to-SCM decompilation. R9 remains incomplete.

Machine-readable acceptance: `validation/factorized_graph_acceptance.json`.
