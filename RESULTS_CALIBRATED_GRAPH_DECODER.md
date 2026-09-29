# Calibrated graph decoder results

Protocol: `docs/CALIBRATED_GRAPH_DECODER_PROTOCOL.md`.

Both 480-world confirmation runs passed complete replay. The preregistered graph-decoding improvement rule **failed**.

| Seed | Argmax exact graph | Calibrated exact graph | Delta | Argmax pair SHD | Calibrated pair SHD |
|---:|---:|---:|---:|---:|---:|
| 1993 | 13.96% | 14.58% | +0.63% | 3.417 | 2.623 |
| 1994 | 13.85% | 15.83% | +1.98% | 3.394 | 2.705 |

Pooled exact-graph accuracy increased by 1.30%, below the frozen 2 percentage-point requirement. Pair SHD decreased in both seeds, and pooled pair accuracy and skeleton F1 improved. Pooled directed-target accuracy decreased, showing that the threshold mainly improves sparsity/skeleton decisions while discarding some correct orientations.

This is a reliable negative result for the fixed calibration rule. It does not change symbolic programs and does not solve R9 or end-to-end SCM recovery. Seeds 1993/1994 are now historical evidence and cannot be reused for method selection.

Machine-readable evidence: `validation/calibrated_graph_decoder_acceptance.json`.
