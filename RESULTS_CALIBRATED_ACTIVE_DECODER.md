# Results: calibrated active graph decoder

## Outcome

The frozen rule narrowly failed. Replacing multiclass argmax with a fixed 0.55
edge-presence threshold improved exact accuracy, SHD, and skeleton F1 in both
formal seeds, while pooled directed accuracy declined by only 0.54 percentage
points. Pooled exact accuracy gained 0.382 points, below the predeclared
0.5-point requirement.

The candidate uses the same frozen active teacher probabilities as version
0.25. Its threshold was fixed from development splits; no formal test label,
mechanism truth, or equation metadata selected or repaired a graph. All 30
node-size/environment cells replayed from saved probabilities.

## Aggregate metrics

| Seed | Decoder | Exact graph | Mean SHD | Skeleton F1 | Directed accuracy |
|---:|---|---:|---:|---:|---:|
| 4793 | legacy argmax | 49.65% | 1.3868 | 0.9146 | 0.8647 |
| 4793 | threshold 0.55 | 50.14% | 1.3556 | 0.9162 | 0.8589 |
| 4794 | legacy argmax | 51.67% | 1.0389 | 0.9246 | 0.8806 |
| 4794 | threshold 0.55 | 51.94% | 0.9993 | 0.9270 | 0.8756 |

Pooled exact accuracy increased from 50.66% to 51.04%, SHD fell from 1.2128 to
1.1774, and skeleton F1 increased from 0.9196 to 0.9216. The largest SHD
reduction occurred at eight nodes and under scale shift. Exact accuracy
declined under function and noise shifts, which confirms that a single global
threshold does not solve the remaining distribution-specific errors.

## Interpretation

The legacy decoder is slightly miscalibrated for edge presence, but calibration
is a secondary bottleneck. The remaining work should model directness and
distribution shift rather than tune another global threshold. This result does
not establish observational identifiability, historical-teacher fidelity, or
general graph recovery.

Machine-readable evidence is
`validation/calibrated_active_decoder_acceptance.json`; the frozen protocol is
`docs/CALIBRATED_ACTIVE_DECODER_PROTOCOL.md`.