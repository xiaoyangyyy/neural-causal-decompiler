# Results: pair-consistent factorized training

## Outcome

The frozen rule failed. Training on pair-symmetrized component logits improved
exact accuracy, skeleton F1, and SHD in seed 4594, but all three regressed in
seed 4593. Directed-target accuracy declined in both seeds. Pooled exact
accuracy fell by 0.66 percentage points and pooled direction accuracy fell by
2.11 points. R9 remains incomplete.

Each seed used 2,400 shared worlds. Both branches had the same 31,636 parameters,
identical initial weights, worlds, batch orders, optimizer, and inference path;
only the logits supplied to the factorized training loss differed. Both runs
passed complete retraining and zero-tolerance checkpoint replay.

## Aggregate metrics

| Seed | Loss input | Exact graph | Mean SHD | Skeleton F1 | Directed accuracy |
|---:|---|---:|---:|---:|---:|
| 4593 | raw | 16.67% | 2.6188 | 0.7959 | 0.4710 |
| 4593 | pair-consistent | 13.82% | 2.7313 | 0.7676 | 0.4392 |
| 4594 | raw | 15.00% | 2.7056 | 0.7777 | 0.4842 |
| 4594 | pair-consistent | 16.53% | 2.4535 | 0.7898 | 0.4737 |

Pooled exact accuracy changed from 15.83% to 15.17%, SHD from 2.6622 to
2.5924, skeleton F1 from 0.7868 to 0.7787, and directed-target accuracy from
0.4776 to 0.4564. Environment exact changes were +0.35 points ID, -2.43
function, -0.52 noise, -0.69 scale, and approximately zero intervention.

## Interpretation

The mismatch between raw component training and symmetrized inference is not a
useful standalone explanation of the direction gap. Averaging component logits
before the loss changes shared-representation gradients, removes ordered-pair
training signal, and can select earlier checkpoints that generalize worse to
function shifts. Future direction work should preserve raw ordered supervision
and use calibration, auxiliary objectives, or independent causal evidence
rather than replacing it with a symmetrized loss.

Machine-readable evidence is `validation/pair_consistent_acceptance.json`; the
frozen protocol is `docs/PAIR_CONSISTENT_TRAINING_PROTOCOL.md`.
