# Calibrated active graph decoder protocol

## Question

Version 0.25 learned strong active-intervention edge probabilities, while
version 0.26 showed that residual graph errors dominate end-to-end SCM error.
This diagnostic asks whether the legacy multiclass argmax decoder is
miscalibrated for edge presence.

## Frozen comparison

Both branches use the same version 0.25 active teacher checkpoints,
probabilities, worlds, and features. The control is the existing multiclass
argmax decoder. The candidate first pair-symmetrizes probabilities, retains an
edge when `1 - p(no edge) >= 0.55`, selects its direction/ambiguity from the
three conditional edge classes, and applies the same acyclic projection.

The threshold 0.55 was fixed from development splits only. No test label,
mechanism truth, or equation metadata chooses or repairs a graph. Formal runs
use the held-out test predictions from seeds 4793 and 4794.

## Frozen acceptance rule

Pass only if:

1. all saved probabilities and decoded metrics replay exactly;
2. candidate exact-graph accuracy is higher in each seed and gains at least
   0.5 pooled percentage points;
3. candidate mean SHD is lower in each seed;
4. candidate skeleton F1 is no lower in each seed;
5. pooled directed-target accuracy declines by no more than two percentage
   points; and
6. all 3/5/8-node and five-environment cells are retained.

Passing would establish calibration of this fixed interventional decoder on the
frozen benchmark. It would not establish observational identifiability,
historical-teacher fidelity, or general graph recovery.