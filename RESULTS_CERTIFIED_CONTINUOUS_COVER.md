# Certified global continuous behavioral-cover results

Version 0.31 covers complete uncountable state domains rather than selected
local boxes. The declared domain is the full unit cube, controls range over the
full unit cube at every step, horizon is three, and epsilon is 0.101.

## Closed complexity intervals

| State/action dim | Continuous domain | Cover cells | Packing points | Packing pairs | Certified result |
|---:|---|---:|---:|---:|---:|
| 1 | [0,1] | 5 | 5 | 10 | K = 5 |
| 2 | [0,1]^2 | 25 | 25 | 300 | K = 25 |

All 30 cell certificates prove uniform epsilon approximation for every state
in the cell and every continuous action word. All 310 packing-pair certificates
prove distance greater than 2*epsilon. The upper and lower bounds therefore
match in both dimensions.

The complete formal workflow verifies 340 constituent certificates. Its
deterministic generation runtime is 1.630 seconds.

## What this establishes

For the declared contractive ReLU systems, the exact minimum number of
finite-horizon response representatives is 5 in one dimension and 25 in two
dimensions. The upper certificate covers every point of the unit interval or
unit square; this is no longer a finite collection of selected initial states
or local neighborhoods.

The exponential change from 5 to 25 also exposes the expected covering-number
growth with state dimension.

## What remains open

The representatives form a behavioral codebook. They are not required to be
closed under neural transitions, so this is not yet a finite-state causal
quotient. Infinite-horizon error accumulation, phase-crossing global domains,
higher-dimensional cover explosion, and minimal transition-closed realization
remain open.

## Reproducibility

- Formal run: runs/certified_continuous_cover_seed9701
- Acceptance: validation/certified_continuous_cover_acceptance.json
- Method: docs/CONTINUOUS_COVER_METHOD.md
- Profiles replayed: 2
- Constituent certificates verified: 340

## Release verification

Version 0.31.0 passed 138 regression tests. The isolated wheel matched every source module and completed all 14 certified generation/replay commands. Wheel SHA-256: 4e7664edff404e47149ba94f049cd05cedb85025bd8a468491dce6646cff512f. Evidence: validation/wheel_v31_rebuild_run/status.json.
