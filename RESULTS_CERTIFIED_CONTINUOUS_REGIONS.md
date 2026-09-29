# Certified continuous initial-region results

Version 0.30 extends continuous certification from fixed initial points to
axis-aligned initial-state boxes. Three formal profiles cover 8/32/64 state
dimensions, 2/3/4 continuous controls, and horizons 3/5/10. Every state
coordinate varies independently within radius 0.001 around its center.

All learned models fit their targets with maximum error below 1.49e-12. The
largest model has 13,528 parameters.

## Near-region uniform upper proofs

The within threshold is 2*epsilon = 0.02.

| State dim / horizon | Relational upper | Relational leaves | Independent upper | Independent leaves |
|---|---:|---:|---:|---:|
| 8 / 3 | 0.016500 | 1 | 0.099342 | 8 |
| 32 / 5 | 0.016500 | 1 | 0.162324 | 8 |
| 64 / 10 | 0.016500 | 1 | 0.207183 | 8 |

The relational method certifies all three complete product domains within
tolerance using one leaf. Independent IBP uses its full eight-leaf budget and
leaves all three unresolved.

## Far-region robust separation

For every profile, the stored uniform action witness has certified lower bound
0.238500 against threshold 0.20. It therefore separates every state in the left
initial box from every state in the right initial box. Both relational and
independent methods close these far-region cases; the relational certificates
also keep their upper bounds near 0.241500.

## Verification

- Formal run: runs/certified_continuous_regions_seed8701
- Formal runtime: 14.642 seconds
- Independent replay: 3 models and 12 certificates verified
- Acceptance: validation/certified_continuous_regions_acceptance.json
- Method: docs/CONTINUOUS_REGION_METHOD.md

## Interpretation

This closes the previously open implementation gap between point-state
certificates and local continuous initial-state regions for the declared stable
ReLU regime. The result is uniform over uncountably many state pairs and action
words inside each product domain.

It does not close the broader realization problem over an entire continuous
state space. The regions are local boxes, the dynamics retain stable paired
ReLU phases, and no globally minimal quotient is constructed.

## Release verification

Version 0.30.0 passed 134 regression tests. The isolated wheel matched every
source module and completed all 12 certified generation/replay commands.
Wheel SHA-256:
92c2a3b66e590cd224f3bb453740dba20878744b7edf1a0fce91684c803cb6b2.
Evidence: validation/wheel_v30_run/status.json.