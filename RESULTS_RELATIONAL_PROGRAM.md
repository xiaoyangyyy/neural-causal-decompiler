# Relational context in extracted graph programs

The protocol was frozen in `docs/RELATIONAL_PROGRAM_PROTOCOL.md` before seeds 693 and 694 were generated. Six programs and all reported metrics were independently replayed; see `validation/relational_program_replay.json`.

## Primary paired exact-graph fidelity

| Seed | Frozen teacher | Local composed | Relational primitive | Difference | Conservative 95% interval |
|---:|---|---:|---:|---:|---|
| 693 | without_relations | 21.1% | 18.3% | -2.8% | [-11.6%, +6.0%] |
| 693 | with_relations | 16.7% | 17.2% | +0.5% | [-8.2%, +9.3%] |
| 694 | without_relations | 20.3% | 17.0% | -3.3% | [-12.1%, +5.4%] |
| 694 | with_relations | 17.1% | 18.0% | +0.9% | [-7.8%, +9.7%] |

Each comparison contains 960 independently generated world units in the fixed mixture of three node sizes and five environments. Intervals treat the paired per-world difference as bounded in [-1, 1]; they do not correct across the four reported comparisons.

## Pooled behavioral metrics

| Seed | Teacher | Program | All-pair fidelity | Active-pair fidelity | Exact-graph fidelity | Program truth accuracy | Complexity |
|---:|---|---|---:|---:|---:|---:|---:|
| 693 | without_relations | local_composed | 76.6% | 50.6% | 21.1% | 9.5% | 19 |
| 693 | without_relations | local_primitive | 74.4% | 43.9% | 19.0% | 9.9% | 17 |
| 693 | without_relations | relational_primitive | 73.2% | 44.4% | 18.3% | 9.6% | 17 |
| 693 | with_relations | local_composed | 72.4% | 44.7% | 16.7% | 9.2% | 11 |
| 693 | with_relations | local_primitive | 71.7% | 41.7% | 17.2% | 7.3% | 9 |
| 693 | with_relations | relational_primitive | 71.7% | 41.7% | 17.2% | 7.3% | 9 |
| 694 | without_relations | local_composed | 76.5% | 50.3% | 20.3% | 10.3% | 19 |
| 694 | without_relations | local_primitive | 74.3% | 43.8% | 18.0% | 10.9% | 17 |
| 694 | without_relations | relational_primitive | 73.3% | 44.5% | 17.0% | 10.1% | 17 |
| 694 | with_relations | local_composed | 71.7% | 44.6% | 17.1% | 9.2% | 11 |
| 694 | with_relations | local_primitive | 71.5% | 42.3% | 18.0% | 8.3% | 9 |
| 694 | with_relations | relational_primitive | 71.5% | 42.3% | 18.0% | 8.3% | 9 |

## Interpretation

- The relational program for the relation-enabled teacher selected contextual primitives, so the added language was available to synthesis. Its exact-graph fidelity increased only 0.5 and 0.9 percentage points across the two seeds; both conservative intervals include zero.
- For the teacher without relation biases, relational primitives reduced exact-graph fidelity by 2.8 and 3.3 percentage points. This is a useful negative control: a larger language can hurt finite-budget MDL search.
- These results do not explain the attention computation fully. Simple incidence means are too weak, or the six-split tree and candidate pruning cannot exploit them.
- Program truth accuracy remains a separate metric. Improving agreement with a frozen teacher does not imply better causal graph recovery.
- Seeds 693/694 have now been examined and cannot be reused as untouched confirmation data for a revised method.
