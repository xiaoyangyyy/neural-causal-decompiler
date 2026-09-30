# Independent-world check of the inferred-role mechanism hybrid

The rule chosen after one development world does **not** reproduce that world's
0.01 local mechanism gate on the first pre-registered confirmation cell.
All ten independent seed-8301, three-node, test-id worlds exceeded 0.01.
The independent checker regenerated the source worlds and observed batches,
replayed the frozen graph teacher, verified model/parent hashes, and recomputed
the local and paired-oracle-noise rollout metrics.

| Scope | Verified worlds | Inferred graph exact | Local gate passes | Hybrid max local normalized MSE |
|---|---:|---:|---:|---:|
| Seed 8301, 3 nodes, test-id | 10 | 8 | 0 | 0.019767 to 1.550574 |
| Seed 8301, 5 nodes, test-id, index 0 | 1 | 0 | 0 | 1.936930 |
| Seed 8301, 8 nodes, test-id, index 0 | 1 | 0 | 0 | 4.847737 |

The 3-node cell contains eight worlds with the **correct inferred graph**;
none passes the local gate. Most worst-case rows in this cell belong to a
node with inferred parents using the mixed-intervention checkpoint. The
failure is therefore not explained solely by graph mistakes. Some other
worlds do have graph mistakes, which must be reported separately.

This is a test of **one frozen candidate rule** under the stated sample,
training and architecture budget. A failed candidate does not refute the
existence of a different successful neural mechanism or symbolic program.
The rollout uses the true exogenous noise as an oracle diagnostic; it does
not demonstrate recovery of a noise distribution or the full intervention
distribution. The 12 verified worlds are not the complete 300-world cohort,
and no 99% statistical guarantee is claimed from this partial sample.

Protocol and methods: docs/INTERVENTIONAL_ROLE_CONFIRMATION_V1.md and
validation/interventional_role_confirmation_protocol_v1.json. The first
cell's hash-bound summary is
validation/interventional_role_confirmation_first_cell_v1.json.
Every published unit has candidate observations, truth-only metadata,
frozen checkpoint records, a result, and an independent verification receipt
under runs/interventional_role_confirmation_v1/units. The remaining declared
worlds continue under the same frozen protocol. The original R0-R13 ledger
remains open: 0 proved, 2 refuted, 36 unresolved.
