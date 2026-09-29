# Shifted-grid continuous realization: smaller executable upper bounds

A nine-center relation gives a deterministic finite realization of the
frozen scalar ReLU dynamics with uniform output error at most 0.101 for
every initial state, every continuous action sequence, and every time.
Its two-dimensional product has 81 states.

| Dimension | Verified executable upper bound | Current lower bound | Resulting interval |
|---:|---:|---:|---:|
| 1 | 9 | 7 from multi-switch transition theorem | 7–9 |
| 2 | 81 | 25 from complete initial-output packing | 25–81 |

Across both profiles, independent replay checked 18 initial cells,
18 observation relations, 84 continuous action segments, and 310 packing
pairs. The largest outward-rounded transition bound is
0.10025000000000153 against an internal relation radius of 0.1005.
The largest observation bound is 0.10050000000000117 against output
tolerance 0.101.

Formal run: `runs/certified_shifted_realization_seed14701`.
Joint acceptance with the multi-switch lower proof:
`validation/certified_shifted_realization_acceptance.json`.
Method: `docs/SHIFTED_REALIZATION_METHOD.md`.

The result improves a certified upper bound, not a claim of exact
minimality. The network is the frozen coordinate-separable benchmark;
the argument does not establish a realization for arbitrary coupled or
trained high-dimensional networks.


Version 0.35.0 release verification: 159 regression tests passed.
The isolated wheel matched all source modules and completed all 24 certified
generation/replay CLI commands. Wheel SHA-256:
533f19e99f879a197d61080ecae0579c1ce1b3a56e1341f684fb9c2e3e7cfaaa.
Evidence: validation/wheel_v35_run/status.json.