# Frozen graph failure audit

Exploratory diagnostics of previously verified predictions. No teacher or program is changed.
Confusion matrices and all five environments are in `validation/relational_failure_audit.json`.
Active pairs are those where either teacher or program predicts an edge; this is a diagnostic conditional subset, not a replacement primary metric.

| Seed | Nodes | Family | Mode | Worlds | Neural missed / extra / reversed / ambiguity | All-pair fidelity | Active-pair fidelity | Exact-graph fidelity |
|---|---|---|---|---:|---|---:|---:|---:|
| 493 | 3 | linear_gaussian | without_relations | 20 | 1 / 1 / 0 / 7 | 68.3% | 57.8% | 35.0% |
| 493 | 3 | nonlinear | without_relations | 76 | 27 / 9 / 9 / 0 | 79.8% | 71.8% | 52.6% |
| 493 | 3 | linear_gaussian | with_relations | 20 | 0 / 1 / 0 / 5 | 33.3% | 13.0% | 5.0% |
| 493 | 3 | nonlinear | with_relations | 76 | 27 / 15 / 10 / 2 | 76.3% | 67.9% | 44.7% |
| 493 | 5 | linear_gaussian | without_relations | 20 | 5 / 15 / 1 / 22 | 79.0% | 56.2% | 10.0% |
| 493 | 5 | nonlinear | without_relations | 76 | 62 / 36 / 12 / 1 | 90.8% | 71.0% | 40.8% |
| 493 | 5 | linear_gaussian | with_relations | 20 | 3 / 13 / 0 / 22 | 82.5% | 62.0% | 25.0% |
| 493 | 5 | nonlinear | with_relations | 76 | 57 / 41 / 12 / 0 | 91.2% | 73.3% | 44.7% |
| 493 | 8 | linear_gaussian | without_relations | 20 | 5 / 19 / 0 / 39 | 88.9% | 53.7% | 15.0% |
| 493 | 8 | nonlinear | without_relations | 76 | 94 / 92 / 20 / 2 | 92.8% | 65.9% | 10.5% |
| 493 | 8 | linear_gaussian | with_relations | 20 | 7 / 15 / 1 / 24 | 88.2% | 51.8% | 10.0% |
| 493 | 8 | nonlinear | with_relations | 76 | 91 / 121 / 25 / 5 | 93.1% | 69.3% | 14.5% |
| 494 | 3 | linear_gaussian | without_relations | 20 | 2 / 6 / 0 / 12 | 70.0% | 64.0% | 45.0% |
| 494 | 3 | nonlinear | without_relations | 76 | 20 / 19 / 11 / 0 | 80.3% | 74.0% | 56.6% |
| 494 | 3 | linear_gaussian | with_relations | 20 | 1 / 3 / 1 / 6 | 40.0% | 28.0% | 5.0% |
| 494 | 3 | nonlinear | with_relations | 76 | 12 / 23 / 12 / 3 | 75.4% | 69.9% | 38.2% |
| 494 | 5 | linear_gaussian | without_relations | 20 | 3 / 23 / 0 / 10 | 76.0% | 51.5% | 10.0% |
| 494 | 5 | nonlinear | without_relations | 76 | 58 / 61 / 21 / 10 | 84.7% | 63.7% | 27.6% |
| 494 | 5 | linear_gaussian | with_relations | 20 | 4 / 11 / 0 / 18 | 70.0% | 34.8% | 10.0% |
| 494 | 5 | nonlinear | with_relations | 76 | 63 / 56 / 18 / 7 | 86.2% | 66.7% | 28.9% |
| 494 | 8 | linear_gaussian | without_relations | 20 | 3 / 37 / 0 / 27 | 88.2% | 52.9% | 5.0% |
| 494 | 8 | nonlinear | without_relations | 76 | 97 / 126 / 25 / 1 | 90.0% | 59.2% | 11.8% |
| 494 | 8 | linear_gaussian | with_relations | 20 | 4 / 24 / 2 / 20 | 85.5% | 36.2% | 0.0% |
| 494 | 8 | nonlinear | with_relations | 76 | 87 / 113 / 27 / 7 | 91.4% | 64.0% | 13.2% |
