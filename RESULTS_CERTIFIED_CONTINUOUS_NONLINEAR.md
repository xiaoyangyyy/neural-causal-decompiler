# Certified phase-crossing continuous results

The formal seed-5701 run verifies learned ReLU dynamics whose declared domains
cross three activation boundaries. All three fitted models have maximum
training error below 2.11e-12. The largest model has 64 state variables, 3
continuous controls, horizon 3, and 13,724 parameters.

## Near-pair certificates

Each cell reports status / upper bound / leaves.

| Profile | hybrid_best | hybrid_widest | independent_best | independent_widest |
|---:|---|---|---|---|
| 16D, H=1 | within / 0.059187 / 27 | within / 0.059187 / 46 | within / 0.059187 / 27 | within / 0.059187 / 70 |
| 32D, H=2 | unresolved / 0.163613 / 64 | within / 0.156216 / 62 | unresolved / 0.163613 / 64 | within / 0.156216 / 62 |
| 64D, H=3 | unresolved / 0.243001 / 24 | unresolved / 0.263491 / 24 | unresolved / 0.243001 / 24 | unresolved / 0.263491 / 24 |

All twelve far-pair configurations are formally separated. Their certified
witness lower bound is approximately 0.189, above 2*epsilon=0.16.

The one-step hybrid/widest certificate uses 12 relational-stable leaves and 34
fallback leaves, compared with 70 leaves for independent/widest. This is direct
evidence that stable subregions remain useful even when the complete domain
crosses activation boundaries. The greedy split closes the one-step case in 27
leaves, but fails to close the two-step case within 64 leaves while the widest
strategy closes it in 62. Greedy one-step bound reduction therefore does not
dominate a simple balanced partition.

At three steps, every method remains unresolved under the fixed 24-leaf budget.
The best certified upper bound is 0.243001 versus the within-tolerance threshold
0.24. This is a narrow but real proof gap; sampling cannot close it.

A targeted post-0.29 follow-up increased only this budget to 28 leaves.
best-bound then closed the certificate at 0.239735521, while widest splitting
remained unresolved at 0.261212231. Every leaf used independent IBP, so this
closure comes from the partition and added budget rather than relational
propagation. See RESULTS_CONTINUOUS_NONLINEAR_BUDGET_CLOSURE.md.

## Reproducibility

- Formal run: runs/certified_continuous_nonlinear_seed5701
- Runtime: 216.318 seconds
- Independent replay: 3 profiles and 24 certificates verified
- Acceptance gate: validation/certified_continuous_nonlinear_acceptance.json
- Method: docs/CONTINUOUS_NONLINEAR_METHOD.md

## What this establishes

The verifier is sound across genuine control-dependent ReLU phase changes. It
can combine exact relational reasoning on certified stable subregions with IBP
fallback elsewhere, and can close selected 16D one-step and 32D two-step
uniform action-box claims. It also exposes two limitations rather than hiding
them: the greedy split heuristic can choose worse long-term partitions, and the
64D three-step near claim remains unresolved at the declared budget.

This does not certify a quotient over an uncountable initial-state region, prove
a globally minimal continuous realization, or establish performance on natural
traffic networks. Those remain separate open problems.

## Release verification

Version 0.29.0 passed 130 regression tests. The isolated wheel matched every
source module and completed all ten certified generation/replay commands.
Wheel SHA-256:
6a097cafe7cdb96734a9c2e67f2c2dd5cf92847de669e96bff3b872afa1c721a.
Evidence: validation/wheel_v29_run/status.json.