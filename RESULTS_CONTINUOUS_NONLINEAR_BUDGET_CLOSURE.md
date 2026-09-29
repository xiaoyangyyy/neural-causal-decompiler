# Continuous nonlinear 28-leaf budget closure

This post-0.29 follow-up changes only the 64D, three-step leaf budget from 24 to
28. It uses the same model seed 7719, 3 continuous controls, epsilon 0.12,
1,024 training samples, state pair, and complete nine-dimensional action-word
box as profile 2 of the formal phase-crossing run.

This is a targeted follow-up chosen after inspecting the 24-leaf proof gap. It
is evidence that the earlier unresolved result was budget-limited; it is not an
independent confirmation experiment.

## Results

| Method | Near status | Upper bound | Leaves | Near runtime |
|---|---|---:|---:|---:|
| hybrid_best | certified within epsilon | 0.239735521 | 28 | 84.03 s |
| independent_best | certified within epsilon | 0.239735521 | 28 | 72.85 s |
| hybrid_widest | unresolved | 0.261212231 | 28 | 9.40 s |
| independent_widest | unresolved | 0.261212231 | 28 | 8.12 s |

The proof threshold is 2*epsilon = 0.24. Both best-bound methods therefore
close the uniform near-pair upper certificate. Both widest-coordinate baselines
remain unresolved at exactly the same leaf budget. All four far-pair methods
remain formally separated with witness lower bound approximately 0.189.

Every near leaf used independent IBP. The hybrid and independent results are
identical. The closure must therefore be attributed to the best-bound
partition and four additional leaves, not to relational propagation.

## Interpretation

A verified 64D, three-step phase-crossing upper certificate now exists for the
fixed initial-state pair and complete continuous action box. The original
24-leaf result remains a valid record of the smaller budget: its best upper
bound was 0.243000607, just above threshold.

The successful strategy is computationally expensive. best-bound evaluates
every candidate action coordinate and took roughly nine times as long as widest
splitting. This experiment does not show that greedy best-bound dominates other
partition rules, nor does it address continuous initial-state regions or global
minimal realization.

## Reproducibility

- Frozen follow-up config: validation/continuous_nonlinear_budget28_config.json
- Run: runs/certified_continuous_nonlinear_budget28_seed7719
- Acceptance: validation/continuous_nonlinear_budget28_acceptance.json
- Acceptance script: scripts/acceptance_continuous_budget_closure.py
