# Conservative causal-guided synthesis protocol

Status: frozen after the 1393/1394 failure audit and before any 1493/1494
artifact is generated.

## Motivation and fixed change

The compensatory causal score improved seed 1393 but degraded all five test
environments for seed 1394. Failure analysis found that the 1394 causal
candidate had a lower validation objective than the historical candidate and
was selected only because causal support compensated for that observable loss.

This protocol changes exactly one selection rule. Candidate generation,
feature interventions, controls, mapping training, weights, world budgets, and
final metrics remain those in CAUSAL_FEATURE_GUIDANCE_PROTOCOL.md.

Let base(p) be validation teacher fidelity minus 0.001 times program
complexity. First select the historical baseline from candidates produced by
unaligned searches. A candidate is eligible for causal selection only when
base(p) is at least the historical baseline base score (tolerance 1e-12 for
floating comparison). Among eligible candidates, maximize base(p) plus 0.02
times mean causal support. Thus causal evidence may choose among validation
noninferior programs but cannot compensate for a lower observable validation
objective. Returning the historical program is an explicit valid abstention.

## Seeds and data

- Integration/development: seed 1492.
- Confirmation A/B: seeds 1493 and 1494 with frozen joint sources 191 and 192.
- Each confirmation uses 1,024 extraction, 1,024 mapping-fit, 3,072 validation,
  and five times 1,024 untouched final-test worlds.
- Mapping uses 384 single-feature fit pairs and 768 validation pairs covering
  all 105 compatible single/two-feature masks.
- Seeds 1393/1394 are historical method-development evidence and cannot be
  confirmation evidence.

## Decision rule

Call the conservative method replicated safe improvement only if:

1. both runs pass complete independent replay;
2. guided final teacher fidelity is no lower than baseline in any of the five
   environments in either seed;
3. at least one seed selects a different program, uses positive causal support,
   and improves mean final fidelity by at least 0.5 percentage points;
4. mean fidelity pooled equally across the two seeds improves by at least 0.5
   percentage points;
5. all selected programs satisfy the validation noninferiority gate.

Abstention counts as zero change, not a selection failure. Truth accuracy is
reported separately and never substitutes for teacher fidelity. The rule is
not changed after either confirmation result is observed.
