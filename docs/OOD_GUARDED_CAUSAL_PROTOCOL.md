# OOD-guarded causal synthesis protocol

Status: frozen after the 1493/1494 conservative-gate failure and before any
1593/1594 artifact is generated.

## Fixed method

Feature interventions, mapping controls, causal support, expression search,
ordinary validation score, complexity weight, and causal-support weight remain
identical to CAUSAL_FEATURE_GUIDANCE_PROTOCOL.md.

After the historical baseline is selected, generate five selection-only guard
partitions using seed + 20000:

- test_id
- test_function
- test_noise
- test_scale
- test_intervention

Each guard contains 512 worlds in formal runs. A causal candidate is eligible
only when:

1. its ordinary validation fidelity-minus-complexity score is at least the
   historical baseline score, with numerical tolerance 1e-12; and
2. its teacher fidelity is at least the historical baseline fidelity in every
   guard partition, again with tolerance 1e-12.

Among eligible candidates, maximize the existing guided score. Returning the
historical candidate is a valid abstention. Guard truth labels are retained
only for diagnostics; selection uses frozen-teacher predictions.

Final test worlds use the original seed, are generated only after selection,
and are identity-disjoint from extraction, mapping-fit, validation, and guard
worlds.

## Seeds and budgets

- Development/integration: 1592.
- Confirmation A/B: 1593 and 1594 with frozen joint sources 191 and 192.
- Per confirmation: 1,024 extraction, 1,024 mapping-fit, 3,072 ordinary
  validation, five times 512 guard, and five times 1,024 final-test worlds.
- Mapping uses 384 single-feature fit pairs and 768 validation pairs covering
  all 105 compatible masks.
- Seeds 1393/1394 and 1493/1494 are method-development evidence and cannot be
  confirmation evidence.

## Decision rule

Call the method a replicated safe improvement only if:

1. both runs pass full independent regeneration replay;
2. selected candidates satisfy the ordinary and all-five-guard noninferiority
   gates;
3. final teacher fidelity is no lower than baseline in every final environment
   in both seeds;
4. at least one seed changes program, uses positive causal support, and gains
   at least 0.5 percentage points in mean final fidelity; and
5. the equally weighted mean gain across both seeds is at least 0.5 percentage
   points.

Abstention contributes zero change. Ground-truth accuracy is diagnostic and
does not enter gates or the decision. No threshold or environment subset is
changed after confirmation begins.
