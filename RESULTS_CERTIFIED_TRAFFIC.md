# Results: certified quantized traffic realization

## Outcome

Two learned traffic transition networks were trained on independently generated
four-link traffic trajectories. Their five labelled controls were normal
operation, demand increase, a link-1 incident, a link-2 red-signal regime, and
a link-3 closure. K-means supplied ten quantized latent traffic states per
network; a ReLU model learned all 50 codebook-state/action transitions.

The trained networks, rather than the traffic simulator or quantization target,
were enumerated as ground-truth systems. Response-only recovery reduced 20
concrete neural states to eight exact computational states and recovered the
oracle quotient in both seeds.

| Quantity | Result |
|---|---:|
| Formal seeds | 3701, 4710 |
| Simulated trajectory states | 10,240 |
| Neural state/action training pairs | 100 |
| Transition training accuracy | 100%, 100% |
| Concrete quantized neural states | 20 |
| Certified minimal states | 8 |
| Response queries | 220 |
| Exact certified recoveries | 2 / 2 |
| Budget-limited unresolved pairs | 30 |

Both upper certificates checked every codebook state and labelled action.
Both lower certificates replayed distinguishing control sequences for every
pair of abstract representatives. Full deterministic replay regenerated the
traffic samples and codebooks, retrained both neural networks, re-enumerated
their actual transitions, reran recovery, and reproduced all metrics.

## Boundary

This result gives the city application natural continuation semantics without
activation surgery. It certifies each learned **quantized** traffic model over
its ten-state codebook and the five declared discrete controls. It does not
certify the unquantized continuous latent space, real-world causal validity, or
controls outside this action alphabet. The traffic simulator creates training
data but is not the object whose mechanism is claimed to be recovered.

The separate continuous separation workflow certifies fixed state-pair queries
over bounded continuous controls. It does not retroactively turn this quantized
traffic experiment into a continuous-state realization certificate.

## Reproduction

    python -m ncd verify-certified-traffic runs/certified_traffic_seed3701

Machine-readable formal evidence is in
'validation/certified_traffic_acceptance.json'.
