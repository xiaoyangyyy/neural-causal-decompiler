# Results: scaled continuous neural certification

## Outcome

The stable relational bound closed every declared state-pair query from 8 to
128 latent dimensions and from horizon 3 to horizon 20. The largest fitted ReLU
system has 52,002 parameters and five continuous controls.

| State dim | Action dim | Horizon | Parameters | Near relational UB | Near independent UB |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 3 | 306 | 0.015000 | 0.060616 |
| 32 | 3 | 5 | 3,600 | 0.015000 | 0.148986 |
| 64 | 4 | 10 | 13,528 | 0.015000 | 0.239716 |
| 128 | 5 | 20 | 52,002 | 0.015000 | 0.247005 |

At epsilon 0.01, every near pair satisfies `UB <= 2 epsilon` under the
relational verifier. Independent IBP leaves all four near pairs unresolved at
the declared eight-leaf budget. For the far pairs, the feasible control witness
has certified distance 0.24, so both methods certify separation at epsilon 0.1.

| Method | Separated | Certified within | Unresolved |
|---|---:|---:|---:|
| Stable relational bound | 4 | 4 | 0 |
| Independent IBP | 4 | 0 | 4 |

Mean measured certificate construction time was 1.24 seconds for the relational
method and 7.66 seconds for independent IBP. Runtime is descriptive rather than
part of the acceptance rule. All fitted models had maximum training error below
`1.8e-12`.

The result demonstrates that preserving the shared intervention dependency is
decisive. Independent boxes interpret common control variation as possible
between-run variation, and their upper bounds grow with dimension and horizon.
The relational recurrence cancels that common term once the paired ReLU phases
are certified.

## Verification and boundary

The independent replay regenerated all training samples, refitted all four
networks, reconstructed every pair, and checked 16 relational/IBP certificates.
The formal evidence is in `runs/certified_continuous_scale_seed4701` and
`validation/certified_continuous_scale_acceptance.json`.

These networks represent learned stable affine dynamics. The experiment does
not show that the same closure rate holds for phase-unstable ReLU networks,
uncountable initial-state regions, arbitrary nonlinear traffic models, or much
longer horizons. Those cases remain unresolved rather than being inferred from
this scaling curve.

Run and replay:

    python -m ncd certified-continuous-scale --output runs/scale --seed 4701
    python -m ncd verify-certified-continuous-scale runs/scale

The full regression suite passed 127 tests. The 0.28.0 wheel matched every
source module and its isolated installation, then executed and replayed all
four certified workflows. Wheel SHA-256:
`18343a2ead65601d11a7c2286a8083e4734e843cf0a95321c3008dc702580143`.
Release evidence is in `validation/wheel_v28_run/status.json`.
