# Generic coupled continuous realization: 0.33

This stage removes the exact-network restriction from the 0.32
infinite-horizon verifier. A candidate finite machine is now checked against
the serialized transition and observation networks supplied with the run.
A separate checker evaluates every stored target and every continuous box
without trusting the generator's nominal quantization choice.

The formal two-dimensional ReLU benchmark couples state coordinates and
crosses an internal ReLU phase boundary. Its 100-state finite machine has a
certified uniform observation error at most 0.12 under every continuous
action sequence and at every time. The internal simulation radius is 0.115.
The largest outward-rounded transition bound is 0.1092500000000016.

| Evidence | Count |
|---|---:|
| Initial state cells checked | 100 |
| Relation-observation checks | 100 |
| Continuous state-action transition boxes checked | 10,000 |
| Initial-output packing pairs checked | 300 |
| Size lower bound | 25 |
| Executable upper bound | 100 |

The accepted run is `runs/certified_generic_grid_seed11701`; its formal
acceptance is `validation/certified_generic_grid_acceptance.json`.
The direct method and limitations are in
`docs/GENERIC_GRID_REALIZATION_METHOD.md`.

The result strengthens the project from a hardcoded separable system to a
checker applicable to supplied ReLU systems. A different network with a
changed coupling weight also passes the same checker in a positive-control
test. This remains a constructed 2D benchmark, not a learned large model.
The state bound is still 25–100, so minimality remains open. A failed grid
is reported as unresolved and does not prove that no finite realization
exists.

