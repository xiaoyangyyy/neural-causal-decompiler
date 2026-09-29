# Exact state-complexity bounds for large continuous ReLU products

The scalar benchmark's exact nine-state minimum composes into a theorem for
the frozen coordinate-separable `d`-dimensional ReLU networks. At exact
binary64 tolerance `epsilon=0.101`, maximum-coordinate observation error,
all initial states in `[0,1]^d`, every continuous action in `[0,1]^d`, and
unbounded horizon, the deterministic finite-state minimum satisfies

    9 * 4^(d-1) <= K_d <= 9^d.

The lower bound uses `4^(d-1)` mutually output-separated controlled
invariant slices, each carrying the complete scalar nine-state obstruction.
The upper bound is an implicit product of the exact scalar nine-state
realization. The verifier proves fixed ReLU phases, extracts the exact realized affine
map independently of hidden neuron order, and replays the 40,545-record
rational scalar proof. It does not enumerate the exponentially large
abstract state spaces.

| Frozen dimension | Exact lower bound | Exact upper bound | Network coefficients checked |
|---:|---:|---:|---:|
| 2 | 36 | 81 | 42 |
| 8 | 147,456 | 43,046,721 | 552 |
| 32 | `9*4^31` (about 4.15e19) | `9^32` (about 3.43e30) | 8,352 |
| 128 | `9*4^127` (about 2.61e77) | `9^128` (about 1.39e122) | 131,712 |

The 128D model was frozen to JSON (about 2 MB) before verification. A
hidden-permuted and rescaled 8D network retains the same certified bounds. The
acceptance also rejects a coupled 2D transition and the separately trained
128D network, rather than transferring the product bound to them. This is a
proof of exponential finite-state complexity under exact coordinate
separability; it is not a trained-network complexity claim or an exact
minimum-state result above one dimension.

Evidence: [method](docs/SEPARABLE_PRODUCT_BOUNDS_METHOD.md),
[acceptance](validation/separable_product_bounds_acceptance.json), and
[frozen 128D certificate](runs/separable_product_bounds_v1/d_128/certificate.json).
Reproduce the frozen replay with

```powershell
python -m scripts.acceptance_separable_product_bounds --verify
```

Release 0.45.0 passed all 199 regression tests. The isolated wheel replayed
all four frozen dimensions, the hidden permutation-and-rescaling control,
and the coupled/trained rejection controls; all 105 installed Python modules
matched the source bytes. Evidence:
[wheel status](validation/wheel_v45_run/status.json). Wheel SHA-256:
`b0c8fc613c396fc3ea03074526feb1deeae83a7024ff1b8858c6ace72e7633af`.
