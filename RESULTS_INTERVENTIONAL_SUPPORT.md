# Exact one-step interventional support from behavior queries

This experiment closes a specific gap left by the 0.39 grid synthesizer:
the candidate causal edges are proposed from interventions on the frozen
network's callable transition, without reading its weights. A separate
checker certifies each proposed edge by an exact rational input-pair witness
and certifies every excluded edge by the absence of a nonzero-weight path.
A pair with a possible path and no witness is reported as unresolved.

The formal study covers 24 fixed cases. Twenty learned-local and
fixed-dictionary/affine trained networks span 8, 32, 64, and 128 state
dimensions. Four constructed controls test coupling, endpoint cancellation,
finite-grid miss, and exact path cancellation.

| Family | Cases | Query calls | Certified present | Certified absent | Unresolved |
|---|---:|---:|---:|---:|---:|
| Learned local phase, two frozen seeds | 8 | 2,400 | 1,856 | 42,208 | 0 |
| Fixed-dictionary nonlinear, two frozen seeds | 8 | 2,400 | 1,856 | 42,208 | 0 |
| Trained affine, four frozen profiles | 4 | 1,230 | 1,472 | 21,104 | 0 |
| Coupled phase-crossing 2D | 1 | 20 | 6 | 2 | 0 |
| Interior tent | 1 | 10 | 1 | 1 | 0 |
| Narrow tent between probe levels | 1 | 10 | 0 | 1 | 1 |
| Exact path cancellation | 1 | 10 | 0 | 1 | 1 |
| **Total** | **24** | **6,080** | **5,191** | **105,525** | **2** |

All 20 frozen trained networks, the coupled network, and the interior-tent
control have complete support certificates: every possible path has a
verified edge witness and every other pair has no path. The 128-dimensional
learned-local cases each certify 512 edges and 16,128 nonedges; the
128-dimensional trained-affine case certifies 896 edges and 16,128 nonedges.
All proposals and certificates regenerate and replay.

The two unresolved controls are scientifically important. The narrow tent
has a genuine input dependency strictly between the five sampled levels;
finite intervention observations miss it, while the verifier refuses to
call it absent. The cancellation network has nonzero internal paths but
constant output; the verifier again refuses to infer absence from a
nonzero-path graph. These controls show that neither the finite query grid
nor structural paths alone establish a complete behavioral graph in
general. A proof-producing function-level absence method is still needed.

The result establishes an exact *coordinate-specific one-step support graph*
for 22 cases, not latent-variable discovery or an optimal causal quotient.
The network weights are trusted as frozen input to the independent
verifier, so this is verifier-guided behavioral discovery, not
black-box-only identification. The study reuses prior synthetic networks;
it does not establish performance on external systems or close original
R4/R5/R8/R9/R10.

Method and counterexample argument:
`docs/INTERVENTIONAL_SUPPORT_METHOD.md`. Evidence:
`runs/interventional_support_global_v1/summary.json` and
`validation/interventional_support_acceptance.json`.
Run `python -m scripts.acceptance_interventional_support` to replay all
cases. The 0.40.0 release check is in
`validation/wheel_v40_run/status.json`: 174 regression tests passed,
source/wheel/isolated-install modules matched byte for byte, and five
installed-package commands passed. Wheel SHA-256:
`e7d5e162c25baa8f630f0acb323b0d124393d088595ec938048634f8ed3653e3`.
