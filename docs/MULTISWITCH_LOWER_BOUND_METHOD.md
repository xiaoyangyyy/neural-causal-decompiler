# Exact multi-switch transition lower-bound method

This strengthens the one-switch theorem in
`TRANSITION_AWARE_LOWER_BOUND_METHOD.md`. It is independently certified in version 0.35.

For a deterministic m-state realization of scalar
`F(x,a)=lambda*x+beta*a`, `O(x)=x`, let `I_q` be the interval hull of
all concrete states reachable while paired with abstract state q. Let
`L=2*epsilon`. The hulls cover [0,1], each has length at most L, and total
overlap redundancy is at most `m*L-1`.

Some source hull has width `D>=1/m`. For a fixed target hull `I_p`,
the set of actions for which the whole moving source window
`J(a)=lambda*I_q+beta*a` fits inside `I_p` is a closed interval of
length at most `(L-lambda*D)/beta`. The target-action intervals cover
[0,1]. Therefore at least

    k = ceil(beta / (L-lambda*D))

distinct target hulls are needed. A connected chain of k target-action
intervals has k-1 adjacent intersections. At each intersection, both
target hulls contain the same window of length lambda*D. Adding the k
target hulls one at a time thus contributes total overlap redundancy at
least `(k-1)*lambda*D`.

The weakest values allowed by the global cover are `D=1/m` and

    k_min = ceil(beta / (L-lambda/m)).

Consequently, m states are impossible whenever

    m*L-1 < (k_min-1)*lambda/m,

provided `L>lambda/m` and `k_min>=2`. If `L<=lambda/m`, a target
cannot hold the source window at all.

For the frozen scalar ReLU benchmark, with exact serialized coefficients
`lambda=1/2`, `beta=float64(0.4)`, and
`epsilon=float64(0.101)`:

| m | Required targets | Forced overlap | Available overlap |
|---:|---:|---:|---:|
| 5 | 4 | 0.300000 | 0.010000 |
| 6 | 4 | 0.250000 | 0.212000 |
| 7 | 4 | 0.214286 | 0.414000 |

The exact-rational verifier in `ncd/multiswitch_lower.py` extracts
`lambda` and `beta` from the serialized ReLU network after checking that
all hidden neurons are active on the complete state-action cube. It checks
every size exclusion and stored target count independently of the generator.
The formal run is `runs/certified_multiswitch_lower_seed13701`; the joint
acceptance `validation/certified_multiswitch_lower_acceptance.json`
replays the previous lower proof and the executable 10-state upper model.
This certifies 7 <= K <= 10. It does not identify the exact minimum.

