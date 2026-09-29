# Results: certified finite interventional realization

## Outcome

The exact finite phase closes its declared certification objective. Twelve
deterministic ReLU Moore transducers contained 90 concrete resettable states and 54
states in their exact minimal quotients. Response-only active recovery returned
the oracle quotient in all 12 cases. All 12 oracle certificates and all 12
response-derived certificates passed independent semantic verification and a
complete deterministic replay.

The result proves minimality only for each stored finite system over its
declared complete action-closed state list, exact output equality, fixed action
labels, and all finite action words. It is not evidence for continuous neural
systems or an approximate realization theorem.

## Formal run

The formal seed is 2701. The 12 cases used 3--6 minimal abstract states and
2--4 additional physical states, with two labelled actions. State splitting
preserved future behavior while allowing different concrete successors and
unrelated activation embeddings.

| Quantity | Result |
|---|---:|
| Concrete states | 90 |
| Certified minimal states | 54 |
| Quotient/concrete ratio | 0.600 |
| Response queries | 678 |
| Active exact and certified | 12 / 12 |
| Upper certificates checked | 12 / 12 |
| Lower certificates checked | 12 / 12 |

Every upper certificate was checked by exact output preservation and every
concrete state/action homomorphism equation. Every lower certificate supplied
a re-executed response witness for every pair of abstract representatives.

## Equal-query-budget comparisons

Random and passive methods were evaluated by the hidden oracle at the exact
same response-query budget used by active recovery. They sometimes happened to
produce the correct partition but none produced a completeness certificate at
that budget.

| Method | Exact cases | Certified cases | False merges | False splits |
|---|---:|---:|---:|---:|
| Active closed/consistent table | 12 | 12 | 0 | 0 |
| Random experiment order | 7 | 0 | 35 | 0 |
| Passive short traces | 11 | 0 | 1 | 0 |
| Activation clustering | 0 | 0 | 56 | 34 |

The activation baseline was given the true number of quotient states and the
network's actual one-hot quantized hidden states, so its
failure is not caused by model-order selection. It shows that physical
activation geometry need not recover computational equivalence in this
controlled benchmark.

## Budget-limited behavior

The limited runs received 45% of the active method's case-specific query
budget. None claimed completion. Together they retained 76 state pairs as
'unresolved'; no unresolved pair was reported as certified equivalent.

## Certified finite complexity--fidelity curve

Each case also used a numeric-output ReLU transducer and L-infinity response
metric. The incompatibility-graph chromatic lower bound and the complete
candidate-system upper bound closed in every case at every declared tolerance.

| Epsilon | Total lower bound | Total upper bound | Closed cases |
|---:|---:|---:|---:|
| 0.00 | 90 | 90 | 12 / 12 |
| 0.25 | 54 | 54 | 12 / 12 |
| 10.00 | 12 | 12 | 12 / 12 |

The upper verifier checked the complete candidate encoding, all numeric output
errors, and all labelled transitions. It did not infer mergeability from
pairwise distances.

## Reproduction

    python -m ncd verify-certified-finite runs/certified_finite_seed2701

The verifier checked every artifact hash, regenerated all systems, reran oracle
minimization and every query strategy, checked 60 semantic certificates (24
exact and 36 approximate), and
recomputed the summary. The full repository regression suite passed 124 tests.

The 0.27.0 wheel was installed into an isolated environment. Its modules match
the source byte-for-byte, and fresh finite, traffic, and continuous certificate
runs plus semantic replay completed from the installed package. Wheel SHA-256:
'33fd202c630b686181d4a5386e4bfe964fa400229797d6cf737f67b663a3b548'.
Release evidence is in 'validation/wheel_v27_run/status.json'.

Formal artifacts are in 'runs/certified_finite_seed2701'. Definitions, proofs,
access assumptions, and limitations are in
'docs/CERTIFIED_INTERVENTIONAL_REALIZATION.md'.
