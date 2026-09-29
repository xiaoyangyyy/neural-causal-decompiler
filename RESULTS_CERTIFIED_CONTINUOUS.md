# Results: certified continuous neural separation

## Outcome

The project now has a sound finite-horizon separation oracle for deterministic
continuous ReLU systems. Its optimization domain is the complete box of
continuous intervention words. Feasible words provide lower bounds; outward-
rounded interval propagation over a branch-and-bound cover provides upper
bounds. An independent verifier reconstructs every split and recomputes every
bound.

The formal benchmark used stable one-state recurrent neural dynamics with a
continuous demand control in `[0,1]` and horizon one. It intentionally contains
all three outcomes:

| Result | Cases |
|---|---:|
| Certified separated (`LB > 2 epsilon`) | 1 |
| Certified within tolerance (`UB <= 2 epsilon`) | 2 |
| Unresolved under the declared budget | 1 |

For the nontrivial close pair, four leaves leave the query unresolved. Sixteen
leaves reduce the certified interval enough to prove it is within tolerance.
The result therefore checks the required evidence policy directly: a failed
search is not reported as equivalence.

Four declared continuous neural states generated six pair certificates. Five
pairs were certified incompatible. Independent exact coloring of that verified
graph gives

\[
K_{\mathcal N}(0.03,\mathcal W)\ge 3
\]

for this finite state collection and the declared continuous intervention
domain.

At epsilon `0.3`, a separately verified two-state candidate uses a partition
tree over the complete continuous demand interval. The upper verifier checks
all four initial states and every action in each tree leaf through the original
ReLU networks. Its maximum certified error is `0.295000000000001`. The
incompatibility graph independently requires two states, so this run closes:

\[
2\le K_{\mathcal N}(0.3,\mathcal W)\le2.
\]

## Boundary

This is a certified neural separation primitive and, for the frozen finite
initial-state collection at horizon one, a complete minimal realization result.
It is not a realization of an uncountable initial-state space. No uniform upper
fidelity certificate over all continuous initial states is claimed. Interval
bounds may remain unresolved when they are too loose or the branch budget is
exhausted.

## Reproduction

    python -m ncd verify-certified-continuous runs/certified_continuous

Machine-readable acceptance is in
`validation/certified_continuous_acceptance.json`.

The isolated 0.27.0 wheel executed this workflow and its verifier successfully.
The full regression suite passed 124 tests. Wheel SHA-256:
`33fd202c630b686181d4a5386e4bfe964fa400229797d6cf737f67b663a3b548`.
