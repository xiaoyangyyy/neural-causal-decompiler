# Exact shared-action pairwise support for finite neural realization

The v0.52 graph includes 2,060 recurrent cells because it encloses each successor coordinate separately for every continuous action and takes the Cartesian product. Its 51,304 graph edges are conservative. When several successor coordinates share the same five action variables, some combinations of independently possible coordinate values are impossible together. This method proves a necessary condition for each pair using only exact rational arithmetic.

The verifier first replays the v0.52 graph certificate and all its predecessors, including the frozen trained affine 128D ReLU network's globally fixed activation phases. It extracts the exact affine transition `f(x,u)=Ax+Bu+b`. For each abstract source center `c`, write its active successor coordinate as `y_i=s_i(c)+B_i u`, with `u` in the full continuous unit action cube. For an active coordinate pair `(i,k)` and positive rational slope `lambda`, the *shared* action imposes the exact support interval

    y_i - lambda*y_k in [s_i-lambda*s_k+sum_j min(0,B_ij-lambda*B_kj),
                         s_i-lambda*s_k+sum_j max(0,B_ij-lambda*B_kj)].

For a proposed pair of quantized recurrent indices `(q_i,q_k)`, the corresponding closed target cells imply

    y_i - lambda*y_k in [q_i/n_i-lambda*(q_k+1)/n_k,
                         (q_i+1)/n_i-lambda*q_k/n_k].

If these two rational intervals are strictly disjoint, that target is impossible for **every** action in the unit cube. Closed target cells intentionally retain boundary ties from floor-and-clamp quantization. No floating-point LP decision enters the proof. All actual machine action centers lie within the unit cube, so excluding a cell this way is sound for the executable machine.

For each of the ten active-coordinate pairs, the checker uses the ratio of total action influences and all five positive ratios of corresponding action coefficients. On this network these give **60 exact pairwise support planes**. The slopes are a deterministic choice that tends to align with the action-image zonotope; completeness is unnecessary. The checker starts from all 243 initial-grid centers, retains every candidate not excluded by any pair, and then computes the least closed recurrent successor graph with the same rule. It enumerates 6,768 initial candidate edges and retains 859 distinct initial successor cells. During graph closure it checks 20,672 recurrent candidate edges, retains 6,368 edges, and reaches 899 recurrent cells. Every retained cell belongs to the independently certified v0.52 graph.

The executable `correlated_initial`, `correlated_output`, and `correlated_step` restrict the prior two-stage transition to this closed set. Since pairwise exclusion only removes impossible transitions, every real machine transition remains in the set. The prior exact initial-output, handoff, and recurrent simulation inequalities still prove output error at most `17/100` for every unit initial state, every continuous unit action word, and every finite horizon. The new deterministic finite realization has `243+899=1,142` states. Combining it with the independent 162-state packing lower certificate yields the same-model general minimum interval **[162,1142]**. The lower uses binary64 `0.17`, slightly looser than exact rational `17/100` used by the upper.

The pairwise conditions are necessary, not sufficient for exact five-dimensional action feasibility. Retained edges and cells need not be concretely reachable. This does not prove global minimality or certify the other trained nonlinear 128D networks.

Replay:

```powershell
python -m scripts.acceptance_correlated_reachability --verify
```

Evidence: `runs/correlated_reachability_v1/affine_d128/certificate.json`, `validation/correlated_reachability_acceptance.json`, and the prior certificates bound by SHA-256.
