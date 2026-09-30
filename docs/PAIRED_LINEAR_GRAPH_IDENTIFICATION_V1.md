# Exact graph recovery from paired two-level interventions

This is a scoped R9 identification result for **linear additive DAG SCMs**.
The candidate estimator receives only two full response vectors per
intervened source node and the two do values. The archive's true graph
and equations are read only by the separate simulator/evaluator.

Let the observed-coordinate SCM be X = C X + U, with C_{ji} the
direct effect i→j and C acyclic. For every source i, evaluate
do(X_i=a_i) and do(X_i=b_i), with a_i≠b_i, using **the same exogenous
vector U within that pair**. Define

    A_{ji} = (X_j[do(X_i=a_i)] - X_j[do(X_i=b_i)])/(a_i-b_i).

Acyclicity means an intervention on i cannot change any ancestor of
i. Subtracting the two structural systems cancels U and gives
(I-C) A_{·i}=e_i. Hence A=(I-C)^{-1}, and

    C = I - A^{-1}.

The graph consists exactly of the nonzero off-diagonal entries of C.
No noise distribution, marginal independence, faithfulness margin or
sample size is needed under this exact paired-response model. This
statement is about the mathematical paired oracle: it does not
certify independently sampled interventions, finite-sample
concentration, experimental reuse of the same latent disturbances,
or device floating-point rounding.

`ncd/paired_linear_graph.py` implements rational inversion and accepts
only response vectors and levels. In the archived `test_id`
linear-Gaussian three-node world, a simulator constructed exact
rational responses for do values ±1 with one fixed rational
exogenous vector. The estimator returned the true graph

    0→1, 0→2, 2→1

and the exact binary64-rational direct coefficients recorded in the
world metadata. The oracle graph and equations enter only the
evaluation code in
`validation/paired_linear_graph_development_v1.py`. Its certificate
is `validation/paired_linear_graph_development_v1.json`.
A separate checker,
`validation/check_paired_linear_graph_development_v1.py`, verifies
the original archive and source hashes, checks every response against
the true equations with the shared exogenous vector, and independently
checks A(I-C)=I and the recovered graph. Its sealed receipt is
`validation/paired_linear_graph_development_verification_v1.json`.
Eleven targeted tests include exact 3/5/8-node DAGs, node relabeling, singular and
malformed/cyclic grids, certificate tampering, and rehashed archive mutation.

The experiment is one historical development world and has access to
a simulator pairing ability. It does not use or validate the frozen
graph-discovery network, does not recover nonlinear graphs, and does
not satisfy the original end-to-end R9 claim. The result isolates a
specific information and algorithm boundary: in this linear class,
exact paired interventions determine the graph, while the current
learned active pipeline has not demonstrated the corresponding
general recovery guarantee.
