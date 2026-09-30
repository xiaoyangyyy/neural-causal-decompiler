"""Truth-free total-effect graph candidate from independent node-do groups.

The procedure estimates differences of *group means*, with no exogenous
pairing and no claim that a finite data set certifies the true graph.
"""
import numpy as np


def recover_independent_do_means(groups, nodes, *, threshold=0.2):
    if type(nodes) is not int or nodes not in (3, 5, 8):
        raise ValueError("Supported graph sizes are 3/5/8")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive finite threshold required")
    if set(groups) != {(source, level) for source in range(nodes)
                      for level in (-1, 1)}:
        raise ValueError("One independent group per source and do level required")
    total = np.zeros((nodes, nodes), dtype=float)
    coverage = []
    for source in range(nodes):
        means, counts = {}, {}
        for level in (-1, 1):
            values = np.asarray(groups[(source, level)], dtype=float)
            if (values.ndim != 2 or values.shape[1] != nodes
                    or len(values) < 2 or not np.isfinite(values).all()):
                raise ValueError("Invalid independent do group")
            if not np.allclose(values[:, source], level, rtol=0, atol=1e-12):
                raise ValueError("Group does not match its declared do coordinate")
            means[level] = values.mean(axis=0)
            counts[level] = len(values)
        total[:, source] = (means[1]-means[-1])/2.
        coverage.append({"source": source, "plus_rows": counts[1],
                         "minus_rows": counts[-1]})
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite total-effect estimate")
    try:
        inverse = np.linalg.inv(total)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Empirical total-effect matrix is singular") from exc
    direct = np.eye(nodes)-inverse
    graph = (np.abs(direct.T) > threshold).astype(int)
    np.fill_diagonal(graph, 0)
    remaining = set(range(nodes))
    while remaining:
        roots = {j for j in remaining if all(graph[i, j] == 0 for i in remaining)}
        if not roots:
            break
        remaining -= roots
    return {
        "schema": "ncd.independent-do-mean-graph-candidate.v1",
        "total_effect": total.tolist(),
        "direct_effect": direct.tolist(),
        "graph_source_target": graph.tolist(),
        "acyclic": not remaining,
        "coverage": coverage,
        "threshold": float(threshold),
        "exogenous_pairing_used": False,
        "truth_graph_read": False,
        "finite_sample_graph_guarantee": False,
    }