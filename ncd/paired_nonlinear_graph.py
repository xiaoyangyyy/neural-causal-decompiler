"""Conditional local graph recovery from paired nonlinear node interventions.

The caller supplies exact rational responses at a shared exogenous state.
Curvature, response-error, and direct-effect margins are assumptions, not
inferred from the responses. This module never reads a true graph or equation.
"""
from fractions import Fraction as Q

from .paired_linear_graph import inverse


def rational(value):
    if type(value) not in (int, str, Q):
        raise ValueError("Exact rational data required")
    return Q(value)


def matrix_norm_inf(matrix):
    return max(sum(map(abs, row), Q(0)) for row in matrix)


def recover_local_graph(baseline, responses, step, *, curvature_bound,
                        response_error_bound, minimum_visible_direct_effect):
    """Recover local support if the stated pairing and separation hold.

    The exact interventional response Y^(i)(t) is twice differentiable for
    0 <= t <= step, with Y^(i)(0) the common baseline and each second
    derivative bounded by curvature_bound. Every supplied coordinate has
    absolute measurement error at most response_error_bound. At this base
    point every direct derivative is either zero or has magnitude at least
    minimum_visible_direct_effect. These premises require separate evidence.
    """
    n = len(baseline)
    if n < 2 or len(responses) != n or any(len(row) != n for row in responses):
        raise ValueError("One full response per source is required")
    base = [rational(v) for v in baseline]
    changed = [[rational(v) for v in row] for row in responses]
    h = rational(step)
    m = rational(curvature_bound)
    delta = rational(response_error_bound)
    gamma = rational(minimum_visible_direct_effect)
    if h <= 0 or m < 0 or delta < 0 or gamma <= 0:
        raise ValueError("Invalid positive step or assumption bounds")
    for source in range(n):
        if abs(changed[source][source] - base[source] - h) > 2 * delta:
            raise ValueError("Intervened coordinate does not track the do value")
    total = [[(changed[source][target] - base[target]) / h
              for source in range(n)] for target in range(n)]
    inv = inverse(total)
    k = matrix_norm_inf(inv)
    entry_error = m * h / 2 + 2 * delta / h
    total_error = n * entry_error
    if k * total_error >= 1:
        raise ValueError("Inverse perturbation bound does not close")
    direct_error = k * k * total_error / (1 - k * total_error)
    if 2 * direct_error >= gamma:
        raise ValueError("Direct-effect margin is not separated")
    direct = [[Q(int(target == source)) - inv[target][source]
               for source in range(n)] for target in range(n)]
    graph = [[0 if source == target else
              int(abs(direct[target][source]) > gamma / 2)
              for target in range(n)] for source in range(n)]
    remaining = set(range(n))
    while remaining:
        roots = {j for j in remaining if all(graph[i][j] == 0 for i in remaining)}
        if not roots:
            raise ValueError("Recovered local graph is cyclic")
        remaining -= roots
    return {
        "schema": "ncd.paired-nonlinear-local-graph.v1",
        "status": "conditional-on-pairing-curvature-and-direct-effect-margin",
        "total_effect_estimate": [[str(v) for v in row] for row in total],
        "inverse_total_effect_estimate": [[str(v) for v in row] for row in inv],
        "direct_effect_estimate": [[str(v) for v in row] for row in direct],
        "total_matrix_error_bound_inf": str(total_error),
        "direct_matrix_error_bound_inf": str(direct_error),
        "graph_source_target": graph,
        "truth_graph_read_by_estimator": False,
        "shared_exogenous_state_assumed": True,
        "original_claim_closed": False,
    }