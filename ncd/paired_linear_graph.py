"""Exact graph recovery from paired two-level interventions in linear DAG SCMs.

The estimator sees only intervention response vectors and levels. The theorem
requires both levels for a source to share the same exogenous realization;
without that coupling, finite-sample exactness is not asserted.
"""
from fractions import Fraction as Q


def _fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Q)):
        raise ValueError("Intervention responses must be exact rational values")
    return Q(value)


def inverse(matrix):
    """Gauss-Jordan inverse over exact rationals."""
    n = len(matrix)
    if n < 1 or any(len(row) != n for row in matrix):
        raise ValueError("Expected a nonempty square matrix")
    left = [[_fraction(value) for value in row] for row in matrix]
    right = [[Q(int(i == j)) for j in range(n)] for i in range(n)]
    for column in range(n):
        pivot = next((row for row in range(column, n)
                      if left[row][column] != 0), None)
        if pivot is None:
            raise ValueError("Total-effect matrix is singular")
        left[column], left[pivot] = left[pivot], left[column]
        right[column], right[pivot] = right[pivot], right[column]
        scale = left[column][column]
        left[column] = [value / scale for value in left[column]]
        right[column] = [value / scale for value in right[column]]
        for row in range(n):
            if row == column:
                continue
            scale = left[row][column]
            left[row] = [a - scale * b for a, b in zip(left[row], left[column])]
            right[row] = [a - scale * b for a, b in zip(right[row], right[column])]
    return right


def recover_graph(plus, minus, levels):
    """Return total effects, direct coefficients and graph from paired do data.

    plus[i] and minus[i] are full observed response vectors under the two
    levels of do(X_i). Exogenous values must be identical within each pair.
    This function never reads an SCM graph, equation or noise model.
    """
    n = len(levels)
    if n < 1 or len(plus) != n or len(minus) != n:
        raise ValueError("One intervention pair is required per source")
    total = [[Q(int(target == source)) for source in range(n)]
             for target in range(n)]
    for source, pair in enumerate(levels):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each source requires two intervention levels")
        high, low = map(_fraction, pair)
        if high == low or len(plus[source]) != n or len(minus[source]) != n:
            raise ValueError("Invalid intervention pair")
        for target in range(n):
            difference = (_fraction(plus[source][target])
                          - _fraction(minus[source][target])) / (high - low)
            if target == source and difference != 1:
                raise ValueError("Intervened coordinate does not track do level")
            total[target][source] = difference
    inv = inverse(total)
    direct = [[Q(int(target == source)) - inv[target][source]
               for source in range(n)] for target in range(n)]
    if any(direct[i][i] != 0 for i in range(n)):
        raise ValueError("Nonzero self mechanism")
    graph = [[int(direct[target][source] != 0)
              for target in range(n)] for source in range(n)]
    remaining = set(range(n))
    while remaining:
        roots = {j for j in remaining if all(not graph[i][j] for i in remaining)}
        if not roots:
            raise ValueError("Recovered graph is cyclic; paired linear-DAG assumptions may fail")
        remaining -= roots
    return {
        "total_effect": [[str(value) for value in row] for row in total],
        "inverse_total_effect": [[str(value) for value in row] for row in inv],
        "direct_effect": [[str(value) for value in row] for row in direct],
        "graph": graph,
        "paired_exogenous_required": True,
        "truth_graph_read_by_estimator": False,
    }
