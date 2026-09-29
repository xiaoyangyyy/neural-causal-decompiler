"""Exact optimality proof inside the uniform coordinate-grid certificate class.

This is a lower bound on the state count of every weighted-grid certificate
for the same frozen model and epsilon, not on arbitrary finite realizations.
"""
from __future__ import annotations

from fractions import Fraction as Q
from math import prod

from .continuous_compositional_realization import verify_weighted
from .continuous_separation import ContinuousReLUSystem, ReLUMLP, _digest


def _absolute_rows(network: ReLUMLP) -> tuple[dict[int, Q], ...]:
    """Sparse exact product of absolute layer-weight matrices."""
    rows = tuple({j: Q(1)} for j in range(network.input_dim))
    for weights in network.weights:
        next_rows = []
        for layer_row in weights:
            result: dict[int, Q] = {}
            for j, weight in enumerate(layer_row):
                coefficient = abs(Q(weight))
                if not coefficient:
                    continue
                for k, inherited in rows[j].items():
                    result[k] = result.get(k, Q(0)) + coefficient * inherited
            next_rows.append(result)
        rows = tuple(next_rows)
    return rows


def _right_multiply(rows: tuple[dict[int, Q], ...],
                    matrix: tuple[dict[int, Q], ...]) -> tuple[dict[int, Q], ...]:
    answer = []
    for row in rows:
        next_row: dict[int, Q] = {}
        for i, left in row.items():
            for j, right in matrix[i].items():
                next_row[j] = next_row.get(j, Q(0)) + left * right
        answer.append(next_row)
    return tuple(answer)


def _neumann_rows(system: ContinuousReLUSystem, horizon: int):
    d = system.state_dim
    transition = _absolute_rows(system.transition)
    state = tuple({j: value for j, value in row.items() if j < d}
                  for row in transition)
    observed = _absolute_rows(system.observation)
    terms = tuple(dict(row) for row in observed)
    totals: tuple[dict[int, Q], ...] = tuple({} for _ in observed)
    for _ in range(horizon + 1):
        for total, row in zip(totals, terms):
            for j, value in row.items():
                total[j] = total.get(j, Q(0)) + value
        terms = _right_multiply(terms, state)
    return totals


def _proof(system: ContinuousReLUSystem, upper_certificate: dict,
           horizon: int) -> dict:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 0 <= horizon <= 32:
        raise ValueError('Invalid finite Neumann horizon')
    d = system.state_dim
    if d < 5 or system.observation.output_dim < 4:
        raise ValueError('Four observed and five distinct state axes required')
    upper = verify_weighted(system, upper_certificate)
    if upper['status'] != 'certified' or upper['upper_bound'] is None:
        raise ValueError('A certified weighted-grid upper is required')
    epsilon = Q(upper_certificate['epsilon'])
    target = int(upper['upper_bound'])
    middle_min = 6
    middle_count = 3
    middle_factor = middle_min ** middle_count
    if target % middle_factor:
        raise ValueError('Upper count does not factor into the proposed proof')
    pair_target = target // middle_factor
    if pair_target < 2:
        raise ValueError('Upper count too small for pair proof')
    rows = _neumann_rows(system, horizon)
    for axis in (1, 2, 3):
        if rows[axis].get(axis, Q(0)) <= 2 * (middle_min - 1) * epsilon:
            raise ValueError('Finite proof cannot force six middle-axis bins')
    left = rows[0].get(0, Q(0))
    right = rows[0].get(d - 1, Q(0))
    if left <= 0 or right <= 0:
        raise ValueError('Finite proof has no two-axis feedback sensitivity')
    active = {0, 1, 2, 3, d - 1}
    background = sum((value for j, value in rows[0].items()
                      if j not in active), Q(0)) / 2
    # AM-GM: epsilon >= left/(2*n0)+right/(2*nlast)
    # implies n0*nlast >= left*right/epsilon**2.
    unrestricted_pair_min = (left * right / epsilon**2).__ceil__()
    if 2 * unrestricted_pair_min < pair_target:
        raise ValueError('Extra-axis branch is not excluded')
    # If all other axes have one bin, this is the smallest possible row-0
    # lower bound for each n0 given n0*nlast < pair_target.
    cases = []
    for n0 in range(1, pair_target):
        nlast = (pair_target - 1) // n0
        if nlast < 1:
            continue
        lower = left / (2 * n0) + right / (2 * nlast) + background
        cases.append((lower, n0, nlast))
    minimum, minimizing_n0, minimizing_last = min(cases)
    if minimum <= epsilon:
        raise ValueError('No-extra-axis branch is not excluded')
    bins = upper_certificate['coordinate_bins']
    if prod(bins) != target:
        raise ValueError('Upper grid state count mismatch')
    return {
        'schema': 'ncd.weighted-grid-optimality.v1',
        'system_sha256': _digest(system.to_dict()),
        'upper_certificate_sha256': _digest(upper_certificate),
        'epsilon': str(epsilon),
        'horizon': horizon,
        'class': 'all positive integer uniform coordinate bins, arbitrary positive radii and action-bin count satisfying the exact weighted verifier',
        'certified_class_minimum_states': target,
        'middle_axes': [1, 2, 3],
        'middle_axis_minimum_bins': middle_min,
        'pair_axes': [0, d - 1],
        'unrestricted_pair_minimum_product': unrestricted_pair_min,
        'extra_axis_minimum_factor': 2,
        'no_extra_axis_minimum_pair_product': pair_target,
        'no_extra_axis_exclusion_cases': len(cases),
        'least_exclusion_case': [minimizing_n0, minimizing_last],
        'least_exclusion_margin': str(minimum - epsilon),
        'neumann_row0_left': str(left),
        'neumann_row0_right': str(right),
        'neumann_row0_unbinned_background': str(background),
        'status': 'certified',
        'scope': 'optimal within the exact weighted uniform coordinate-grid certificate class only',
    }


def certify_weighted_grid_optimality(system: ContinuousReLUSystem,
                                    upper_certificate: dict,
                                    *, horizon: int = 10) -> dict:
    certificate = _proof(system, upper_certificate, horizon)
    verify_weighted_grid_optimality(system, upper_certificate, certificate)
    return certificate


def verify_weighted_grid_optimality(system: ContinuousReLUSystem,
                                   upper_certificate: dict,
                                   certificate: dict) -> dict:
    if certificate.get('schema') != 'ncd.weighted-grid-optimality.v1':
        raise ValueError('Unsupported weighted-grid optimality certificate')
    expected = _proof(system, upper_certificate, certificate['horizon'])
    if certificate != expected:
        raise ValueError('Weighted-grid optimality replay mismatch')
    return {'status': 'verified',
            'class_minimum_states': expected['certified_class_minimum_states'],
            'horizon': expected['horizon'],
            'exclusion_cases': expected['no_extra_axis_exclusion_cases']}
