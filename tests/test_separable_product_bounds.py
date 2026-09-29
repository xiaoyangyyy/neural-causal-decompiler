"""Exact product amplification and fail-closed network controls."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from fractions import Fraction as Q
import pytest
from ncd.continuous_cover import benchmark_cover_system
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import value
from ncd.separable_product_bounds import (
    Unresolved, _check_semantic_product, certify_product_bounds, verify_product_bounds,
)

PROOF = Path(__file__).resolve().parents[1] / 'validation' / 'exact_interval_lower_8.json.gz'


def _proof():
    with gzip.open(PROOF,'rt',encoding='utf-8') as handle:
        return json.load(handle)


def test_product_lift_replays_scalar_and_two_dimensional_bounds():
    proof = _proof()
    system = benchmark_cover_system(2)
    certificate = certify_product_bounds(system,proof)
    assert verify_product_bounds(system,proof,certificate) == {
        'status':'verified', 'dimension':2, 'coefficients_checked':42,
        'lower_bound':'36', 'upper_bound':'81', 'horizon':'unbounded'}
    changed = deepcopy(certificate)
    changed['lower_bound'] = '81'
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_product_bounds(system,proof,changed)


def test_coupling_and_output_mixing_fail_closed_before_scalar_transfer():
    proof = _proof()
    base = benchmark_cover_system(2).to_dict()
    coupled = deepcopy(base)
    coupled['transition']['weights'][1][0][1] = 0.015625
    coupled['transition']['biases'][1][0] -= 0.015625
    with pytest.raises(Unresolved,match='coupled'):
        certify_product_bounds(ContinuousReLUSystem.from_dict(coupled),proof)
    mixed = deepcopy(base)
    mixed['observation']['weights'][1][0][1] = 0.015625
    mixed['observation']['biases'][1][0] -= 0.015625
    with pytest.raises(Unresolved,match='identity'):
        certify_product_bounds(ContinuousReLUSystem.from_dict(mixed),proof)


def test_hidden_permutation_and_scaling_preserve_exact_semantic_certificate():
    original = benchmark_cover_system(8)
    transformed = deepcopy(original.to_dict())
    for key in ('transition','observation'):
        network = transformed[key]
        permutation = list(reversed(range(len(network['weights'][0]))))
        network['weights'][0] = [network['weights'][0][j] for j in permutation]
        network['biases'][0] = [network['biases'][0][j] for j in permutation]
        network['weights'][1] = [[row[j] for j in permutation]
                                 for row in network['weights'][1]]
        network['weights'][0] = [[2*x for x in row] for row in network['weights'][0]]
        network['biases'][0] = [2*x for x in network['biases'][0]]
        network['weights'][1] = [[x/2 for x in row] for row in network['weights'][1]]
    permuted = ContinuousReLUSystem.from_dict(transformed)
    assert permuted.to_dict() != original.to_dict()
    assert _check_semantic_product(permuted) == _check_semantic_product(original)
    state = [Q(i+1,10) for i in range(8)]
    action = [Q(8-i,10) for i in range(8)]
    actual = value(permuted.transition,state+action)
    expected = [Q(1,2)*x+Q(0.4)*a for x,a in zip(state,action)]
    assert actual == expected == value(original.transition,state+action)
    assert value(permuted.observation,state) == state


def test_phase_changing_but_functionally_unused_unit_is_unresolved():
    proof = _proof()
    data = deepcopy(benchmark_cover_system(2).to_dict())
    transition = data['transition']
    transition['weights'][0].append([1.0,0.0,0.0,0.0])
    transition['biases'][0].append(-0.5)
    for row in transition['weights'][1]:
        row.append(0.0)
    equivalent = ContinuousReLUSystem.from_dict(data)
    with pytest.raises(Unresolved,match='phase changes'):
        certify_product_bounds(equivalent,proof)
