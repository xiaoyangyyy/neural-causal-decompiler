"""Checks of three-node learned-SCM composition without truth-oracle access."""
from fractions import Fraction as Q
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'validation/frozen_three_node_package_v1'))
from frozen_three_node.proof import child_lipschitz, historical_bindings, source_from_original
from frozen_three_node.__main__ import execute


def test_coordinatewise_lipschitz_exact_weight_product():
    network = {
        'parents': [1, 2], 'std': ['2', '4'], 'output_scale': '2',
        'layers': [
            {'weights': [['1', '0'], ['0', '1']], 'activation': 'tanh'},
            {'weights': [['1', '0'], ['0', '1']], 'activation': 'tanh'},
            {'weights': [['2', '3']], 'activation': 'identity'},
        ],
    }
    assert child_lipschitz(network) == {'1': Q(2), '2': Q(3, 2)}
    network['layers'][1]['activation'] = 'sin'
    with pytest.raises(ValueError, match='activation order'):
        child_lipschitz(network)


def test_all_joint_do_masks_execute_in_learned_graph_order():
    program = {
        'schema': 'ncd.frozen-three-node-program.v1',
        'graph': [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
        'domain': [['-1', '1']] * 3,
        'root_expressions': {'1': {'op': 'constant', 'value': 0.25},
                             '2': {'op': 'constant', 'value': -0.5}},
        'child_piecewise_program': {
            'schema': 'ncd.piecewise-cdir-mechanism-program.v1',
            'domain': [['-1', '1']] * 3,
            'tree': {'expression': {'op': 'add', 'args': [
                {'op': 'var', 'index': 1}, {'op': 'var', 'index': 2}]}},
        },
        'noise_model': None,
    }
    for mask in range(8):
        interventions = {str(j): 0.1 * (j + 1) for j in range(3) if mask & (1 << j)}
        result = execute(program, interventions)
        values = result['values']
        for j, natural in ((1, 0.25), (2, -0.5)):
            assert values[j] == pytest.approx(interventions.get(str(j), natural))
        expected = interventions.get('0', values[1] + values[2])
        assert values[0] == pytest.approx(expected)
        assert result['intervened_nodes'] == [j for j in range(3) if mask & (1 << j)]
    with pytest.raises(ValueError, match='outside declared box'):
        execute(program, {'1': 2.0})
    with pytest.raises(ValueError, match='outside declared box'):
        execute(program, {'3': 0.0})


def test_accepted_child_or_root_weight_tamper_rejected():
    source = source_from_original(ROOT)
    historical_bindings(source)
    changed = dict(source)
    changed['mechanism_1.pt'] = source['mechanism_2.pt']
    with pytest.raises(ValueError, match='Changed root checkpoint'):
        historical_bindings(changed)
    changed = dict(source)
    changed['child_program.json'] = b'{}'
    with pytest.raises(ValueError, match='Changed accepted child artifact'):
        historical_bindings(changed)