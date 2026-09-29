"""Exact one-step behavioral packing across affine and nonlinear ReLUs."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest
from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.dynamic_packing import (
    Unresolved, certify_dynamic_packing, certify_staggered_ring_packing,
    verify_dynamic_packing,
)
from ncd.io import read_json

ROOT = Path(__file__).resolve().parents[1]
AFFINE = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_000'/'system.json'
ABLATED = ROOT/'runs'/'affine_quotient_global_v1'/'controls'/'ablated_affine_d128'/'system.json'


def test_staggered_135_point_packing_and_tamper_rejection():
    system = ContinuousReLUSystem.from_dict(read_json(AFFINE))
    certificate = certify_staggered_ring_packing(system)
    result = verify_dynamic_packing(system,certificate)
    assert result['lower_bound'] == 135
    assert result['pair_count'] == 9045
    assert result['first_witness_counts'] == [8991,54]
    assert result['evaluation_mode'] == 'globally-affine'
    assert Q(result['minimum_pair_distance']) > Q(result['threshold_exact_float'])
    changed = deepcopy(certificate)
    changed['metrics']['lower_bound'] = 136
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_dynamic_packing(system,changed)
    changed = deepcopy(certificate)
    changed['claim'] = 'exact minimum is 135'
    with pytest.raises(ValueError,match='mismatch'):
        verify_dynamic_packing(system,changed)


def test_nonlinear_fixed_point_evaluator_certifies_one_step_separation():
    transition = ReLUMLP(
        weights=(((1.0,0.0),(1.0,0.0)),((4.0,-4.0),)),
        biases=((-0.5,-0.75),(0.0,)))
    observation = ReLUMLP(weights=(((1.0,),),),biases=((0.0,),))
    system = ContinuousReLUSystem(1,1,transition,observation)
    certificate = certify_dynamic_packing(
        system,((Q(7,20),),(Q(13,20),)),((Q(0),),))
    result = verify_dynamic_packing(system,certificate)
    assert result['evaluation_mode'] == 'direct-exact-ReLU'
    assert result['first_witness_counts'] == [0,1]
    assert result['minimum_pair_distance'] == '3/5'
    tighter = certify_dynamic_packing(
        system,((Q(7,20),),(Q(13,20),)),((Q(0),),),epsilon=0.1)
    assert Q(verify_dynamic_packing(system,tighter)['threshold_exact_float']) == 2*Q(0.1)


def test_removed_feedback_edge_breaks_this_specific_dynamic_witness():
    system = ContinuousReLUSystem.from_dict(read_json(ABLATED))
    left = [Q(0)]*system.state_dim
    right = [Q(0)]*system.state_dim
    right[0],right[-1] = Q(1,10),Q(1)
    with pytest.raises(Unresolved,match='not separated'):
        certify_dynamic_packing(system,(left,right),((Q(0),)*system.action_dim,))
