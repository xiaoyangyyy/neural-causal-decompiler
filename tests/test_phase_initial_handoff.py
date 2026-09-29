"""Phase-aware initialization minimum, actual handoff and unsupported cases."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json
from ncd.nonlinear_action_closure import program_initial,program_step,program_output,verify_nonlinear_closure
from ncd.phase_initial_handoff import verify_phase_handoff,_decompose,_observation
from ncd.reachable_two_stage import _center

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/'d_128'
TARGET=ROOT/'runs'/'phase_initial_handoff_v1'/'seed_6101'/'d_128'


def test_phase_handoff_minimum_and_actual_first_transition():
    system=ContinuousReLUSystem.from_dict(read_json(SOURCE/'system.json'))
    recurrent=read_json(SOURCE/'certificate.json')
    initial=read_json(TARGET/'initial_certificate.json')
    closure=read_json(TARGET/'closure_certificate.json')
    program=read_json(TARGET/'program.json')
    assert verify_phase_handoff(system,recurrent,initial)=={
        'status':'verified','initial_state_count':81,'recurrent_state_count':216,
        'upper_bound':297,'minimum_initial_label_count':81}
    assert verify_nonlinear_closure(system,recurrent,initial,closure)=={
        'status':'verified','initial_state_count':81,'recurrent_state_count':55,'upper_bound':136}
    assert initial['state_pair_cells_checked']==156
    assert Q(initial['minimum_handoff_slack'])>0
    legacy=read_json(ROOT/'runs'/'optimal_initial_grid_v1'/'seed_6101'/'d_128'/'certificate.json')
    assert legacy['initial_state_count']==108 and legacy['baseline_handoff_excluded_coordinates']==[0]
    assert all(Q(e)<=Q(r) for e,r in zip(initial['initial_to_recurrent_error_upper'],recurrent['coordinate_radii']))
    assert Q(initial['initial_packing']['minimum_output_separation'])>Q(initial['initial_packing']['twice_epsilon'])
    for point,action in (([Q(0)]*128,(Q(0),Q(1))),([Q(1)]*128,(Q(1),Q(0))),
                         ([Q(i%3,2) for i in range(128)],(Q(1,2),Q(1,2)))):
        state=program_initial(program,point)
        assert max(abs(a-b) for a,b in zip(program_output(program,state),value(system.observation,point)))<=Q(17,100)
        target=program_step(program,state,action)
        actual=value(system.transition,list(point)+list(action))
        center=_center(target[1],recurrent['coordinate_bins'])
        assert all(abs(a-b)<=Q(r) for a,b,r in zip(actual,center,recurrent['coordinate_radii']))


def test_phase_handoff_rejects_mixed_hidden_support():
    payload=read_json(SOURCE/'system.json')
    system=ContinuousReLUSystem.from_dict(payload)
    j=next(j for j,w in enumerate(system.transition.weights[1][0])
        if w and sum(bool(x) for x in system.transition.weights[0][j][:-2])==2)
    payload=deepcopy(payload)
    payload['transition']['weights'][0][j][-2]='1'
    changed=ContinuousReLUSystem.from_dict(payload)
    with pytest.raises(ValueError,match='Mixed state/action'):
        _decompose(changed,0)


def test_phase_handoff_tamper_and_observation_guard():
    system=ContinuousReLUSystem.from_dict(read_json(SOURCE/'system.json'))
    recurrent=read_json(SOURCE/'certificate.json')
    forged=deepcopy(read_json(TARGET/'initial_certificate.json'))
    forged['coordinate_proofs'][0]['state_cells'][0]['deviation']='0'
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_phase_handoff(system,recurrent,forged)
    payload=read_json(SOURCE/'system.json')
    payload['observation']['biases'][0][0]=0.01
    shifted=ContinuousReLUSystem.from_dict(payload)
    with pytest.raises(ValueError,match='Four direct observed'):
        _observation(shifted)
