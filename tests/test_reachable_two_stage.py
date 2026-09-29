"""Exact reachable two-stage machine and executable transitions."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest

from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json
from ncd.reachable_two_stage import (
    two_stage_initial,two_stage_output,two_stage_step,verify_two_stage)

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_only'/'certificate.json'
CERT=ROOT/'runs'/'reachable_two_stage_v1'/'affine_d128'/'certificate.json'


def test_exact_4723_state_machine_replays_and_runs():
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    certificate=read_json(CERT)
    assert verify_two_stage(system,recurrent,certificate)=={
        'status':'verified','initial_state_count':243,
        'recurrent_state_count':4480,'upper_bound':4723}
    assert [certificate['recurrent_index_high'][i]-
            certificate['recurrent_index_low'][i]+1
            for i in (0,1,2,3,127)]==[14,4,4,4,5]
    point=tuple(Q(i%2) for i in range(system.state_dim))
    state=two_stage_initial(system,point)
    assert state[0]=='initial'
    concrete=list(point)
    for action in (
            tuple(Q(0) for _ in range(system.action_dim)),
            tuple(Q(1) for _ in range(system.action_dim)),
            tuple(Q(1,2) for _ in range(system.action_dim))):
        output=two_stage_output(system,recurrent,state)
        actual=value(system.observation,concrete)
        assert max(abs(a-b) for a,b in zip(actual,output))<=Q(recurrent['epsilon'])
        state=two_stage_step(system,recurrent,state,action)
        assert state[0]=='recurrent'
        concrete=value(system.transition,concrete+list(action))
    forged=deepcopy(certificate)
    forged['upper_bound']=4722
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_two_stage(system,recurrent,forged)
