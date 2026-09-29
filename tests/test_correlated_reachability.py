"""Exact shared-action closure, executable reduction and tamper rejection."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest

from ncd.correlated_reachability import (
    correlated_initial,correlated_output,correlated_step,
    verify_correlated_closure)
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_only'/'certificate.json'
TWO_STAGE=ROOT/'runs'/'reachable_two_stage_v1'/'affine_d128'/'certificate.json'
ABSTRACT=ROOT/'runs'/'abstract_reachability_v1'/'affine_d128'/'certificate.json'
CERT=ROOT/'runs'/'correlated_reachability_v1'/'affine_d128'/'certificate.json'


def test_correlated_closure_replay_machine_and_tampering():
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    two_stage=read_json(TWO_STAGE)
    abstract=read_json(ABSTRACT)
    certificate=read_json(CERT)
    assert verify_correlated_closure(
        system,recurrent,two_stage,abstract,certificate)=={
            'status':'certified','initial_state_count':243,
            'initial_successor_count':859,'recurrent_state_count':899,
            'initial_candidate_edges':6768,
            'recurrent_candidate_edges':20672,
            'recurrent_retained_edges':6368,'upper_bound':1142}
    assert certificate['pair_separator_count']==60
    kept={tuple(row) for row in certificate['recurrent_active_indices']}
    omitted=next(tuple(row) for row in abstract['recurrent_active_indices']
                 if tuple(row) not in kept)
    state=[0]*system.state_dim
    for axis,index in zip(certificate['active_axes'],omitted):
        state[axis]=index
    with pytest.raises(ValueError,match='not in correlated graph closure'):
        correlated_output(system,recurrent,certificate,('recurrent',tuple(state)))
    point=tuple(Q(i%2) for i in range(system.state_dim))
    concrete=list(point)
    abstract_state=correlated_initial(system,point)
    actions=(tuple(Q(0) for _ in range(system.action_dim)),
             tuple(Q(1) for _ in range(system.action_dim)),
             tuple(Q(1,2) for _ in range(system.action_dim)))
    for action in actions:
        output=correlated_output(system,recurrent,certificate,abstract_state)
        actual=value(system.observation,concrete)
        assert max(abs(a-b) for a,b in zip(output,actual))<=Q(recurrent['epsilon'])
        abstract_state=correlated_step(
            system,recurrent,certificate,abstract_state,action)
        concrete=value(system.transition,concrete+list(action))
    forged=deepcopy(certificate)
    forged['recurrent_active_indices'].pop()
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_correlated_closure(system,recurrent,two_stage,abstract,forged)
