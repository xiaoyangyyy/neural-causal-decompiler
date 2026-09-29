"""Exact joint state/action grid closure and executable reduction."""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import pytest

from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dual_grid_reachability import (
    dual_grid_initial,dual_grid_output,dual_grid_step,
    verify_dual_grid_closure)
from ncd.io import read_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
TWO_STAGE=ROOT/'runs'/'dual_grid_reachability_v1'/'affine_d128'/'two_stage_certificate.json'
CERT=ROOT/'runs'/'dual_grid_reachability_v1'/'affine_d128'/'closure_certificate.json'


def test_dual_grid_exact_replay_and_executable_closure():
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    two_stage=read_json(TWO_STAGE)
    certificate=read_json(CERT)
    assert [recurrent['coordinate_bins'][i] for i in (0,1,2,3,127)]==[14,6,6,6,9]
    assert verify_dual_grid_closure(system,recurrent,two_stage,certificate)=={
        'status':'certified','action_bins':512,'pair_separator_count':60,
        'initial_state_count':243,'initial_candidate_edges':6480,
        'initial_retained_edges':2051,'initial_successor_count':780,
        'recurrent_candidate_edges':14696,'recurrent_retained_edges':4901,
        'recurrent_state_count':798,'upper_bound':1041}
    active=certificate['active_axes']
    kept={tuple(row) for row in certificate['recurrent_active_indices']}
    omitted=next(q for q in product(*(
        range(two_stage['recurrent_index_low'][i],
              two_stage['recurrent_index_high'][i]+1)
        for i in active)) if q not in kept)
    state=[0]*system.state_dim
    for axis,index in zip(active,omitted):
        state[axis]=index
    with pytest.raises(ValueError,match='not in dual-grid closure'):
        dual_grid_output(system,recurrent,certificate,('recurrent',tuple(state)))
    point=tuple(Q(i%2) for i in range(system.state_dim))
    concrete=list(point)
    abstract=dual_grid_initial(system,point)
    actions=(tuple(Q(0) for _ in range(system.action_dim)),
             tuple(Q(1) for _ in range(system.action_dim)),
             tuple(Q(1,2) for _ in range(system.action_dim)))
    for action in actions:
        output=dual_grid_output(system,recurrent,certificate,abstract)
        actual=value(system.observation,concrete)
        assert max(abs(a-b) for a,b in zip(output,actual))<=Q(recurrent['epsilon'])
        abstract=dual_grid_step(system,recurrent,certificate,abstract,action)
        concrete=value(system.transition,concrete+list(action))
    forged=deepcopy(certificate)
    forged['recurrent_active_indices'].pop()
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_dual_grid_closure(system,recurrent,two_stage,forged)
