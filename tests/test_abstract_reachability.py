"""Exact abstract graph closure and executable pruned machine."""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import pytest

from ncd.abstract_reachability import (
    graph_initial,graph_output,graph_step,verify_abstract_closure)
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_only'/'certificate.json'
TWO_STAGE=ROOT/'runs'/'reachable_two_stage_v1'/'affine_d128'/'certificate.json'
CERT=ROOT/'runs'/'abstract_reachability_v1'/'affine_d128'/'certificate.json'


def test_exact_graph_closure_replays_and_prunes_outer_box():
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    prior=read_json(TWO_STAGE)
    certificate=read_json(CERT)
    assert verify_abstract_closure(system,recurrent,prior,certificate)=={
        'status':'verified','initial_state_count':243,
        'initial_successor_count':1952,'recurrent_state_count':2060,
        'closure_edge_count':51304,'upper_bound':2303}
    active=certificate['active_axes']
    low=[prior['recurrent_index_low'][i] for i in active]
    high=[prior['recurrent_index_high'][i] for i in active]
    kept={tuple(x) for x in certificate['recurrent_active_indices']}
    omitted=next(q for q in product(
        *(range(a,b+1) for a,b in zip(low,high))) if q not in kept)
    state=[0]*system.state_dim
    for i,q in zip(active,omitted):state[i]=q
    with pytest.raises(ValueError,match='not in certified graph closure'):
        graph_output(system,recurrent,certificate,('recurrent',tuple(state)))
    point=tuple(Q(i%2) for i in range(system.state_dim))
    concrete=list(point)
    abstract=graph_initial(system,point)
    for action in (tuple(Q(0) for _ in range(system.action_dim)),
                   tuple(Q(1) for _ in range(system.action_dim))):
        output=graph_output(system,recurrent,certificate,abstract)
        actual=value(system.observation,concrete)
        assert max(abs(a-b) for a,b in zip(output,actual))<=Q(recurrent['epsilon'])
        abstract=graph_step(system,recurrent,certificate,abstract,action)
        concrete=value(system.transition,concrete+list(action))
    forged=deepcopy(certificate)
    forged['recurrent_active_indices'].pop()
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_abstract_closure(system,recurrent,prior,forged)
