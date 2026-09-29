"""Exact initial-grid optimality and executable full-cube handoff."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.initial_grid_synthesis import (
    verify_initial_grid,initial_grid_initial,initial_grid_output,initial_grid_step)
from ncd.io import read_json

ROOT=Path(__file__).resolve().parents[1]


def test_exact_initial_grid_minimum_and_runtime():
    for dimension,initial in ((8,81),(128,108)):
        source=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/f'd_{dimension}'
        proof=ROOT/'runs'/'optimal_initial_grid_v1'/'seed_6101'/f'd_{dimension}'/'certificate.json'
        system=ContinuousReLUSystem.from_dict(read_json(source/'system.json'))
        recurrent=read_json(source/'certificate.json')
        certificate=read_json(proof)
        assert verify_initial_grid(system,recurrent,certificate)=={
            'status':'verified','initial_state_count':initial,
            'exact_minimum_initial_grid_count':initial,
            'recurrent_state_count':216,'upper_bound':initial+216}
        assert certificate['initial_coordinate_bins'][-1]==1
        exclusions=certificate['baseline_handoff_excluded_coordinates']
        assert exclusions==([] if dimension==8 else [0])
        if dimension==128:
            assert Q(certificate['baseline_handoff_error_upper'][0])>Q(recurrent['coordinate_radii'][0])
        point=tuple(Q(i%2) for i in range(dimension))
        concrete=list(point)
        state=initial_grid_initial(system,certificate,point)
        actions=(Q(0),Q(1,128),Q(1,2),Q(127,128),Q(1))
        for time in range(len(actions)+1):
            output=initial_grid_output(system,recurrent,certificate,state)
            actual=value(system.observation,concrete)
            assert max(abs(a-b) for a,b in zip(output,actual))<=Q(recurrent['epsilon'])
            if time<len(actions):
                action=(actions[time],)*system.action_dim
                state=initial_grid_step(system,recurrent,certificate,state,action)
                concrete=value(system.transition,concrete+list(action))
        forged=deepcopy(certificate)
        forged['exact_minimum_initial_grid_count']-=1
        with pytest.raises(ValueError,match='mismatch'):
            verify_initial_grid(system,recurrent,forged)
        with pytest.raises(ValueError):
            initial_grid_output(system,recurrent,certificate,('initial',0))
        with pytest.raises(ValueError):
            initial_grid_step(system,recurrent,certificate,state,(Q(-1),)*system.action_dim)
