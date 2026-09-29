"""Boundary geometry, frozen 128D closed machine and tamper rejection."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json
from ncd.nonlinear_action_closure import (
    SQUARE,_clip,_evaluate,_compile,_regions,verify_nonlinear_closure,
    nonlinear_initial,nonlinear_output,nonlinear_step)

ROOT=Path(__file__).resolve().parents[1]


def test_closed_polygon_clipping_keeps_boundary_points_and_segments():
    assert set(_clip(SQUARE,(Q(1),Q(0),Q(-1))))=={(Q(1),Q(0)),(Q(1),Q(1))}
    point=_clip(_clip(SQUARE,(Q(1),Q(0),Q(-1))),(Q(0),Q(1),Q(-1)))
    assert point==((Q(1),Q(1)),)
    assert _clip(point,(Q(-1),Q(0),Q(1)))==point
    assert _clip(point,(Q(-1),Q(0),Q(0)))==()
    assert _clip(SQUARE,(Q(0),Q(0),Q(0)))==SQUARE


def test_exact_128d_nonlinear_closed_machine():
    source=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/'d_128'
    system=ContinuousReLUSystem.from_dict(read_json(source/'system.json'))
    recurrent=read_json(source/'certificate.json')
    initial=read_json(ROOT/'runs'/'optimal_initial_grid_v1'/'seed_6101'/'d_128'/'certificate.json')
    certificate=read_json(ROOT/'runs'/'nonlinear_action_closure_v1'/'seed_6101'/'d_128'/'certificate.json')
    assert verify_nonlinear_closure(system,recurrent,initial,certificate)=={
        'status':'verified','initial_state_count':108,'recurrent_state_count':55,'upper_bound':163}
    assert certificate['distinct_initial_successors']==55
    assert certificate['initial_retained_edges']==617
    assert certificate['recurrent_retained_edges']==311
    active=certificate['active_coordinates']
    compiled=_compile(system,active)
    center=[Q(1,2)]*system.state_dim
    crossing,pieces=_regions(compiled,center,{})
    assert len(crossing)==4 and len(pieces)==5
    for poly,forms in pieces:
        # One exact representative per region independently evaluates the
        # original dense network, rather than the extracted sparse forms.
        point=tuple(sum((p[k] for p in poly),Q(0))/len(poly) for k in range(2))
        actual=value(system.transition,center+list(point))
        assert tuple(_evaluate(f,point) for f in forms)==tuple(actual[i] for i in active)
    point=tuple(Q(i%2) for i in range(system.state_dim))
    concrete=list(point)
    state=nonlinear_initial(system,recurrent,initial,certificate,point)
    actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,128),Q(127,128)),(Q(1,2),Q(1,2)),(Q(1),Q(1)))
    for time in range(len(actions)+1):
        output=nonlinear_output(system,recurrent,initial,certificate,state)
        actual=value(system.observation,concrete)
        assert max(abs(a-b) for a,b in zip(output,actual))<=Q(recurrent['epsilon'])
        if time<len(actions):
            state=nonlinear_step(system,recurrent,initial,certificate,state,actions[time])
            concrete=value(system.transition,concrete+list(actions[time]))
    forged=deepcopy(certificate)
    forged['recurrent_graph'][0]['successors'].pop()
    with pytest.raises(ValueError,match='mismatch'):
        verify_nonlinear_closure(system,recurrent,initial,forged)
    omitted=tuple(3 if i==0 else 0 for i in range(system.state_dim))
    with pytest.raises(ValueError,match='outside certified nonlinear closure'):
        nonlinear_output(system,recurrent,initial,certificate,('recurrent',omitted))



def test_portable_program_exact_execution_without_source_network(monkeypatch):
    import ncd.nonlinear_action_closure as module
    source=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/'d_128'
    system=ContinuousReLUSystem.from_dict(read_json(source/'system.json'))
    recurrent=read_json(source/'certificate.json')
    initial=read_json(ROOT/'runs'/'optimal_initial_grid_v1'/'seed_6101'/'d_128'/'certificate.json')
    target=ROOT/'runs'/'nonlinear_action_closure_v1'/'seed_6101'/'d_128'
    certificate=read_json(target/'certificate.json')
    program=read_json(target/'program.json')
    assert module.verify_nonlinear_program(system,recurrent,initial,certificate,program)=={
        'status':'verified','upper_bound':163,'action_template_count':1}
    point=(Q(1),)*system.state_dim
    actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,2),Q(1,2)),(Q(1,128),Q(127,128)),(Q(1),Q(1)))
    expected=[]
    state=nonlinear_initial(system,recurrent,initial,certificate,point)
    expected.append((state,nonlinear_output(system,recurrent,initial,certificate,state)))
    for action in actions:
        state=nonlinear_step(system,recurrent,initial,certificate,state,action)
        expected.append((state,nonlinear_output(system,recurrent,initial,certificate,state)))
    forged=deepcopy(program)
    forged['action_templates'][0]['affine_forms'][0][0][-1]='1/1000'
    with pytest.raises(ValueError,match='replay mismatch'):
        module.verify_nonlinear_program(system,recurrent,initial,certificate,forged)

    def forbidden(*args,**kwargs):
        raise AssertionError('Standalone execution accessed source network')
    for name in ('initial_grid_step','initial_grid_output','_compile','_regions'):
        monkeypatch.setattr(module,name,forbidden)
    state=module.program_initial(program,point)
    for time in range(len(actions)+1):
        assert (state,module.program_output(program,state))==expected[time]
        if time<len(actions):
            state=module.program_step(program,state,actions[time])
