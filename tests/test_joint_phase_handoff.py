"""Independent vertex enumeration and coupled-copy analytic extrema."""
from fractions import Fraction as Q
from itertools import combinations,product
import pytest
from ncd.rational_polytope import box,clip,evaluate
from ncd.joint_phase_initial_handoff import _difference_pieces


def _solve(rows):
    n=len(rows)
    matrix=[list(r[:-1])+[-r[-1]] for r in rows]
    for col in range(n):
        pivot=next((k for k in range(col,n) if matrix[k][col]),None)
        if pivot is None:
            return None
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        divisor=matrix[col][col]
        matrix[col]=[v/divisor for v in matrix[col]]
        for k in range(n):
            if k!=col and matrix[k][col]:
                scale=matrix[k][col]
                matrix[k]=[a-scale*b for a,b in zip(matrix[k],matrix[col])]
    return tuple(r[-1] for r in matrix)


def test_four_dimensional_clipping_matches_independent_basis_enumeration():
    poly=box([0]*4,[1]*4)
    cuts=((Q(1),Q(2),Q(-1),Q(1),Q(-1,3)),(Q(-1),Q(1),Q(2),Q(-2),Q(1,4)))
    for form in cuts:
        poly=clip(poly,form)
    exact=set()
    for rows in combinations(poly.constraints,4):
        point=_solve(rows)
        if point is not None and all(evaluate(f,point)>=0 for f in poly.constraints):
            exact.add(point)
    assert set(poly.vertices)==exact
    assert any(p[k].denominator>1 for p in poly.vertices for k in range(4))
    for direction in product((-1,1),repeat=4):
        form=tuple(map(Q,direction))+(Q(0),)
        assert max(evaluate(form,p) for p in poly.vertices)==max(evaluate(form,p) for p in exact)


def test_closed_face_segment_and_point_survive_clipping():
    poly=box([0]*4,[1]*4)
    for coordinate in range(3):
        form=tuple(Q(-int(k==coordinate)) for k in range(4))+(Q(0),)
        poly=clip(poly,form)
        assert len(poly.vertices)==2**(3-coordinate)
    assert poly.vertices==((Q(0),Q(0),Q(0),Q(0)),(Q(0),Q(0),Q(0),Q(1)))
    point=clip(poly,(Q(0),Q(0),Q(0),Q(-1),Q(0)))
    assert point.vertices==((Q(0),)*4,)
    assert not clip(point,(Q(0),Q(0),Q(0),Q(0),Q(-1))).vertices


def test_signed_coupled_relu_difference_has_exact_tighter_extrema():
    terms=((Q(1),(Q(1),Q(0),Q(1),Q(0),Q(-1,2))),
           (Q(-1),(Q(1),Q(0),Q(1),Q(0),Q(-3,4))))
    pieces,minimum,maximum,splits=_difference_pieces(terms,[0]*4,[1]*4,(Q(1,2),Q(1,2)))
    assert minimum[0]==Q(-1,4) and maximum[0]==Q(1,4)
    assert splits>0 and len(pieces)>1
    for poly,form in pieces:
        for p in poly.vertices:
            direct=sum(w*max(Q(0),evaluate(f,p)) for w,f in terms)
            centered=(Q(1,2),Q(1,2),p[2],p[3])
            direct-=sum(w*max(Q(0),evaluate(f,centered)) for w,f in terms)
            assert direct==evaluate(form,p)
    with pytest.raises(ValueError,match='region budget'):
        _difference_pieces(terms,[0]*4,[1]*4,(Q(1,2),Q(1,2)),max_regions=1)


from copy import deepcopy
from pathlib import Path
from ncd.io import read_json
from ncd.continuous_separation import ContinuousReLUSystem,ReLUMLP
from ncd.continuous_compositional_realization import certify_weighted
from ncd.joint_phase_initial_handoff import certify_joint_handoff,verify_joint_handoff,_extract
from ncd.phase_initial_handoff import certify_phase_handoff
from ncd.trained_nonlinear_realization import candidate_relation

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'runs/mixed_training_pilot_v1/seed_7000/d_8'


def test_mixed_program_matches_neural_machine_with_source_helpers_disabled(monkeypatch):
    import ncd.nonlinear_action_closure as module
    system=ContinuousReLUSystem.from_dict(read_json(PILOT/'system.json'))
    recurrent=read_json(PILOT/'recurrent_certificate.json')
    initial=read_json(PILOT/'initial_certificate.json')
    closure=read_json(PILOT/'closure_certificate.json')
    program=read_json(PILOT/'program.json')
    assert verify_joint_handoff(system,recurrent,initial)['minimum_initial_label_count']==81
    with pytest.raises(ValueError,match='Mixed state/action'):
        certify_phase_handoff(system,recurrent)
    assert module.verify_nonlinear_program(system,recurrent,initial,closure,program)=={
        'status':'verified','upper_bound':135,'action_template_count':10}
    point=(Q(1),)*8
    actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,2),Q(1,2)),(Q(1,128),Q(127,128)),(Q(1),Q(1)))
    state=module.nonlinear_initial(system,recurrent,initial,closure,point)
    expected=[(state,module.nonlinear_output(system,recurrent,initial,closure,state))]
    for action in actions:
        state=module.nonlinear_step(system,recurrent,initial,closure,state,action)
        expected.append((state,module.nonlinear_output(system,recurrent,initial,closure,state)))
    def forbidden(*args,**kwargs):
        raise AssertionError('Mixed standalone program called neural execution/compiler')
    for name in ('initial_grid_step','initial_grid_output','_compile','_regions'):
        monkeypatch.setattr(module,name,forbidden)
    state=module.program_initial(program,point)
    for time in range(len(actions)+1):
        assert (state,module.program_output(program,state))==expected[time]
        if time<len(actions):
            state=module.program_step(program,state,actions[time])


def test_joint_proof_rejects_forged_geometry_and_extra_state_support():
    payload=read_json(PILOT/'system.json')
    system=ContinuousReLUSystem.from_dict(payload)
    recurrent=read_json(PILOT/'recurrent_certificate.json')
    forged=deepcopy(read_json(PILOT/'initial_certificate.json'))
    forged['coordinate_proofs'][0]['state_cells'][0]['joint_partition_sha256']='0'*64
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_joint_handoff(system,recurrent,forged)
    extra=deepcopy(payload)
    extra['transition']['weights'][0][0][2]=0.1
    with pytest.raises(ValueError,match='two state support'):
        _extract(ContinuousReLUSystem.from_dict(extra),0)
    changed=deepcopy(payload)
    changed['transition']['weights'][0][0][0]+=0.01
    with pytest.raises(ValueError,match='Weighted certificate'):
        verify_joint_handoff(ContinuousReLUSystem.from_dict(changed),recurrent,read_json(PILOT/'initial_certificate.json'))


def test_valid_recurrent_relation_does_not_make_failed_initial_handoff_certified():
    import ncd.nonlinear_action_closure as module
    d=6
    first=tuple(tuple(float(i==j) for j in range(d+2)) for i in range(d))
    gains=(.25,.25,.25,.25,.2,.55)
    second=tuple(tuple(gains[i] if i==j else 0.0 for j in range(d)) for i in range(d))
    transition=ReLUMLP((first,second),((0.0,)*d,(.2,)*d))
    observation=ReLUMLP((tuple(tuple(float(i==j) for j in range(d)) for i in range(4)),),((0.0,)*4,))
    system=ContinuousReLUSystem(d,2,transition,observation)
    bins,radii=candidate_relation(d)
    recurrent=certify_weighted(system,bins,radii,action_bins=128,epsilon='17/100')
    assert recurrent['status']=='certified'
    initial=certify_joint_handoff(system,recurrent)
    assert initial['status']=='unresolved' and initial['failed_coordinates']==[5]
    assert initial['upper_bound'] is None and initial['minimum_initial_label_count'] is None
    assert Q(initial['minimum_handoff_slack'])<0
    assert verify_joint_handoff(system,recurrent,initial)['status']=='unresolved'
    with pytest.raises(ValueError,match='Unresolved joint initial handoff'):
        module.certify_nonlinear_closure(system,recurrent,initial)
    with pytest.raises(ValueError,match='Positive'):
        certify_joint_handoff(system,recurrent,max_regions=0)
