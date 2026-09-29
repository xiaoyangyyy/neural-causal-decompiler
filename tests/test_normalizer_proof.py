from copy import deepcopy
from fractions import Fraction as Q
import numpy as np
import pytest
from normalizer_proof.realization import Algebraic,certify_normalizer,verify_normalizer,execute_exact,normalizer_branches
from normalizer_proof.audit import audit_actual_interventions,number
from ncd.cdir import Node

TARGET='runs/oblique_seed1193/teacher.pt'
@pytest.fixture(scope='module')
def certificate():return certify_normalizer(TARGET)


def test_radical_comparison_and_clipping_are_exact():
    a=Algebraic.sqrt(2)
    assert a.compare_rational(Q(7,5))==1 and a.compare_rational(Q(3,2))==-1
    assert Algebraic(-2,2).compare_rational(Q(-7,5))==-1
    assert Algebraic(-2,2).compare_rational(Q(-3,2))==1
    assert Algebraic.sqrt(Q(9,4)).to_dict()==Algebraic(Q(3,2)).to_dict()
    assert Algebraic(Q(10)**200).clip(-20,20).to_dict()==Algebraic(20).to_dict()
    assert Algebraic(-Q(10)**200).clip(-20,20).to_dict()==Algebraic(-20).to_dict()
    assert Algebraic(20).clip(-20,20).to_dict()==Algebraic(20).to_dict()


def test_zero_variance_floor_equality_and_clip_boundaries(certificate):
    program=certificate['program'];f=Q(program['floor'])
    zero=execute_exact(program,[[0,0]]*16)
    assert all(g['natural_std_case']=='floor' for g in zero['guards'])
    at=execute_exact(program,[[(-f if i%2 else f),0] for i in range(16)])
    assert at['guards'][0]['natural_std_case']=='std'
    assert at['guards'][0]['visited_conditional_child']==2
    sources=[[[0,0]]*16 for _ in range(4)]
    rows=[[v*f,0] for v in [-21,-20,0,20,21]+[0]*11]
    result=execute_exact(program,rows,sources,[True,False,True,False])
    assert [number(row[0]) for row in result['outputs'][:5]]==[-20,-20,0,20,20]
    assert result['guards'][0]['clipping_cases'][:5]==['lower','inside','inside','inside','upper']


def test_independent_sources_and_collateral_changes(certificate):
    program=certificate['program'];data=[[i,-i] for i in range(16)]
    sources=[[[3,0]]*16,[[0,-7]]*16,[[(-2 if i%2 else 2),0] for i in range(16)],[[0,(-4 if i%2 else 4)] for i in range(16)]]
    actual=execute_exact(program,data,sources,[True]*4)
    assert [number(actual['states'][name]) for name in ['mean_x','mean_y','std_x','std_y']]==[3,-7,2,4]
    partial=execute_exact(program,data,sources,[True,False,False,False])
    natural=execute_exact(program,data)
    assert partial['states']['std_raw_x']==natural['states']['std_raw_x']
    assert partial['states']['mean_y']==natural['states']['mean_y']
    assert partial['states']['clipped_x']!=natural['states']['clipped_x']
    assert partial['states']['clipped_y']==natural['states']['clipped_y']
    with pytest.raises(ValueError):execute_exact(program,data,sources[:1],[True]*4)
    with pytest.raises(ValueError):execute_exact(program,data,sources,[1,False,False,False])


def test_lowered_cdir_matches_unintervened_numeric_execution(certificate):
    data=np.random.default_rng(8100).normal(size=(96,2))
    exact=execute_exact(certificate['program'],data.tolist())
    for j in range(2):
        expression=Node.from_dict(certificate['program']['coordinates'][j]['normalized_expression'])
        expected=np.clip(expression.evaluate(data),-20,20)
        assert np.allclose(expected,[number(row[j]) for row in exact['outputs']],atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('field',['graph','masks','cut','program','closure','weight'])
def test_certificate_forgery_rejected(certificate,field):
    altered=deepcopy(certificate)
    if field=='graph':next(n for n in altered['fx_export']['nodes'] if n['target']=='std')['kwargs']['unbiased']=True
    elif field=='masks':altered['masks'].pop()
    elif field=='cut':altered['continued_boundary_edges'].pop()
    elif field=='program':altered['program']['coordinates'][0]['normalized_expression']['op']='mul'
    elif field=='closure':altered['original_R5_closed']=True
    else:altered['fx_export']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):verify_normalizer(altered)


def test_unsupported_sample_variance_and_division_floor_rejected(certificate):
    altered=deepcopy(certificate['fx_export'])
    next(n for n in altered['nodes'] if n['target']=='std')['kwargs']['unbiased']=True
    with pytest.raises(ValueError):normalizer_branches(altered)
    altered=deepcopy(certificate['fx_export'])
    for n in altered['nodes']:
        if n['target']=='clamp_min':n['args'][1]=1e-10
    with pytest.raises(ValueError):normalizer_branches(altered)


def test_actual_network_combined_interventions_and_continuation(certificate):
    verified=verify_normalizer(certificate)
    assert verified['mathematical_final_logits_equal'] and not verified['original_R5_closed']
    audit=audit_actual_interventions(certificate)
    assert audit['executions']==96 and audit['independent_sources']
    assert audit['max_final_absolute_error']<1e-9
    assert set(audit['floor_cases'])=={'floor','std'}
    assert set(audit['clip_cases'])=={'lower','inside','upper'}
