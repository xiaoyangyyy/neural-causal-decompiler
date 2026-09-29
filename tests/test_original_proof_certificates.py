from copy import deepcopy
from decimal import Decimal,localcontext
from fractions import Fraction as Q
import json
import pytest
from ncd.proof_intervals import Interval,exp_point,log_point,hyperbolic_tangent,trig_point,protected_division,attention
from ncd.gaussian_identifiability import certify_gaussian_nonidentifiability,verify_gaussian_nonidentifiability
from ncd.frozen_mechanism_proof import certify_mechanism,verify_mechanism
from ncd.original_proof_workflow import prove,verify_proof,audit_requirements,statistical_bound,verify_statistical
from ncd.scm_error_certificate import certify_scm_error,verify_scm_error
from ncd.io import save_json


def decimal_value(q):return Decimal(q.numerator)/Decimal(q.denominator)

@pytest.mark.parametrize('x',['-3','-1/10','0','1/10','3'])
def test_transcendental_enclosures_against_independent_decimal(x):
    q=Q(x)
    with localcontext() as ctx:
        ctx.prec=100
        d=decimal_value(q);e=exp_point(q);t=hyperbolic_tangent(Interval.point(q))
        truth=d.exp();tanh=(d*2).exp();tanh=(tanh-1)/(tanh+1)
        assert decimal_value(e.lo)<=truth<=decimal_value(e.hi)
        assert decimal_value(t.lo)<=tanh<=decimal_value(t.hi)
        positive=q*q+1;l=log_point(positive)
        assert decimal_value(l.lo)<=decimal_value(positive).ln()<=decimal_value(l.hi)


def test_interval_sqrt_and_protected_zero_boundary():
    x=Interval(Q(1,3),Q(7,5));r=x.sqrt()
    assert r.lo*r.lo<=x.lo and r.hi*r.hi>=x.hi
    v=protected_division(Interval.point(1),Interval('-1/1000000000','1/1000000000'))
    assert v.lo<=-100000000 and v.hi>=100000000
    a=attention([Interval.point(0)],[[Interval.point(0)],[Interval.point(0)]],[[Interval.point(1)],[Interval.point(3)]])
    assert a[0].lo<=2<=a[0].hi
    assert trig_point(0).lo==trig_point(0).hi==0


def test_gaussian_full_law_witness_and_forged_covariance():
    c=certify_gaussian_nonidentifiability();assert verify_gaussian_nonidentifiability(c)['conclusion']=='refuted'
    changed=deepcopy(c);changed['observational_covariance'][0][1]='3/4'
    with pytest.raises(ValueError):verify_gaussian_nonidentifiability(changed)
    with pytest.raises(ValueError):certify_gaussian_nonidentifiability('1')


def simple_network():
    return {'schema':'ncd.frozen-tanh-mechanism.v1','parents':[0],'mean':['0'],'std':['1'],
        'output_mean':'0','output_scale':'1','layers':[{'weights':[['1']],'bias':['0'],'activation':'identity'}]}


def test_cover_and_strict_counterexample_are_distinct():
    net=simple_network();identity={'op':'var','index':0};constant={'op':'constant','value':0}
    point=certify_mechanism(net,identity,[['0','0']])
    assert verify_mechanism(net,identity,point)['conclusion']=='proved'
    bad=certify_mechanism(net,constant,[['1','2']])
    assert verify_mechanism(net,constant,bad)['conclusion']=='refuted'
    unknown=certify_mechanism(net,identity,[['-1','1']],max_boxes=1)
    assert verify_mechanism(net,identity,unknown)['conclusion']=='unresolved'
    erased=deepcopy(unknown);erased['tree'][0]['left']=erased['tree'][0]['right']
    with pytest.raises(ValueError):verify_mechanism(net,identity,erased)
    forged=deepcopy(point);forged['tree'][0]['error']=['-1','1']
    with pytest.raises(ValueError):verify_mechanism(net,identity,forged)


def test_conditional_scm_never_verifies_its_own_assumptions():
    c=certify_scm_error([[0,1],[0,0]],['1/100','1/50'],[['0','2'],['0','0']],['0','1/100'])
    assert c['coordinate_error_bounds']==['1/100','1/20']
    assert verify_scm_error(c)['premises_verified'] is False
    do=certify_scm_error([[0,1],[0,0]],['1','1'],[['0','2'],['0','0']],['0','0'],[0])
    assert do['coordinate_error_bounds']==['0','1']
    altered=deepcopy(c);altered['joint_wasserstein_l1_upper']='0'
    with pytest.raises(ValueError):verify_scm_error(altered)
    with pytest.raises(ValueError):certify_scm_error([[0,1],[1,0]],['0','0'],[['0','1'],['1','0']],['0','0'])


def test_statistical_world_units_and_family_correction():
    units=[{'world_id':str(i),'success':'1'} for i in range(100)]
    c=statistical_bound(units,family_size=10)
    assert c['local_delta']=='1/1000' and c['independence_proved_by_this_certificate'] is False
    assert verify_statistical(c)['conclusion']=='proved-conditionally'
    with pytest.raises(ValueError):statistical_bound([units[0],units[0]])


def test_ledger_keeps_all_requirements_open_after_scoped_results(tmp_path):
    (tmp_path/'requirements.md').write_text('Original requirements R0 through R13')
    config={'root':'.','output':'bundle','requirements':'requirements.md','jobs':[{'id':'gaussian','kind':'gaussian_nonidentifiability'}]}
    save_json(tmp_path/'config.json',config)
    prove(tmp_path/'config.json');audit=audit_requirements(tmp_path/'bundle')
    assert len(audit['requirements'])==14 and audit['overall_objective_achieved'] is False
    assert audit['requirements']['R0']['refuted']==2 and 'R10.noise' in audit['unresolved_claims']
    ledger=json.loads((tmp_path/'bundle/ledger.json').read_text())
    ledger['claims'][0]['status']='proved';save_json(tmp_path/'bundle/ledger.json',ledger)
    with pytest.raises(ValueError):verify_proof(tmp_path/'bundle')
