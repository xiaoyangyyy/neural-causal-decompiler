from fractions import Fraction as Q
from copy import deepcopy
import numpy as np
import pytest
from ncd.proof_intervals import Interval
from ncd.cdir import Node
from proof_workbench.piecewise_mechanism import neural_gradient,certify_piecewise,verify_piecewise,program,execute_program


def example():
    return {'schema':'ncd.frozen-tanh-mechanism.v1','checkpoint_sha256':'unit-fixture',
        'parents':[0,1],'mean':['0','0'],'std':['1','1'],'output_mean':'0','output_scale':'1',
        'layers':[{'weights':[['1/4','1/4']],'bias':['0'],'activation':'tanh'},
            {'weights':[['1']],'bias':['0'],'activation':'tanh'},
            {'weights':[['1']],'bias':['0'],'activation':'identity'}]}


def test_affine_cells_cover_entire_domain_and_boundaries():
    domain=[['-1','1'],['-1','1']]
    c=certify_piecewise(example(),domain,max_cells=511,seconds=10)
    assert verify_piecewise(c)['conclusion']=='proved'
    spec=program(c);x=np.array([[a,b] for a in np.linspace(-1,1,41) for b in np.linspace(-1,1,41)])
    truth=np.tanh(np.tanh(x.sum(1)/4))
    assert np.max(abs(execute_program(spec,x)-truth))<=.01
    bad=deepcopy(c);split=next(row for row in bad['nodes'] if row.get('kind')=='split')
    bad['nodes'][split['right']]['box'][split['axis']][0]='1'
    with pytest.raises(ValueError):verify_piecewise(bad)
    bad=deepcopy(c);leaf=next(row for row in bad['nodes'] if row.get('kind')=='leaf')
    leaf['candidate']['coefficients'].pop()
    with pytest.raises(ValueError):verify_piecewise(bad)


def test_budget_exhaustion_retains_unresolved_cells_and_resume_contract():
    first=certify_piecewise(example(),[['-1','1'],['-1','1']],max_cells=1,seconds=1)
    assert verify_piecewise(first)['conclusion']=='unresolved'
    with pytest.raises(ValueError):program(first)
    resumed=certify_piecewise(example(),first['domain'],max_cells=511,seconds=10,checkpoint=first)
    assert verify_piecewise(resumed)['conclusion']=='proved'
    altered=deepcopy(example());altered['layers'][0]['weights'][0][0]='2'
    with pytest.raises(ValueError):certify_piecewise(altered,first['domain'],checkpoint=first)


def test_gradients_are_real_network_derivatives_including_saturation():
    for a,b in [(0,0),(.5,-.2),(20,20)]:
        _,gradient=neural_gradient(example(),[Interval.point(Q(a)),Interval.point(Q(b))])
        t=np.tanh((a+b)/4);expected=(1-t*t)*(1-np.tanh(t)**2)/4
        assert all(abs(float((v.lo+v.hi)/2)-expected)<1e-14 for v in gradient)
    root=example();root['parents']=[];root['mean']=['0'];root['std']=['1'];root['layers'][0]['weights']=[['1']]
    _,gradient=neural_gradient(root,[Interval.point(0)]*2)
    assert all(v.lo==v.hi==0 for v in gradient)
