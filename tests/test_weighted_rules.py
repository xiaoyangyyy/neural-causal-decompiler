import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from ncd.rules import fit_rule


def test_default_matches_frozen_release():
    path=Path(__file__).resolve().parents[1]/'validation/wheel_v3_env/ncd/rules.py'
    if not path.exists():pytest.skip('Frozen release is a local regression artifact')
    spec=importlib.util.spec_from_file_location('ncd._frozen_rules',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    rng=np.random.default_rng(19);x=rng.normal(size=(48,7));y=((x[:,0]+x[:,2]>0).astype(int)+2*(x[:,3]>1)).astype(int)
    kwargs=dict(max_splits=2,min_leaf=3)
    old,ot=module.fit_rule(x,y,tuple('abcdefg'),**kwargs)
    new,nt=fit_rule(x,y,tuple('abcdefg'),**kwargs)
    unit,ut=fit_rule(x,y,tuple('abcdefg'),sample_weight=np.ones(len(x)),**kwargs)
    assert old.to_dict()==new.to_dict()==unit.to_dict()
    assert ot==nt
    assert [{k:v for k,v in t.items() if k!='weighted_fidelity'} for t in ut]==nt


def test_weighted_minority_changes_executable_program_and_scale_invariance():
    x=np.r_[np.full(12,-1.),np.full(12,1.)][:,None]
    y=np.r_[np.zeros(12,int),np.zeros(9,int),np.ones(3,int)]
    weights=np.where(y==1,10.,1.)
    kwargs=dict(arithmetic=False,min_leaf=2,max_splits=2,penalty=0.)
    plain,_=fit_rule(x,y,('x',),**kwargs)
    weighted,trace=fit_rule(x,y,('x',),sample_weight=weights,**kwargs)
    scaled,st=fit_rule(x,y,('x',),sample_weight=weights*4,**kwargs)
    assert plain.predict([[-1.],[1.]]).tolist()==[0,0]
    assert weighted.predict([[-1.],[1.]]).tolist()==[0,1]
    assert scaled.to_dict()==weighted.to_dict() and st==trace
    assert trace[-1]['weighted_fidelity']==pytest.approx(42/51)
    assert trace[-1]['fidelity']==pytest.approx(15/24)


@pytest.mark.parametrize('weights',[[1], [1,0], [1,-1], [1,float('nan')], [1,float('inf')], [[1],[1]]])
def test_reject_invalid_weights(weights):
    with pytest.raises(ValueError,match='sample_weight'):
        fit_rule([[0],[1]],[0,1],('x',),sample_weight=weights)
