import numpy as np
import pytest
from ncd.dsl import Expr,Program
from ncd.statistics import FEATURES
from ncd.synthesis import synthesize
from ncd.counterexamples import refine,compare_search
from ncd.alignment import interchange
from ncd.metrics import evaluate

def test_ast_roundtrip_and_rejection():
    e=Expr("sub",(Expr("feature",feature="resdep_xy"),Expr("feature",feature="resdep_yx")))
    p=Program(expr=e,threshold=0,left=Program(label=0),right=Program(label=1))
    restored=Program.from_dict(p.to_dict())
    x=np.random.default_rng(1).normal(size=(32,len(FEATURES)))
    np.testing.assert_array_equal(restored.predict(x),p.predict(x))
    assert restored.complexity==7
    with pytest.raises(ValueError): Expr("exec")
    with pytest.raises(ValueError): Expr("feature",feature="label")
    with pytest.raises(ValueError): Program(label=4)
    with pytest.raises(ValueError): Program(expr=e,threshold=float("nan"),left=Program(label=0),right=Program(label=1))

def test_search_recovers_known_rule_and_penalty():
    x=np.zeros((120,len(FEATURES)))
    x[:,FEATURES.index("var_log_ratio")]=np.linspace(-1,1,120)
    truth=(x[:,2]>=0).astype(int)
    p,_=synthesize(x,truth,max_splits=2,beam_width=2,min_leaf=4)
    assert np.mean(p.predict(x)==truth)>.95
    simple,_=synthesize(x,truth,penalty=1,max_splits=2,beam_width=2,min_leaf=4)
    assert simple.label is not None
    q,_=refine(Program(label=0),x[:40],truth[:40],x[40:],truth[40:],rounds=1,per_round=16,max_splits=2,beam_width=2)
    assert np.mean(q.predict(x)==truth)>.5

def test_query_budget_and_truth_error_separation():
    x=np.zeros((40,len(FEATURES)))
    truth=np.array([0]*20+[1]*20)
    calls=[]
    def query(ids):
        calls.extend(ids)
        return np.tile([1.,0.,0.,0.],(len(ids),1))
    result=compare_search(Program(label=0),x,truth,query,budget=8)
    assert len(calls)==16
    assert result["strategies"]["program_guided"]["joint_errors"]==8
    assert result["strategies"]["program_guided"]["fidelity_errors"]==0

def test_interchange_decoder_and_identity():
    rng=np.random.default_rng(1)
    z=rng.normal(size=(12,8)); beta=rng.normal(size=8)
    target=rng.normal(size=12)
    np.testing.assert_allclose(interchange(z,beta,target)@beta,target,atol=1e-10)
    np.testing.assert_allclose(interchange(z,beta,z@beta),z)
    with pytest.raises(ValueError): interchange(z,np.zeros(8),target)

def test_metrics_partition():
    truth=np.array([0,0,0,0,0])
    neural=np.eye(4)[[0,1,0,1,1]]
    program=np.array([0,1,1,0,2])
    m=evaluate(truth,neural,program)
    assert sum(m[k] for k in ("joint_correct","joint_same_wrong","neural_only_correct","program_only_correct","both_wrong_different"))==5
    assert all(m[k]==1 for k in ("joint_correct","joint_same_wrong","neural_only_correct","program_only_correct","both_wrong_different"))
