import numpy as np
import pytest
from ncd.cdir import Node,discovery_feature_nodes
from ncd.rules import Rule
from ncd.program_trace import ProgramExecutor

def example():
    return Rule(("a","b"),{"expr":Node("add",(Node("var",index=0),Node("var",index=1))).to_dict(),
        "threshold":0.,"left":{"label":0},"right":{"expr":Node("var",index=1).to_dict(),
        "threshold":0.,"left":{"label":1},"right":{"label":2}}})

def test_occurrences_differ_even_for_identical_subtrees():
    a=Node("var",index=0);p=Node("sub",(a,a));e=ProgramExecutor(p)
    x=np.arange(4.).reshape(-1,1)
    np.testing.assert_array_equal(e.execute(x).output,0)
    trace=e.execute(x,{e.address("root/0"):np.ones(4)*10})
    np.testing.assert_array_equal(trace.output,10-x[:,0])
    assert e.address("root/0")!=e.address("root/1")
    assert ProgramExecutor(Node.from_dict(p.to_dict())).program_id==e.program_id

def test_all_discovery_ast_values_match_original():
    x=np.random.default_rng(19).normal(size=(48,2))
    for n in discovery_feature_nodes():
        np.testing.assert_allclose(ProgramExecutor(n).execute(x).output,n.evaluate(x),rtol=1e-10,atol=1e-10)

def test_rule_paths_predicates_and_numeric_interventions():
    p=example();e=ProgramExecutor(p);x=np.array([[-2.,1.],[2.,-1.],[2.,1.]])
    t=e.execute(x)
    np.testing.assert_array_equal(t.output,p.predict(x))
    np.testing.assert_array_equal(t.records[e.address("root/right/predicate")]["visited"],[False,True,True])
    patched=e.execute(x,{e.address("root/predicate"):np.zeros(3,bool)})
    np.testing.assert_array_equal(patched.output,[2,1,2])
    numeric=e.execute(x,{e.address("root/expr/0"):np.ones(3)*-10})
    np.testing.assert_array_equal(numeric.output,[0,0,0])
    restored=ProgramExecutor(Rule.from_dict(p.to_dict())).execute(x)
    assert restored.to_dict()==t.to_dict()

def test_reject_invalid_or_ambiguous_interventions():
    e=ProgramExecutor(example());x=np.zeros((3,2));address=e.address("root/predicate")
    for value in ([1,0,1],True,[True,False]):
        with pytest.raises(ValueError):e.execute(x,{address:value})
    with pytest.raises(ValueError):e.execute(x,{"foreign:root":0})
    with pytest.raises(ValueError):e.execute(x,{e.address("root/expr"):np.zeros(3),e.address("root/expr/0"):np.ones(3)})
    with pytest.raises(ValueError):e.execute(x,{e.address("root/left"):np.ones(3)})
    with pytest.raises(ValueError):e.execute(x,{e.address("root/expr"):np.full(3,np.nan)})

def test_conditional_records_only_executed_branch():
    condition=Node("lt",(Node("mean",(Node("var",index=0),)),Node("constant")))
    p=Node("if",(condition,Node("constant",value=2),Node("constant",value=3)))
    e=ProgramExecutor(p);x=np.ones((4,1))
    t=e.execute(x);assert t.output==3 and e.address("root/1") not in t.records
    t=e.execute(x,{e.address("root/0"):True})
    assert t.output==2 and e.address("root/2") not in t.records
    unused=e.execute(x,{e.address("root/1"):9.})
    assert unused.output==3 and not unused.records[e.address("root/1")]["visited"]

def test_matrix_interventions_use_actual_branch_shape():
    x=np.arange(12.).reshape(4,3)
    matrix=Node("columns",(Node("var",index=0),Node("var",index=1)))
    other=Node("columns",(Node("var",index=2),))
    condition=Node("lt",(Node("constant"),Node("constant",value=1.)))
    e=ProgramExecutor(Node("if",(condition,matrix,other)))
    out=e.execute(x,{e.address("root"):np.ones((4,2))})
    np.testing.assert_array_equal(out.output,np.ones((4,2)))
    unused=e.execute(x,{e.address("root/2"):np.zeros((4,1))})
    np.testing.assert_array_equal(unused.output,x[:,:2])
    assert not unused.records[e.address("root/2")]["visited"]
    with pytest.raises(ValueError):e.execute(x,{e.address("root"):np.ones((4,3))})
