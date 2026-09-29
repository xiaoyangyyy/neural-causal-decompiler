import numpy as np
import pytest
import torch
from torch import nn
from ncd.cdir import Node,composed_features,discovery_feature_nodes
from ncd.active_search import search_scm
from ncd.model import Discoverer
from ncd.worlds import generate_worlds
from ncd.rules import Rule
from ncd.distributed_alignment import train_mapping,measure_mapping
from ncd.pysr_baseline import from_sympy,SymbolicScores
from ncd.traces import TorchCircuit
from ncd.evidence import fidelity_bound

def test_active_budget_structure_and_teacher_only_search():
    model=Discoverer(8);seeds=generate_worlds("refinement",4,n=32)
    rule=Rule(tuple(str(i) for i in range(14)),{"label":0})
    result=search_scm(model,rule,seeds,budget=12,population=4,feature_extractor=composed_features)
    assert result["queries"]==12
    assert not result["causal_truth_used_for_optimization"]
    originals={w.identity:w for w in seeds}
    for record in result["records"]:
        origin=originals[record["origin"]];world=record["world"]
        assert world["family"]==origin.family
        assert world["direction"]==origin.direction
        assert world["function"]==origin.function
        assert np.sign(world["coefficient"])==np.sign(origin.coefficient)

def test_composed_ir_is_executed_not_a_feature_label():
    data=np.stack([w.sample() for w in generate_worlds("extraction",4,n=32)])
    values=composed_features(data)
    assert values.shape==(4,14)
    for i,node in enumerate(discovery_feature_nodes()):
        np.testing.assert_allclose(values[:,i],[node.evaluate(d) for d in data])
    np.testing.assert_allclose(values,composed_features(data[:,::-1].copy()),atol=1e-9)

def test_distributed_mapping_positive_control():
    class KnownDecoder(nn.Module):
        def from_hidden(self,h,hs):
            z=h[:,0]*5
            return torch.stack([-z,z,torch.full_like(z,-10),torch.full_like(z,-10)],dim=1)
    rng=np.random.default_rng(12)
    h=rng.normal(size=(256,4));features=np.zeros((256,14));features[:,0]=h[:,0]
    rule=Rule(tuple(str(i) for i in range(14)),{"expr":Node("var",index=0).to_dict(),"threshold":0.,
        "left":{"label":0},"right":{"label":1}})
    model=KnownDecoder()
    q=train_mapping(model,h[:160],features[:160],rule,0,rank=1,steps=80,seed=12)
    measured=measure_mapping(model,h[160:],features[160:],rule,0,q,pair_count=256)
    assert measured["informative_pairs"]>60
    assert measured["informative_accuracy"]>.9

def test_pysr_safe_ast_conversion():
    import sympy as sp
    a,b=sp.symbols("f0 f1")
    node=from_sympy(2*a*a+sp.Abs(b),["f0","f1"])
    x=np.random.default_rng(1).normal(size=(32,2))
    np.testing.assert_allclose(node.evaluate(x),2*x[:,0]**2+abs(x[:,1]))
    with pytest.raises(ValueError):from_sympy(sp.exp(a),["f0"])
    restored=SymbolicScores.from_dict(SymbolicScores([node,Node("constant")],["a","b"]).to_dict())
    np.testing.assert_array_equal(restored.predict(x),np.zeros(32,int))

def test_trace_patching_and_cleanup():
    model=nn.Sequential(nn.Linear(2,3),nn.Tanh(),nn.Linear(3,2))
    circuit=TorchCircuit(model,["0","2"])
    base=torch.zeros(4,2);source=torch.ones(4,2)
    trace=circuit.capture(source)
    patched=circuit.intervene(base,trace,"0")
    torch.testing.assert_close(patched,model(source))
    assert all(not module._forward_hooks for module in model.modules())
    tiny=TorchCircuit(model,["0"],max_elements=1)
    with pytest.raises(ValueError):tiny.capture(source)
    assert all(not module._forward_hooks for module in model.modules())

def test_fidelity_bound_uses_independent_units():
    bound=fidelity_bound(np.ones(100))
    assert 0<bound["lower_bound"]<1
    assert bound["epsilon_upper"]>0
    with pytest.raises(ValueError):fidelity_bound([1,2])

def test_expression_search_composes_unlisted_feature_pairs():
    from ncd.rules import fit_rule
    rng=np.random.default_rng(91);x=rng.normal(size=(256,14))
    labels=(x[:,0]*x[:,7]>0).astype(int)
    rule,_=fit_rule(x,labels,[f"f{i}" for i in range(14)],max_splits=1,beam_width=2)
    assert np.mean(rule.predict(x)==labels)>.98
    assert rule.tree["expr"]["op"] in ("mul","div")
    heldout=rng.normal(size=(512,14))
    assert np.mean(rule.predict(heldout)==(heldout[:,0]*heldout[:,7]>0))>.98
