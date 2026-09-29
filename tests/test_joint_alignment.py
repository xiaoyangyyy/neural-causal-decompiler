import numpy as np
import pytest
import torch
from torch import nn
from ncd.cdir import Node
from ncd.rules import Rule
from ncd.program_trace import ProgramExecutor
from ncd.joint_alignment import make_pairs,patch_blocks,train_joint_mapping,measure_joint_mapping

def setup():
    # Two independent intermediate sums, and a four-way decision.
    expr=Node("var",index=1).to_dict()
    rule=Rule(("a","b"),{"expr":Node("var",index=0).to_dict(),"threshold":0.,
        "left":{"expr":expr,"threshold":0.,"left":{"label":0},"right":{"label":1}},
        "right":{"expr":expr,"threshold":0.,"left":{"label":2},"right":{"label":3}}})
    executor=ProgramExecutor(rule)
    # A shared high-level b variable has two occurrence sites; use root a and
    # left b here, exercising a conditional intermediate and its coverage.
    addresses=[executor.address("root/expr"),executor.address("root/left/expr")]
    return executor,addresses

class KnownDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(6.))
    def from_hidden(self,h,hs):
        a,b=h[:,0]*self.scale,h[:,1]*self.scale
        return torch.stack([-a-b,-a+b,a-b,a+b],1)

def test_joint_patch_orthogonal_blocks_and_distinct_sources():
    b=np.zeros((2,4));s=np.arange(16.).reshape(2,2,4)
    q=np.eye(4)[:,:2].reshape(4,2,1);mask=np.array([[1,0],[1,1]],bool)
    out=patch_blocks(b,s,q,mask)
    np.testing.assert_array_equal(out[:,0],s[:,0,0])
    np.testing.assert_array_equal(out[:,1],[0,s[1,1,1]])
    np.testing.assert_array_equal(out[:,2:],0)

def test_pairs_disjoint_and_intermediate_target_replay():
    e,a=setup();x=np.random.default_rng(4).normal(size=(96,2))
    combos=np.array([[1,0],[0,1],[1,1]],bool)
    pairs=make_pairs(e,x,a,combos,count=24,disjoint=True)
    assert len(np.unique(np.c_[pairs.base,pairs.sources]))==72
    source=e.execute(x)
    for i in range(len(pairs.base)):
        patches={v:np.array([source.records[v]["value"][pairs.sources[i,j]]]) for j,v in enumerate(a) if pairs.masks[i,j]}
        assert e.execute(x[pairs.base[i:i+1]],patches).output[0]==pairs.target[i]
    with pytest.raises(ValueError):make_pairs(e,x,a,[[1,0]],count=5)
    with pytest.raises(ValueError):make_pairs(e,x,[a[0],a[0]],combos)

def test_joint_learning_preserves_teacher_and_rejects_rank():
    # Use a rule depending only on two numeric intermediates within one sum,
    # so both interventions correspond exactly to distinct neural coordinates.
    expr=Node("add",(Node("var",index=0),Node("var",index=1)))
    e=ProgramExecutor(Rule(("a","b"),{"expr":expr.to_dict(),"threshold":0.,
        "left":{"label":0},"right":{"label":1}}))
    addresses=[e.address("root/expr/0"),e.address("root/expr/1")]
    class SumDecoder(nn.Module):
        def __init__(self):
            super().__init__();self.scale=nn.Parameter(torch.tensor(6.))
        def from_hidden(self,h,hs):
            z=(h[:,0]+h[:,1])*self.scale
            return torch.stack([-z,z,torch.full_like(z,-20),torch.full_like(z,-20)],1)
    rng=np.random.default_rng(8);h=rng.normal(size=(512,4));x=h[:,:2]
    model=SumDecoder();original=model.scale.detach().clone()
    combos=np.array([[1,0],[0,1]],bool)
    q,record=train_joint_mapping(model,h[:320],x[:320],e,addresses,combos,steps=100,seed=8)
    assert model.training and model.scale.requires_grad
    torch.testing.assert_close(original,model.scale)
    assert model.scale.grad is None
    np.testing.assert_allclose(q.reshape(4,2).T@q.reshape(4,2),np.eye(2),atol=1e-5)
    # A held-out combination: never trained both interventions together.
    pairs=make_pairs(e,x[320:],addresses,np.array([[1,1]],bool),count=256,seed=18)
    metrics=measure_joint_mapping(model,h[320:],q,pairs)
    assert metrics["overall"]["informative_pairs"]>70
    assert metrics["overall"]["informative_accuracy"]>.85
    assert record["training_combinations"]==[[True,False],[False,True]]
    with pytest.raises(ValueError):train_joint_mapping(model,h,x,e,addresses,combos,rank=3,steps=1)
    with pytest.raises(ValueError):measure_joint_mapping(model,h[320:],np.ones((4,2,1)),pairs)

def test_predicate_shadow_is_rejected_and_joint_source_coverage():
    e,a=setup();x=np.array([[-1.,-2.],[-1.,2.],[1.,-2.],[1.,2.]])
    with pytest.raises(ValueError):
        e.execute(x,{e.address("root/predicate"):np.ones(4,bool),
                     e.address("root/expr"):np.zeros(4)})
    pairs=make_pairs(e,x,a,np.array([[False,True]],bool),count=128,seed=7)
    np.testing.assert_array_equal(pairs.source_visited[:,1],x[pairs.sources[:,1],0]<0)
    h=np.c_[x,np.zeros_like(x)]
    q=np.eye(4)[:,:2].reshape(4,2,1)
    measured=measure_joint_mapping(KnownDecoder(),h,q,pairs)
    expected=(x[pairs.base,0]<0)&(x[pairs.sources[:,1],0]<0)
    assert measured["overall"]["all_active_intermediates_executed"]==int(expected.sum())
