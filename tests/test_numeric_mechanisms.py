import numpy as np
import pytest
import torch
from ncd.cdir import Node
from ncd.rules import Rule
from ncd.model import Discoverer
from ncd.program_trace import ProgramExecutor
from ncd.neural_sites import SITES,SiteDecoder
from ncd.mechanism_pairs import compatible,compatible_combinations,execution_conditioned_pairs,numeric_frontier
from ncd.numeric_audit import fit_readout,audit_numeric,NumericReadout
from ncd.joint_alignment import patch_blocks

def sum_rule():
    expr=Node("add",(Node("add",(Node("var",index=0),Node("var",index=1))),Node("var",index=2)))
    return ProgramExecutor(Rule(("a","b","c"),{"expr":expr.to_dict(),"threshold":0.,"left":{"label":0},"right":{"label":1}}))

def test_neural_cuts_exactly_reconstruct_teacher_and_gradients():
    torch.manual_seed(15);teacher=Discoverer(8);teacher.train()
    x=torch.randn(9,32,2);original={k:v.clone() for k,v in teacher.state_dict().items()}
    for site in SITES:
        decoder=SiteDecoder(teacher,site);h=decoder.extract(x.numpy())
        torch.testing.assert_close(decoder.logits(h),teacher(x),atol=1e-6,rtol=1e-5)
        ht=torch.tensor(h,dtype=torch.float32,requires_grad=True)
        decoder.logits(ht).sum().backward()
        assert ht.grad is not None and torch.isfinite(ht.grad).all()
        assert teacher.training
    for key,value in teacher.state_dict().items():torch.testing.assert_close(value,original[key])

def test_compatible_masks_reject_branches_and_shadowing():
    expr=Node("var",index=0).to_dict();leaf={"label":0}
    e=ProgramExecutor(Rule(("x",),{"expr":expr,"threshold":0.,
        "left":{"expr":expr,"threshold":-1.,"left":leaf,"right":leaf},
        "right":{"expr":expr,"threshold":1.,"left":leaf,"right":leaf}}))
    a=[e.address(p) for p in ("root/expr","root/left/expr","root/right/expr")]
    assert compatible(e,a[:2]) and not compatible(e,a[1:])
    masks=compatible_combinations(e,a)
    assert masks.shape==(5,3) and not np.any(np.all(masks==[False,True,True],axis=1))
    with pytest.raises(ValueError):execution_conditioned_pairs(e,np.ones((12,1)),a,np.array([[False,True,True]]))
    e=sum_rule()
    assert not compatible(e,[e.address("root/expr"),e.address("root/expr/0/0")])

def test_disjoint_sampler_consumes_only_active_worlds_and_replays():
    e=sum_rule();a=numeric_frontier(e);x=np.random.default_rng(15).normal(size=(256,3))
    pairs,record=execution_conditioned_pairs(e,x,a,np.eye(3,dtype=bool),count=100,seed=8)
    assert record["accepted_pairs"]==100 and record["worlds_consumed"]==200
    used=[]
    for i,b in enumerate(pairs.base):
        used.append(b);used.extend(pairs.sources[i,pairs.masks[i]])
    assert len(set(used))==200
    again,rr=execution_conditioned_pairs(e,x,a,np.eye(3,dtype=bool),count=100,seed=8)
    assert record==rr
    np.testing.assert_array_equal(again.target,pairs.target)
    assert not record["neural_or_truth_filtering"]

def test_rejected_groups_never_reuse_worlds_and_empty_is_explicit():
    expr=Node("var",index=0).to_dict()
    e=ProgramExecutor(Rule(("x",),{"expr":expr,"threshold":0.,"left":{"label":0},
        "right":{"expr":expr,"threshold":1.,"left":{"label":1},"right":{"label":2}}}))
    a=[e.address("root/right/expr")]
    pairs,record=execution_conditioned_pairs(e,-np.ones((40,1)),a,np.ones((1,1),bool),count=4)
    assert len(pairs.base)==0 and record["worlds_consumed"]==40 and record["budget_exhausted"]
    assert len({i for r in record["attempts"] for i in r["world_indices"]})==40

def test_numeric_audit_distinguishes_equal_class_outputs():
    e=sum_rule();a=numeric_frontier(e);rng=np.random.default_rng(19)
    train=rng.uniform(1,2,(256,3));test=rng.uniform(1,2,(256,3))
    # Hidden coordinates implement the three numeric operands exactly.
    train_h=np.c_[train,rng.normal(size=(256,3))]
    test_h=np.c_[test,rng.normal(size=(256,3))]
    readout=fit_readout(train_h,e,train,a,ridge=1e-6)
    restored=NumericReadout.from_dict(readout.to_dict())
    np.testing.assert_allclose(restored.predict(test_h),readout.predict(test_h))
    pairs,_=execution_conditioned_pairs(e,test,a,np.eye(3,dtype=bool),count=80,seed=19)
    correct=np.eye(6)[:,:3].reshape(6,3,1);wrong=correct[:,[1,2,0],:]
    good=audit_numeric(readout,e,test,test_h,correct,pairs)
    bad=audit_numeric(readout,e,test,test_h,wrong,pairs)
    # Both interventions retain identical class 1 predictions, but the wrong
    # variable map damages numerical targets and collateral intermediates.
    for q in (correct,wrong):
        patched=patch_blocks(test_h[pairs.base],test_h[pairs.sources],q,pairs.masks)
        assert np.all(patched[:,:3].sum(1)>0) and np.all(pairs.target==1)
    assert max(r["targeted"]["nmse"] for r in good["per_node"])<1e-10
    assert min(r["targeted"]["nmse"] for r in bad["per_node"])>.5
    assert max(r["collateral"]["nmse"] for r in good["per_node"])<1e-10
    assert max(r["collateral"]["nmse"] for r in bad["per_node"])>.1


def test_cached_numeric_readout_and_audit_match_direct_paths():
    from ncd.numeric_audit import (fit_readout,fit_readout_values,numeric_values,
        intervention_numeric_targets,audit_numeric,audit_numeric_values)
    from ncd.mechanism_pairs import compatible_combinations,execution_conditioned_pairs,numeric_frontier
    e=sum_rule();rng=np.random.default_rng(99);x=rng.normal(size=(80,3));addresses=numeric_frontier(e)
    h=np.c_[x,x*x,rng.normal(size=(80,2))]
    values,visited=numeric_values(e,x,addresses)
    direct=fit_readout(h,e,x,addresses);cached=fit_readout_values(h,addresses,values,visited)
    assert direct.to_dict()==cached.to_dict()
    masks=compatible_combinations(e,addresses);pairs,_=execution_conditioned_pairs(e,x,addresses,masks,count=12,seed=9)
    targets,target_visited=intervention_numeric_targets(e,x,addresses,pairs)
    q=np.linalg.qr(rng.normal(size=(h.shape[1],len(addresses))))[0].reshape(h.shape[1],len(addresses),1)
    assert audit_numeric(direct,e,x,h,q,pairs)==audit_numeric_values(cached,h,q,pairs,values,visited,targets,target_visited)

