import numpy as np
import torch
from torch import nn
from ncd.cdir import Node
from ncd.rules import Rule
from ncd.program_trace import ProgramExecutor
from ncd.numeric_audit import fit_readout,audit_numeric
from ncd.numeric_mapping import train_numeric_mapping
from ncd.mechanism_pairs import execution_conditioned_pairs,numeric_frontier

def test_numeric_mapping_generalizes_combination_without_changing_teacher_or_probe():
    class SumDecoder(nn.Module):
        def __init__(self):
            super().__init__();self.gain=nn.Parameter(torch.tensor(4.))
        def from_hidden(self,h,hs):
            z=(h[:,0]+h[:,1])*self.gain
            return torch.stack([-z,z,z*0-20,z*0-20],1)
    rng=np.random.default_rng(291);h=rng.normal(size=(1024,4));x=h[:,:2]
    expr=Node("add",(Node("var",index=0),Node("var",index=1)))
    executor=ProgramExecutor(Rule(("a","b"),{"expr":expr.to_dict(),"threshold":0.,
        "left":{"label":0},"right":{"label":1}}))
    addresses=numeric_frontier(executor)
    probe=fit_readout(h[:512],executor,x[:512],addresses,ridge=1e-6);saved=probe.to_dict()
    decoder=SumDecoder();gain=decoder.gain.detach().clone()
    q,record=train_numeric_mapping(decoder,h[:512],x[:512],executor,addresses,
        np.eye(2,dtype=bool),probe,steps=100,seed=291,numeric_weight=1.)
    pairs,pairing=execution_conditioned_pairs(executor,x[512:],addresses,
        np.array([[True,True]]),count=128,seed=392)
    assert pairing["accepted_pairs"]==128
    metrics=audit_numeric(probe,executor,x[512:],h[512:],q,pairs)
    assert max(v["targeted"]["nmse"] for v in metrics["per_node"])<.1
    torch.testing.assert_close(decoder.gain,gain)
    assert decoder.gain.grad is None and decoder.gain.requires_grad and decoder.training
    assert probe.to_dict()==saved and record["optimizer_updates"]==100
    assert record["training_combinations"]==[[True,False],[False,True]]


def test_fixed_pair_numeric_mapping_preserves_positive_control():
    from ncd.fixed_numeric_mapping import train_fixed_numeric_mapping
    from ncd.numeric_audit import numeric_values,intervention_numeric_targets,audit_numeric_values
    class SumDecoder(nn.Module):
        def __init__(self):super().__init__();self.gain=nn.Parameter(torch.tensor(4.))
        def from_hidden(self,h,hs):
            z=(h[:,0]+h[:,1])*self.gain;return torch.stack([-z,z,z*0-20,z*0-20],1)
    rng=np.random.default_rng(792);h=rng.normal(size=(1024,4));x=h[:,:2]
    expr=Node("add",(Node("var",index=0),Node("var",index=1)))
    executor=ProgramExecutor(Rule(("a","b"),{"expr":expr.to_dict(),"threshold":0.,"left":{"label":0},"right":{"label":1}}))
    addresses=numeric_frontier(executor);probe=fit_readout(h[:512],executor,x[:512],addresses,ridge=1e-6)
    train,_=execution_conditioned_pairs(executor,x[:512],addresses,np.eye(2,dtype=bool),count=128,seed=19)
    target,visit=intervention_numeric_targets(executor,x[:512],addresses,train);decoder=SumDecoder();saved=decoder.gain.detach().clone()
    q,record=train_fixed_numeric_mapping(decoder,h[:512],train,target,visit,probe,steps=100,seed=792,numeric_weight=1.)
    test,_=execution_conditioned_pairs(executor,x[512:],addresses,np.array([[True,True]]),count=128,seed=20)
    tt,tv=intervention_numeric_targets(executor,x[512:],addresses,test);natural,nv=numeric_values(executor,x[512:],addresses)
    result=audit_numeric_values(probe,h[512:],q,test,natural,nv,tt,tv)
    assert max(v['targeted']['nmse'] for v in result['per_node'])<.1
    torch.testing.assert_close(decoder.gain,saved);assert record['optimizer_updates']==100


def test_oblique_patch_recovers_entangled_coordinates_that_orthogonal_blocks_mix():
    from ncd.oblique_mapping import biorthogonal_bases,patch_oblique,validate_oblique
    rng=np.random.default_rng(1091);z=rng.normal(size=(300,2))
    mixing=np.array([[1.,.85],[0.,.5267826876]])
    h=z@mixing.T
    read,write=biorthogonal_bases(mixing,2,1)
    diagnostics=validate_oblique(read,write)
    base=np.arange(80);sources=np.column_stack([np.arange(80,160),np.arange(160,240)])
    masks=np.tile([[True,False],[False,True],[True,True],[True,False]],(20,1))
    patched=patch_oblique(h[base],h[sources],read,write,masks)
    expected=z[base].copy()
    for j in range(2):expected[masks[:,j],j]=z[sources[masks[:,j],j],j]
    np.testing.assert_allclose(patched,expected@mixing.T,atol=1e-9)
    assert diagnostics['max_write_cosine']>.8
    q=np.linalg.qr(mixing)[0].reshape(2,2,1)
    from ncd.joint_alignment import patch_blocks
    orthogonal=patch_blocks(h[base],h[sources],q,masks)
    assert np.mean((orthogonal-expected@mixing.T)**2)>.05


def test_fixed_oblique_mapping_is_biorthogonal_and_freezes_teacher():
    from ncd.oblique_mapping import train_fixed_oblique_mapping,validate_oblique
    from ncd.numeric_audit import numeric_values,intervention_numeric_targets,audit_oblique_numeric_values
    class SumDecoder(nn.Module):
        def __init__(self):super().__init__();self.gain=nn.Parameter(torch.tensor(3.))
        def from_hidden(self,h,hs):
            z=(h[:,0]+h[:,1])*self.gain;return torch.stack([-z,z,z*0-20,z*0-20],1)
    rng=np.random.default_rng(1092);z=rng.normal(size=(900,2));mix=np.array([[1.,.8],[0.,.6]])
    core=z@mix.T;h=np.c_[core,rng.normal(size=(900,2))];x=z
    expr=Node('add',(Node('var',index=0),Node('var',index=1)))
    executor=ProgramExecutor(Rule(('a','b'),{'expr':expr.to_dict(),'threshold':0.,'left':{'label':0},'right':{'label':1}}))
    addresses=numeric_frontier(executor);probe=fit_readout(h[:500],executor,x[:500],addresses,ridge=1e-6)
    train,_=execution_conditioned_pairs(executor,x[:500],addresses,np.eye(2,dtype=bool),count=160,seed=31)
    target,visit=intervention_numeric_targets(executor,x[:500],addresses,train);decoder=SumDecoder();saved=decoder.gain.detach().clone()
    read,write,record=train_fixed_oblique_mapping(decoder,h[:500],train,target,visit,probe,steps=40,seed=1092,numeric_weight=1.)
    validate_oblique(read,write);test,_=execution_conditioned_pairs(executor,x[500:],addresses,np.array([[True,True]]),count=100,seed=32)
    tt,tv=intervention_numeric_targets(executor,x[500:],addresses,test);natural,nv=numeric_values(executor,x[500:],addresses)
    result=audit_oblique_numeric_values(probe,h[500:],read,write,test,natural,nv,tt,tv)
    assert max(v['targeted']['nmse'] for v in result['per_node'])<.05
    assert record['diagnostics']['biorthogonality_max_error']<2e-5
    torch.testing.assert_close(decoder.gain,saved);assert decoder.training and decoder.gain.requires_grad
