from pathlib import Path
Path('ncd/fixed_numeric_mapping.py').write_text('''"""Numerical mapping training on a frozen, precomputed intervention pair pool."""
import numpy as np
import torch
from torch import nn
from .joint_alignment import frozen_teacher,patch_blocks

def train_fixed_numeric_mapping(decoder,h,pairs,targets,visited,readout,*,rank=1,steps=120,batch_size=128,seed=791,numeric_weight=.5,shuffle_targets=False):
    h=np.asarray(h,dtype=float);targets=np.asarray(targets,dtype=float);visited=np.asarray(visited,dtype=bool)
    d=h.shape[1];k=len(readout.addresses);count=len(pairs.base)
    if h.ndim!=2 or targets.shape!=visited.shape or targets.shape!=(count,k) or pairs.masks.shape!=(count,k):raise ValueError("Fixed numeric pool mismatch")
    if not count or type(rank) is not int or rank<1 or k*rank>d or steps<1 or batch_size<1 or numeric_weight<0 or not np.isfinite(targets).all():raise ValueError("Invalid fixed mapping budget")
    generator=torch.Generator().manual_seed(seed);rng=np.random.default_rng(seed)
    parameter=nn.Parameter(torch.randn(d,k*rank,generator=generator)*.1);optimizer=torch.optim.Adam([parameter],lr=.025)
    ht=torch.tensor(h,dtype=torch.float32);mean=torch.tensor(readout.mean,dtype=torch.float32);scale=torch.tensor(readout.scale,dtype=torch.float32)
    beta=torch.tensor(readout.coefficients,dtype=torch.float32);variance=torch.tensor(readout.target_variance,dtype=torch.float32)
    target_order=rng.permutation(count) if shuffle_targets else np.arange(count);history=[]
    with frozen_teacher(decoder):
        for step in range(steps):
            order=rng.permutation(count);ids=order[:min(batch_size,count)];tids=target_order[ids]
            q=torch.linalg.qr(parameter,mode="reduced").Q.reshape(d,k,rank)
            patched=patch_blocks(ht[pairs.base[ids]],ht[pairs.sources[ids]],q,torch.tensor(pairs.masks[ids],dtype=torch.float32))
            width=d//2;logits=decoder.from_hidden(patched[:,:width],patched[:,width:])
            prediction=torch.cat([(patched-mean)/scale,torch.ones(len(ids),1)],1)@beta
            target=torch.tensor(targets[tids],dtype=torch.float32);visit=torch.tensor(visited[tids])
            error=(prediction-target)/torch.sqrt(variance)
            numeric=nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction="none")[visit].mean()
            behavioral=nn.functional.cross_entropy(logits,torch.tensor(pairs.target[tids]))
            loss=behavioral+numeric_weight*numeric;optimizer.zero_grad();loss.backward();optimizer.step()
            history.append({"step":step,"batch":ids.tolist(),"target_indices":tids.tolist(),"loss":float(loss.detach()),
                "behavioral_loss":float(behavioral.detach()),"numeric_loss":float(numeric.detach())})
    basis=torch.linalg.qr(parameter.detach(),mode="reduced").Q.numpy().astype(float).reshape(d,k,rank)
    return basis,{"seed":seed,"rank":rank,"steps":steps,"batch_size":batch_size,"numeric_weight":numeric_weight,
        "shuffle_targets":shuffle_targets,"teacher_frozen":True,"readout_frozen":True,"addresses":list(readout.addresses),
        "pair_pool_size":count,"training_combinations":np.unique(pairs.masks,axis=0).tolist(),"optimizer_updates":steps,"history":history}
''',encoding='utf-8')
p=Path('tests/test_numeric_mapping.py');s=p.read_text(encoding='utf-8');s+='''

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
''';p.write_text(s,encoding='utf-8')
