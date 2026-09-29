"""Joint numerical and behavioral mapping fit; the teacher and probe stay frozen."""
import numpy as np
import torch
from torch import nn
from .joint_alignment import frozen_teacher,patch_blocks
from .mechanism_pairs import execution_conditioned_pairs
from .numeric_audit import intervention_numeric_targets

def train_numeric_mapping(decoder,h,features,executor,addresses,masks,readout,*,rank=1,steps=120,seed=291,numeric_weight=.5,shuffle_targets=False):
    h=np.asarray(h);d=h.shape[1];k=len(addresses)
    if tuple(addresses)!=readout.addresses or len(h)!=len(features):raise ValueError("Readout schema mismatch")
    if type(rank) is not int or rank<1 or k*rank>d or steps<1 or not np.isfinite(numeric_weight) or numeric_weight<0:
        raise ValueError("Invalid numeric mapping budget")
    generator=torch.Generator().manual_seed(seed);rng=np.random.default_rng(seed)
    parameter=nn.Parameter(torch.randn(d,k*rank,generator=generator)*.1)
    optimizer=torch.optim.Adam([parameter],lr=.025)
    ht=torch.tensor(h,dtype=torch.float32)
    mean=torch.tensor(readout.mean,dtype=torch.float32);scale=torch.tensor(readout.scale,dtype=torch.float32)
    beta=torch.tensor(readout.coefficients,dtype=torch.float32)
    variance=torch.tensor(readout.target_variance,dtype=torch.float32)
    history=[]
    with frozen_teacher(decoder):
        for step in range(steps):
            pairs,record=execution_conditioned_pairs(executor,features,addresses,masks,count=128,seed=seed+step)
            if not len(pairs.base):
                history.append({"step":step,"accepted":0,"loss":None,"pairing":record});continue
            targets,visited=intervention_numeric_targets(executor,features,addresses,pairs)
            label=pairs.target.copy()
            if shuffle_targets:
                permutation=rng.permutation(len(label));label=label[permutation]
                targets=targets[permutation];visited=visited[permutation]
            target=torch.tensor(targets,dtype=torch.float32);visit=torch.tensor(visited)
            q=torch.linalg.qr(parameter,mode="reduced").Q.reshape(d,k,rank)
            patched=patch_blocks(ht[pairs.base],ht[pairs.sources],q,torch.tensor(pairs.masks,dtype=torch.float32))
            width=d//2;logits=decoder.from_hidden(patched[:,:width],patched[:,width:])
            prediction=torch.cat([(patched-mean)/scale,torch.ones(len(patched),1)],1)@beta
            # Scale comes only from the fit probe; all executed numeric nodes
            # contribute, including collateral nodes not directly intervened on.
            normalized_error=(prediction-target)/torch.sqrt(variance)
            numeric=nn.functional.smooth_l1_loss(normalized_error,torch.zeros_like(normalized_error),reduction="none")[visit].mean()
            behavioral=nn.functional.cross_entropy(logits,torch.tensor(label))
            loss=behavioral+numeric_weight*numeric
            optimizer.zero_grad();loss.backward();optimizer.step()
            history.append({"step":step,"accepted":len(label),"loss":float(loss.detach()),
                "behavioral_loss":float(behavioral.detach()),"numeric_loss":float(numeric.detach()),"pairing":record})
    basis=torch.linalg.qr(parameter.detach(),mode="reduced").Q.numpy().astype(float).reshape(d,k,rank)
    return basis,{"seed":seed,"rank":rank,"steps":steps,"numeric_weight":numeric_weight,
                  "shuffle_targets":shuffle_targets,"teacher_frozen":True,"readout_frozen":True,
                  "addresses":list(addresses),"training_combinations":np.asarray(masks).tolist(),
                  "optimizer_updates":sum(r["accepted"]>0 for r in history),"history":history}
