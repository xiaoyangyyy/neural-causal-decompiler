"""Permutation-equivariant graph teacher with explicit node contexts."""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from .graph_model import GraphDiscoverer,GRAPH_FEATURES,graph_probabilities
from .model import set_seed
from .io import save_json


class NodeContextGraphDiscoverer(GraphDiscoverer):
    def __init__(self,width=48):
        super().__init__(width)
        self.context=nn.Sequential(nn.Linear(6*width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())

    def representation(self,x):
        if x.ndim!=4 or x.shape[1]!=x.shape[2] or x.shape[-1]!=self.mean.numel() or x.shape[1]<2:
            raise ValueError("Expected B x N x N graph features")
        b,n,_,_=x.shape
        h=self.encoder(((x-self.mean)/self.std).clamp(-20,20))
        valid=(~torch.eye(n,dtype=torch.bool,device=x.device)).to(h.dtype)
        outgoing=(h*valid[None,:,:,None]).sum(2)/(n-1)
        incoming=(h*valid[None,:,:,None]).sum(1)/(n-1)
        def source(v):return v[:,:,None,:].expand(-1,-1,n,-1)
        def target(v):return v[:,None,:,:].expand(-1,n,-1,-1)
        global_mean=(h*valid[None,:,:,None]).sum((1,2))/(n*(n-1))
        global_edges=global_mean[:,None,None,:].expand(-1,n,n,-1)
        enriched=h+self.context(torch.cat((h,source(outgoing),source(incoming),
                                            target(outgoing),target(incoming),global_edges),-1))
        flat=enriched.reshape(b,n*n,self.width)
        key_mask=torch.eye(n,dtype=torch.bool,device=x.device).reshape(1,-1).expand(b,-1)
        attended,_=self.attention(flat,flat,flat,key_padding_mask=key_mask,need_weights=False)
        return self.norm(flat+attended).reshape(b,n,n,self.width)


def train_node_context_graph(groups,dev_groups,directory,*,epochs=40,width=48,seed=3993):
    set_seed(seed);model=NodeContextGraphDiscoverer(width)
    flat=np.concatenate([x[:,~np.eye(x.shape[1],dtype=bool)].reshape(-1,len(GRAPH_FEATURES)) for x,y in groups])
    model.mean.copy_(torch.tensor(flat.mean(0),dtype=torch.float32));model.std.copy_(torch.tensor(flat.std(0).clip(.03),dtype=torch.float32))
    counts=np.zeros(4)
    for x,y in groups:counts+=np.bincount(y[:,~np.eye(y.shape[1],dtype=bool)].ravel(),minlength=4)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(4*np.maximum(counts,1)),dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    history=[];best=float("inf");state=None;best_epoch=0
    for epoch in range(epochs):
        model.train()
        for x,y in groups:
            order=torch.randperm(len(x)).numpy();mask=~torch.eye(x.shape[1],dtype=torch.bool)
            for ids in np.array_split(order,max(1,int(np.ceil(len(order)/24)))):
                logits=model(torch.as_tensor(x[ids],dtype=torch.float32));truth=torch.as_tensor(y[ids],dtype=torch.long)
                optimizer.zero_grad();loss=loss_fn(logits[:,mask].reshape(-1,4),truth[:,mask].reshape(-1))
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        losses=[]
        for x,y in dev_groups:
            p=graph_probabilities(model,x);mask=~np.eye(x.shape[1],dtype=bool)
            losses.append(float(-np.log(np.take_along_axis(p[:,mask],y[:,mask,None],axis=-1).clip(1e-9)).mean()))
        val=float(np.mean(losses));history.append({"epoch":epoch+1,"dev_cross_entropy":val})
        if val<best:best=val;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch==0 or (epoch+1)%10==0:print(f"node-context epoch {epoch+1}/{epochs}: dev loss {val:.4f}",flush=True)
    model.load_state_dict(state);model.eval();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    torch.save({"width":width,"state_dict":state,"architecture":"node_context_v1"},directory/"graph_teacher.pt")
    save_json(directory/"training.json",{"seed":seed,"history":history,"best_epoch":best_epoch,
        "architecture":"node_context_v1","parameter_count":sum(p.numel() for p in model.parameters()),
        "selection":"minimum unweighted dev cross-entropy; test labels not used"})
    return model


def load_node_context_graph(path):
    d=torch.load(path,map_location="cpu",weights_only=True)
    if d["architecture"]!="node_context_v1":raise ValueError("Unknown node-context architecture")
    model=NodeContextGraphDiscoverer(d["width"]);model.load_state_dict(d["state_dict"]);model.eval();return model

