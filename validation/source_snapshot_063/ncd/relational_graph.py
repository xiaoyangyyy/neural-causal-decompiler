"""Controlled edge-incidence attention for permutation-equivariant graph discovery.

Enabled/disabled variants have identical parameters and training procedure.
This is a new teacher architecture, not a recovery claim about older networks.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from .graph_model import GraphDiscoverer,GRAPH_FEATURES,graph_probabilities
from .model import set_seed
from .io import save_json

RELATIONS=("same_edge","reverse_edge","shared_source","shared_target","successor","predecessor")

def edge_relations(nodes,device=None):
    if type(nodes) is not int or nodes<2:raise ValueError("At least two nodes required")
    i=torch.arange(nodes,device=device).repeat_interleave(nodes)
    j=torch.arange(nodes,device=device).repeat(nodes)
    source=i[:,None];target=j[:,None];other_source=i[None,:];other_target=j[None,:]
    return torch.stack(((source==other_source)&(target==other_target),
        (source==other_target)&(target==other_source),source==other_source,target==other_target,
        target==other_source,source==other_target))

class RelationalGraphDiscoverer(GraphDiscoverer):
    def __init__(self,width=48,relations=True):
        if width<4 or width%4:raise ValueError("Width must be divisible by four")
        super().__init__(width)
        self.relations_enabled=bool(relations)
        self.relation_bias=nn.Parameter(torch.zeros(4,len(RELATIONS)))

    def representation(self,x):
        if x.ndim!=4 or x.shape[1]!=x.shape[2] or x.shape[-1]!=len(GRAPH_FEATURES) or x.shape[1]<2:
            raise ValueError("Expected B x N x N x feature schema")
        b,n,_,_=x.shape
        h=self.encoder(((x-self.mean)/self.std).clamp(-20,20)).reshape(b,n*n,self.width)
        relations=edge_relations(n,x.device).to(h.dtype)
        bias=torch.einsum("rxy,hr->hxy",relations,self.relation_bias)*float(self.relations_enabled)
        key_mask=torch.eye(n,dtype=torch.bool,device=x.device).reshape(-1)
        bias=bias.masked_fill(key_mask[None,None,:],float("-inf"))
        mask=bias.unsqueeze(0).expand(b,-1,-1,-1).reshape(b*4,n*n,n*n)
        context,_=self.attention(h,h,h,attn_mask=mask,need_weights=False)
        return self.norm(h+context).reshape(b,n,n,self.width)

def train_relational_graph(groups,dev_groups,directory,*,relations=True,epochs=40,width=48,seed=493):
    if epochs<1:raise ValueError("Positive epoch budget required")
    set_seed(seed);model=RelationalGraphDiscoverer(width,relations)
    flat=np.concatenate([x[:,~np.eye(x.shape[1],dtype=bool)].reshape(-1,len(GRAPH_FEATURES)) for x,y in groups])
    model.mean.copy_(torch.tensor(flat.mean(0),dtype=torch.float32))
    model.std.copy_(torch.tensor(flat.std(0).clip(.03),dtype=torch.float32))
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
                logits=model(torch.as_tensor(x[ids],dtype=torch.float32))
                truth=torch.as_tensor(y[ids],dtype=torch.long)
                optimizer.zero_grad();loss=loss_fn(logits[:,mask].reshape(-1,4),truth[:,mask].reshape(-1))
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        losses=[]
        for x,y in dev_groups:
            p=graph_probabilities(model,x);mask=~np.eye(x.shape[1],dtype=bool)
            losses.append(float(-np.log(np.take_along_axis(p[:,mask],y[:,mask,None],axis=-1).clip(1e-9)).mean()))
        val=float(np.mean(losses));history.append({"epoch":epoch+1,"dev_cross_entropy":val})
        if val<best:best=val;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch==0 or (epoch+1)%10==0:print(f"relations={relations} epoch {epoch+1}/{epochs}: dev loss {val:.4f}",flush=True)
    model.load_state_dict(state);model.eval();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    torch.save({"width":width,"relations":relations,"state_dict":state,"architecture":"edge_incidence_bias_v1"},directory/"graph_teacher.pt")
    save_json(directory/"training.json",{"seed":seed,"history":history,"best_epoch":best_epoch,
        "relations":bool(relations),"relation_schema":list(RELATIONS),"frozen":True,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "relation_bias":model.relation_bias.detach().numpy().tolist(),
        "selection":"minimum unweighted dev cross-entropy; test labels not used"})
    return model

def load_relational_graph(path):
    d=torch.load(path,map_location="cpu",weights_only=True)
    if d["architecture"]!="edge_incidence_bias_v1":raise ValueError("Unknown relational architecture")
    model=RelationalGraphDiscoverer(d["width"],d["relations"]);model.load_state_dict(d["state_dict"]);model.eval()
    return model
