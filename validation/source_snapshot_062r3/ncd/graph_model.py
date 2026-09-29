"""Amortized graph inference with permutation-equivariant pair-token attention.

The fixed statistical front end is explicit. Learned attention operates on local
edge/conditioning summaries; it does not receive true graphs or equation metadata.
"""
from itertools import combinations
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from .statistics import FEATURES,extract_one
from .graphs import partial_correlation,fisher_p,acyclic_projection,topological_order
from .model import set_seed
from .io import save_json

GRAPH_FEATURES=FEATURES+("min_abs_partial_1","min_abs_partial_2","max_ci_p","ci_fraction",
                         "mean_other_corr_x","mean_other_corr_y")
SWAP_FEATURES=[0,1,2,4,3,6,5,7,9,8,11,10,13,12,14,15,16,17,19,18]
SWAP_LABELS=[0,2,1,3]

def pair_features(data):
    d=np.asarray(data,dtype=float);n=d.shape[1]
    result=np.zeros((n,n,len(GRAPH_FEATURES)))
    for i,j in combinations(range(n),2):
        base=extract_one(d[:,[i,j]])
        others=[k for k in range(n) if k not in (i,j)]
        one=[abs(partial_correlation(d,i,j,(k,))) for k in others]
        two=[abs(partial_correlation(d,i,j,z)) for z in combinations(others,2)]
        pvalues=[fisher_p(d,i,j,z) for size in range(min(2,len(others))+1) for z in combinations(others,size)]
        extra=[min(one,default=abs(base[0])),min(two,default=min(one,default=abs(base[0]))),
               max(pvalues),float(np.mean(np.array(pvalues)>.01)),
               np.mean([abs(partial_correlation(d,i,k)) for k in others]) if others else 0.,
               np.mean([abs(partial_correlation(d,j,k)) for k in others]) if others else 0.]
        result[i,j]=np.r_[base,extra]
        result[j,i]=result[i,j,SWAP_FEATURES];result[j,i,2]*=-1
    return result

def graph_labels(target):
    a=np.asarray(target,dtype=bool);n=len(a);labels=np.zeros((n,n),int)
    labels[a&~a.T]=1;labels[a.T&~a]=2;labels[a&a.T]=3
    return labels

class GraphDiscoverer(nn.Module):
    def __init__(self,width=48):
        super().__init__();self.width=width
        self.register_buffer("mean",torch.zeros(len(GRAPH_FEATURES)))
        self.register_buffer("std",torch.ones(len(GRAPH_FEATURES)))
        self.encoder=nn.Sequential(nn.Linear(len(GRAPH_FEATURES),width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.attention=nn.MultiheadAttention(width,4,batch_first=True,dropout=0)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,4))

    def representation(self,x):
        b,n,_,_=x.shape
        z=((x-self.mean)/self.std).clamp(-20,20)
        h=self.encoder(z).reshape(b,n*n,self.width)
        mask=torch.eye(n,dtype=torch.bool,device=x.device).reshape(1,-1).expand(b,-1)
        context,_=self.attention(h,h,h,key_padding_mask=mask,need_weights=False)
        return self.norm(h+context).reshape(b,n,n,self.width)

    def from_hidden(self,h):
        logits=self.head(h)
        return .5*(logits+logits.transpose(1,2)[...,SWAP_LABELS])

    def forward(self,x):return self.from_hidden(self.representation(x))

def graph_probabilities(model,features,batch_size=32):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(torch.as_tensor(features[i:i+batch_size],dtype=torch.float32)).softmax(-1)
                          for i in range(0,len(features),batch_size)]).numpy()

def train_graph_model(groups,dev_groups,directory,*,epochs=35,width=48,seed=42):
    set_seed(seed);model=GraphDiscoverer(width)
    flat=np.concatenate([x[:,~np.eye(x.shape[1],dtype=bool)].reshape(-1,len(GRAPH_FEATURES)) for x,y in groups])
    # Symmetric normalization keeps pair orientation changes well-conditioned.
    model.mean.copy_(torch.tensor(flat.mean(0),dtype=torch.float32))
    model.std.copy_(torch.tensor(flat.std(0).clip(.03),dtype=torch.float32))
    counts=np.zeros(4)
    for x,y in groups:counts+=np.bincount(y[:,~np.eye(y.shape[1],dtype=bool)].ravel(),minlength=4)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(4*np.maximum(counts,1)),dtype=torch.float32))
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    history=[];best=float("inf");state=None;best_epoch=0
    for epoch in range(epochs):
        model.train()
        for x,y in groups:
            order=torch.randperm(len(x)).numpy();mask=~torch.eye(x.shape[1],dtype=torch.bool)
            for ids in np.array_split(order,max(1,int(np.ceil(len(order)/24)))):
                logits=model(torch.as_tensor(x[ids],dtype=torch.float32))
                truth=torch.as_tensor(y[ids],dtype=torch.long)
                opt.zero_grad();loss=loss_fn(logits[:,mask].reshape(-1,4),truth[:,mask].reshape(-1));loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        losses=[]
        for x,y in dev_groups:
            p=graph_probabilities(model,x);mask=~np.eye(x.shape[1],dtype=bool)
            pp,yy=p[:,mask],y[:,mask]
            losses.append(float(-np.log(np.take_along_axis(pp,yy[...,None],axis=-1).clip(1e-9)).mean()))
        val=float(np.mean(losses));history.append({"epoch":epoch+1,"dev_cross_entropy":val})
        if val<best:best=val;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch==0 or (epoch+1)%10==0:print(f"graph epoch {epoch+1}/{epochs}: dev loss {val:.4f}",flush=True)
    model.load_state_dict(state);model.eval()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    torch.save({"width":width,"state_dict":state},directory/"graph_teacher.pt")
    save_json(directory/"training.json",{"history":history,"best_epoch":best_epoch,"seed":seed,
                "architecture":"pair-statistic front end + shared set attention","frozen":True})
    return model

def load_graph_model(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    model=GraphDiscoverer(state["width"]);model.load_state_dict(state["state_dict"]);model.eval();return model

def decode_graph(probabilities):
    p=np.asarray(probabilities);n=len(p)
    labels=p.argmax(-1)
    raw=np.zeros((n,n),bool)
    for i,j in combinations(range(n),2):
        cls=int(labels[i,j])
        if cls==1:raw[i,j]=True
        elif cls==2:raw[j,i]=True
        elif cls==3:raw[i,j]=raw[j,i]=True
    directed=raw&~raw.T
    scores=np.where(directed,p[:,:,1],0)
    dag,rejected=acyclic_projection(scores,threshold=1e-12)
    partial=dag|(raw&raw.T)
    return partial,{"removed_cycle_edges":rejected,"raw_graph":raw.astype(int).tolist(),
                    "semantics":"partially_directed_prediction_not_guaranteed_completed_PDAG"}

def decode_graph_threshold(probabilities,edge_threshold):
    """Decode pair probabilities with an explicit edge-presence threshold.

    Pair probabilities are symmetrized before thresholding.  Directed choices
    are projected to a DAG; undirected choices remain explicit CPDAG edges.
    """
    p=np.asarray(probabilities,dtype=float);n=len(p)
    if p.shape!=(n,n,4) or not np.isfinite(p).all() or not 0<=edge_threshold<=1:
        raise ValueError("Invalid threshold decoder input")
    raw=np.zeros((n,n),bool);confidence=np.zeros((n,n),float)
    for i,j in combinations(range(n),2):
        q=.5*(p[i,j]+p[j,i,SWAP_LABELS])
        if 1-q[0]<edge_threshold:continue
        cls=1+int(np.argmax(q[1:]))
        if cls==1:raw[i,j]=True;confidence[i,j]=q[1]
        elif cls==2:raw[j,i]=True;confidence[j,i]=q[2]
        else:raw[i,j]=raw[j,i]=True
    directed=raw&~raw.T
    dag,rejected=acyclic_projection(np.where(directed,confidence,0.),threshold=1e-12)
    partial=dag|(raw&raw.T)
    return partial,{"edge_threshold":float(edge_threshold),"removed_cycle_edges":rejected,
                    "raw_graph":raw.astype(int).tolist(),"semantics":"calibrated_partially_directed_prediction"}
def dag_completion(partial):
    """One explicit acyclic completion, not an identification of ambiguous edges."""
    p=np.asarray(partial,dtype=bool);directed=p&~p.T
    order=topological_order(directed);rank={v:i for i,v in enumerate(order)}
    completed=directed.copy();chosen=[]
    for i,j in combinations(range(len(p)),2):
        if p[i,j] and p[j,i]:
            u,v=(i,j) if rank[i]<rank[j] else (j,i)
            completed[u,v]=True;chosen.append([u,v])
    topological_order(completed)
    return completed,chosen

def graph_metrics(truth,neural,symbolic):
    truth=np.asarray(truth,bool);neural=np.asarray(neural,bool);symbolic=np.asarray(symbolic,bool)
    n=truth.shape[1];pairs=np.triu(np.ones((n,n),bool),1)
    def categories(a):return np.stack([graph_labels(g)[pairs] for g in a])
    t,nn,p=categories(truth),categories(neural),categories(symbolic)
    edge_t,edge_p=t>0,p>0
    tp=int(np.sum(edge_t&edge_p));fp=int(np.sum(~edge_t&edge_p));fn=int(np.sum(edge_t&~edge_p))
    f1=2*tp/max(2*tp+fp+fn,1)
    class_scores=[]
    for cls in range(4):
        a=int(np.sum((p==cls)&(t==cls)));b=int(np.sum((p==cls)&(t!=cls)));c=int(np.sum((p!=cls)&(t==cls)))
        class_scores.append(2*a/max(2*a+b+c,1))
    return {"worlds":len(truth),"nodes":n,"pair_fidelity":float(np.mean(p==nn)),
            "pair_accuracy":float(np.mean(p==t)),"macro_f1":float(np.mean(class_scores)),
            "skeleton_f1":f1,"mean_pair_shd":float(np.mean(np.sum(p!=t,axis=1))),
            "exact_graph_accuracy":float(np.mean(np.all(p==t,axis=1))),
            "neural_pair_accuracy":float(np.mean(nn==t)),
            "directed_target_pairs":int(np.sum((t==1)|(t==2))),
            "directed_target_accuracy":float(np.mean(p[(t==1)|(t==2)]==t[(t==1)|(t==2)])) if np.any((t==1)|(t==2)) else None}
