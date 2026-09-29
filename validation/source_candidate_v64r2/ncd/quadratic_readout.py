"""Explicit diagonal-quadratic numeric readouts and frozen mapping trainers."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .joint_alignment import frozen_teacher,patch_blocks
from .oblique_mapping import patch_oblique,biorthogonal_bases,validate_oblique

@dataclass
class QuadraticNumericReadout:
    addresses:tuple
    hidden_mean:np.ndarray
    hidden_scale:np.ndarray
    feature_mean:np.ndarray
    feature_scale:np.ndarray
    coefficients:np.ndarray
    target_variance:np.ndarray
    fit_counts:np.ndarray
    def _features(self,h):
        h=np.asarray(h,float)
        if h.ndim!=2 or h.shape[1]!=len(self.hidden_mean) or not np.isfinite(h).all():raise ValueError("Quadratic readout input mismatch")
        z=(h-self.hidden_mean)/self.hidden_scale;phi=np.c_[z,z*z]
        return (phi-self.feature_mean)/self.feature_scale
    def predict(self,h):return np.c_[self._features(h),np.ones(len(h))]@self.coefficients
    def torch_predict(self,h):
        hm=torch.as_tensor(self.hidden_mean,dtype=h.dtype,device=h.device);hs=torch.as_tensor(self.hidden_scale,dtype=h.dtype,device=h.device)
        fm=torch.as_tensor(self.feature_mean,dtype=h.dtype,device=h.device);fs=torch.as_tensor(self.feature_scale,dtype=h.dtype,device=h.device);beta=torch.as_tensor(self.coefficients,dtype=h.dtype,device=h.device)
        z=(h-hm)/hs;phi=torch.cat([z,z*z],1);design=torch.cat([(phi-fm)/fs,torch.ones(len(h),1,dtype=h.dtype,device=h.device)],1)
        return design@beta
    def to_dict(self):
        return {"kind":"diagonal_quadratic","addresses":list(self.addresses),**{k:getattr(self,k).tolist() for k in ("hidden_mean","hidden_scale","feature_mean","feature_scale","coefficients","target_variance","fit_counts")}}
    @classmethod
    def from_dict(cls,d):
        if d.get("kind")!="diagonal_quadratic":raise ValueError("Quadratic readout kind mismatch")
        return cls(tuple(d["addresses"]),*(np.asarray(d[k]) for k in ("hidden_mean","hidden_scale","feature_mean","feature_scale","coefficients","target_variance","fit_counts")))

def fit_quadratic_readout_values(h,addresses,target,visited,ridge=.01):
    h=np.asarray(h,float);target=np.asarray(target,float);visited=np.asarray(visited,bool)
    if h.ndim!=2 or target.shape!=visited.shape or target.shape!=(len(h),len(addresses)) or not np.isfinite(h).all() or ridge<=0:raise ValueError("Invalid quadratic readout data")
    hm=h.mean(0);hs=h.std(0).clip(1e-6);z=(h-hm)/hs;phi=np.c_[z,z*z];fm=phi.mean(0);fs=phi.std(0).clip(1e-6);design=np.c_[(phi-fm)/fs,np.ones(len(h))]
    coefficients=[];variances=[];counts=[]
    for j in range(len(addresses)):
        ids=visited[:,j]
        if ids.sum()<4:raise ValueError("Insufficient quadratic fitting worlds")
        a=design[ids];y=target[ids,j];penalty=np.eye(a.shape[1])*ridge;penalty[-1,-1]=0
        coefficients.append(np.linalg.solve(a.T@a+penalty,a.T@y));variances.append(max(float(y.var()),1e-8));counts.append(int(ids.sum()))
    return QuadraticNumericReadout(tuple(addresses),hm,hs,fm,fs,np.column_stack(coefficients),np.asarray(variances),np.asarray(counts))

def train_quadratic_orthogonal(decoder,h,pairs,targets,visited,readout,*,rank=1,steps=120,batch_size=128,seed=3191,numeric_weight=.5,behavioral_weight=1.,shuffle_targets=False):
    h=np.asarray(h,float);targets=np.asarray(targets,float);visited=np.asarray(visited,bool);d=h.shape[1];k=len(readout.addresses);count=len(pairs.base)
    if h.ndim!=2 or targets.shape!=(count,k) or visited.shape!=targets.shape or pairs.masks.shape!=(count,k) or not count or k*rank>d:raise ValueError("Quadratic orthogonal pool mismatch")
    generator=torch.Generator().manual_seed(seed);parameter=nn.Parameter(torch.randn(d,k*rank,generator=generator)*.1);optimizer=torch.optim.Adam([parameter],lr=.025);ht=torch.tensor(h,dtype=torch.float32);variance=torch.tensor(readout.target_variance,dtype=torch.float32);rng=np.random.default_rng(seed);target_order=rng.permutation(count) if shuffle_targets else np.arange(count);history=[]
    with frozen_teacher(decoder):
      for step in range(steps):
        ids=rng.permutation(count)[:min(batch_size,count)];tids=target_order[ids];q=torch.linalg.qr(parameter,mode="reduced").Q.reshape(d,k,rank);patched=patch_blocks(ht[pairs.base[ids]],ht[pairs.sources[ids]],q,torch.tensor(pairs.masks[ids],dtype=torch.float32));logits=decoder.from_hidden(patched[:,:d//2],patched[:,d//2:]);prediction=readout.torch_predict(patched);target=torch.tensor(targets[tids],dtype=torch.float32);visit=torch.tensor(visited[tids]);error=(prediction-target)/torch.sqrt(variance);numeric=nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction="none")[visit].mean();behavioral=nn.functional.cross_entropy(logits,torch.tensor(pairs.target[tids]));loss=behavioral_weight*behavioral+numeric_weight*numeric;optimizer.zero_grad();loss.backward();optimizer.step();history.append({"step":step,"batch":ids.tolist(),"target_indices":tids.tolist(),"loss":float(loss.detach()),"behavioral_loss":float(behavioral.detach()),"numeric_loss":float(numeric.detach())})
    basis=torch.linalg.qr(parameter.detach(),mode="reduced").Q.numpy().astype(float).reshape(d,k,rank);return basis,{"seed":seed,"rank":rank,"steps":steps,"batch_size":batch_size,"numeric_weight":numeric_weight,"behavioral_weight":behavioral_weight,"shuffle_targets":shuffle_targets,"teacher_frozen":True,"readout_frozen":True,"readout_kind":"diagonal_quadratic","addresses":list(readout.addresses),"pair_pool_size":count,"training_combinations":np.unique(pairs.masks,axis=0).tolist(),"optimizer_updates":steps,"history":history}

def train_quadratic_oblique(decoder,h,pairs,targets,visited,readout,*,rank=1,steps=120,batch_size=128,seed=3191,numeric_weight=.5,behavioral_weight=1.,conditioning_weight=.05,shuffle_targets=False,initial_write=None):
    h=np.asarray(h,float);targets=np.asarray(targets,float);visited=np.asarray(visited,bool);d=h.shape[1];k=len(readout.addresses);count=len(pairs.base);m=k*rank;rng=np.random.default_rng(seed)
    if h.ndim!=2 or targets.shape!=(count,k) or visited.shape!=targets.shape or pairs.masks.shape!=(count,k) or not count or m>d:raise ValueError("Quadratic oblique pool mismatch")
    initial=rng.normal(size=(d,m)) if initial_write is None else np.asarray(initial_write,float).reshape(d,m)
    if np.linalg.matrix_rank(initial)<m:raise ValueError("Invalid quadratic oblique initialization")
    parameter=nn.Parameter(torch.tensor(initial,dtype=torch.float32));optimizer=torch.optim.Adam([parameter],lr=.005);ht=torch.tensor(h,dtype=torch.float32);variance=torch.tensor(readout.target_variance,dtype=torch.float32);target_order=rng.permutation(count) if shuffle_targets else np.arange(count);history=[]
    def objective(ids,tids):
        columns=parameter/torch.linalg.vector_norm(parameter,dim=0,keepdim=True).clamp_min(1e-8);gram=columns.T@columns;dual=columns@torch.linalg.inv(gram+torch.eye(m)*1e-6);read=dual.reshape(d,k,rank);write=columns.reshape(d,k,rank);patched=patch_oblique(ht[pairs.base[ids]],ht[pairs.sources[ids]],read,write,torch.tensor(pairs.masks[ids],dtype=torch.float32));logits=decoder.from_hidden(patched[:,:d//2],patched[:,d//2:]);prediction=readout.torch_predict(patched);target=torch.tensor(targets[tids],dtype=torch.float32);visit=torch.tensor(visited[tids]);error=(prediction-target)/torch.sqrt(variance);numeric=nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction="none")[visit].mean();behavioral=nn.functional.cross_entropy(logits,torch.tensor(pairs.target[tids]));conditioning=-torch.linalg.slogdet(gram+torch.eye(m)*1e-6)[1]/m;return behavioral_weight*behavioral+numeric_weight*numeric+conditioning_weight*conditioning,behavioral,numeric,conditioning,torch.linalg.eigvalsh(gram)[0]
    all_ids=np.arange(count);best_parameter=parameter.detach().clone();best_step=-1
    with frozen_teacher(decoder):
      with torch.no_grad():best_loss=float(objective(all_ids,target_order)[0]);initial_loss=best_loss
      for step in range(steps):
        ids=rng.permutation(count)[:min(batch_size,count)];tids=target_order[ids];loss,behavioral,numeric,conditioning,mineig=objective(ids,tids);optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():full=float(objective(all_ids,target_order)[0])
        if full<best_loss:best_loss=full;best_step=step;best_parameter=parameter.detach().clone()
        history.append({"step":step,"batch":ids.tolist(),"target_indices":tids.tolist(),"loss":float(loss.detach()),"full_training_loss":full,"behavioral_loss":float(behavioral.detach()),"numeric_loss":float(numeric.detach()),"conditioning_loss":float(conditioning.detach()),"minimum_gram_eigenvalue":float(mineig.detach())})
    read,write=biorthogonal_bases(best_parameter.numpy().astype(float),k,rank);record={"seed":seed,"rank":rank,"steps":steps,"batch_size":batch_size,"numeric_weight":numeric_weight,"behavioral_weight":behavioral_weight,"conditioning_weight":conditioning_weight,"shuffle_targets":shuffle_targets,"teacher_frozen":True,"readout_frozen":True,"readout_kind":"diagonal_quadratic","addresses":list(readout.addresses),"pair_pool_size":count,"training_combinations":np.unique(pairs.masks,axis=0).tolist(),"optimizer_updates":steps,"initialization":"provided" if initial_write is not None else "seeded_random","checkpoint_selection":"minimum full training-pool objective including step 0","initial_full_training_loss":initial_loss,"selected_full_training_loss":best_loss,"selected_step":best_step,"diagnostics":validate_oblique(read,write),"history":history};return read,write,record
