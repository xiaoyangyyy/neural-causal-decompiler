"""Biorthogonal concept interventions for entangled neural representations.

The historical mapping uses one jointly orthogonal basis for both reading and
writing a concept. Here read and write are dual bases. Individual concept
projectors may therefore be oblique while interventions remain isolated.
"""
import numpy as np
import torch
from torch import nn

from .joint_alignment import frozen_teacher


def _dual_from_write(write):
    gram=write.T@write
    if isinstance(write,torch.Tensor):
        return write@torch.linalg.inv(gram)
    return write@np.linalg.inv(gram)


def biorthogonal_bases(write,variables,rank=1):
    """Normalize write columns and construct their exact dual read columns."""
    if write.ndim!=2 or type(variables) is not int or type(rank) is not int or variables<1 or rank<1:
        raise ValueError("Invalid oblique basis dimensions")
    if write.shape[1]!=variables*rank or write.shape[0]<write.shape[1]:
        raise ValueError("Oblique basis must fit in hidden dimension")
    if isinstance(write,torch.Tensor):
        normalized=write/torch.linalg.vector_norm(write,dim=0,keepdim=True).clamp_min(1e-8)
        if torch.linalg.matrix_rank(normalized)<normalized.shape[1]:raise ValueError("Rank-deficient oblique basis")
    else:
        write=np.asarray(write,dtype=float)
        if not np.isfinite(write).all():raise ValueError("Finite oblique basis required")
        normalized=write/np.maximum(np.linalg.norm(write,axis=0,keepdims=True),1e-8)
        if np.linalg.matrix_rank(normalized)<normalized.shape[1]:raise ValueError("Rank-deficient oblique basis")
    read=_dual_from_write(normalized)
    d=normalized.shape[0]
    return read.reshape(d,variables,rank),normalized.reshape(d,variables,rank)


def patch_oblique(base,sources,read,write,masks):
    """Patch source concept coordinates into base states (torch or numpy)."""
    if read.ndim!=3 or write.shape!=read.shape or sources.ndim!=3 or sources.shape[:2]!=masks.shape:
        raise ValueError("Oblique patch dimensions")
    if sources.shape[0]!=base.shape[0] or sources.shape[2]!=base.shape[1] or read.shape[:2]!=(base.shape[1],sources.shape[1]):
        raise ValueError("Incompatible oblique hidden/basis shapes")
    out=base
    for j in range(read.shape[1]):
        delta=(sources[:,j,:]-base)@read[:,j,:]
        out=out+(delta@write[:,j,:].T)*masks[:,j:j+1]
    return out


def validate_oblique(read,write,atol=2e-5):
    read=np.asarray(read,dtype=float);write=np.asarray(write,dtype=float)
    if read.ndim!=3 or write.shape!=read.shape or not np.isfinite(read).all() or not np.isfinite(write).all():
        raise ValueError("Invalid oblique mapping")
    r=read.reshape(read.shape[0],-1);w=write.reshape(write.shape[0],-1)
    cross=r.T@w
    if not np.allclose(cross,np.eye(cross.shape[0]),atol=atol):raise ValueError("Read/write bases are not biorthogonal")
    gram=w.T@w
    return {"biorthogonality_max_error":float(np.max(np.abs(cross-np.eye(len(cross))))),
        "write_gram_condition":float(np.linalg.cond(gram)),
        "max_write_cosine":float(np.max(np.abs(gram-np.eye(len(gram))))) if len(gram)>1 else 0.}


def _initial_write(readout,rank,seed):
    d=len(readout.mean);k=len(readout.addresses);rng=np.random.default_rng(seed)
    physical=np.asarray(readout.coefficients[:-1],dtype=float)/np.asarray(readout.scale)[:,None]
    if rank==1 and np.linalg.matrix_rank(physical)==k:
        return physical@np.linalg.inv(physical.T@physical)
    initial=rng.normal(size=(d,k*rank))
    if rank>1 and np.linalg.matrix_rank(physical)==k:
        initial[:,::rank]=physical@np.linalg.inv(physical.T@physical)
    return initial


def train_fixed_oblique_mapping(decoder,h,pairs,targets,visited,readout,*,rank=1,steps=120,
        batch_size=128,seed=1091,numeric_weight=.5,behavioral_weight=1.,conditioning_weight=.05,shuffle_targets=False,initial_write=None):
    """Fit biorthogonal read/write bases on a frozen, fixed pair pool."""
    h=np.asarray(h,dtype=float);targets=np.asarray(targets,dtype=float);visited=np.asarray(visited,dtype=bool)
    d=h.shape[1] if h.ndim==2 else 0;k=len(readout.addresses);count=len(pairs.base);m=k*rank
    if h.ndim!=2 or targets.shape!=visited.shape or targets.shape!=(count,k) or pairs.masks.shape!=(count,k):
        raise ValueError("Fixed oblique pool mismatch")
    if not count or type(rank) is not int or rank<1 or m>d or steps<1 or batch_size<1 or numeric_weight<0 or behavioral_weight<0 or conditioning_weight<0 or not np.isfinite(targets).all():
        raise ValueError("Invalid fixed oblique mapping budget")
    initial=_initial_write(readout,rank,seed) if initial_write is None else np.asarray(initial_write,dtype=float).reshape(d,m)
    if initial.shape!=(d,m) or not np.isfinite(initial).all() or np.linalg.matrix_rank(initial)<m:
        raise ValueError("Invalid oblique initialization")
    parameter=nn.Parameter(torch.tensor(initial,dtype=torch.float32))
    optimizer=torch.optim.Adam([parameter],lr=.005);rng=np.random.default_rng(seed)
    ht=torch.tensor(h,dtype=torch.float32);mean=torch.tensor(readout.mean,dtype=torch.float32);scale=torch.tensor(readout.scale,dtype=torch.float32)
    beta=torch.tensor(readout.coefficients,dtype=torch.float32);variance=torch.tensor(readout.target_variance,dtype=torch.float32)
    target_order=rng.permutation(count) if shuffle_targets else np.arange(count);history=[]
    def objective(ids,tids):
        columns=parameter/torch.linalg.vector_norm(parameter,dim=0,keepdim=True).clamp_min(1e-8)
        gram=columns.T@columns;dual=columns@torch.linalg.inv(gram+torch.eye(m)*1e-6)
        read=dual.reshape(d,k,rank);write=columns.reshape(d,k,rank)
        patched=patch_oblique(ht[pairs.base[ids]],ht[pairs.sources[ids]],read,write,torch.tensor(pairs.masks[ids],dtype=torch.float32))
        width=d//2;logits=decoder.from_hidden(patched[:,:width],patched[:,width:])
        prediction=torch.cat([(patched-mean)/scale,torch.ones(len(ids),1)],1)@beta
        target=torch.tensor(targets[tids],dtype=torch.float32);visit=torch.tensor(visited[tids])
        error=(prediction-target)/torch.sqrt(variance)
        numeric=nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction="none")[visit].mean()
        behavioral=nn.functional.cross_entropy(logits,torch.tensor(pairs.target[tids]))
        # Unit diagonal makes -log(det(Gram))/m zero at orthogonality and
        # increasingly penalizes unstable overlap without forbidding it.
        conditioning=-torch.linalg.slogdet(gram+torch.eye(m)*1e-6)[1]/m
        total=behavioral+numeric_weight*numeric+conditioning_weight*conditioning if behavioral_weight==1. else behavioral_weight*behavioral+numeric_weight*numeric+conditioning_weight*conditioning
        return total,behavioral,numeric,conditioning,torch.linalg.eigvalsh(gram)[0]
    all_ids=np.arange(count);best_parameter=parameter.detach().clone();best_step=-1
    with frozen_teacher(decoder):
        with torch.no_grad():best_loss=float(objective(all_ids,target_order)[0])
        initial_loss=best_loss
        for step in range(steps):
            ids=rng.permutation(count)[:min(batch_size,count)];tids=target_order[ids]
            loss,behavioral,numeric,conditioning,mineig=objective(ids,tids)
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():full_loss=float(objective(all_ids,target_order)[0])
            if full_loss<best_loss:
                best_loss=full_loss;best_step=step;best_parameter=parameter.detach().clone()
            history.append({"step":step,"batch":ids.tolist(),"target_indices":tids.tolist(),"loss":float(loss.detach()),
                "full_training_loss":full_loss,"behavioral_loss":float(behavioral.detach()),"numeric_loss":float(numeric.detach()),
                "conditioning_loss":float(conditioning.detach()),"minimum_gram_eigenvalue":float(mineig.detach())})
    read,write=biorthogonal_bases(best_parameter.numpy().astype(float),k,rank)
    diagnostics=validate_oblique(read,write)
    record={"seed":seed,"rank":rank,"steps":steps,"batch_size":batch_size,"numeric_weight":numeric_weight,
        "conditioning_weight":conditioning_weight,"shuffle_targets":shuffle_targets,"teacher_frozen":True,"readout_frozen":True,
        "addresses":list(readout.addresses),"pair_pool_size":count,"training_combinations":np.unique(pairs.masks,axis=0).tolist(),
        "optimizer_updates":steps,"initialization":"provided" if initial_write is not None else "probe_dual_or_seeded_random",
        "checkpoint_selection":"minimum full training-pool objective including step 0","initial_full_training_loss":initial_loss,
        "selected_full_training_loss":best_loss,"selected_step":best_step,"diagnostics":diagnostics,"history":history}
    if behavioral_weight!=1.:record["behavioral_weight"]=behavioral_weight
    return read,write,record

def measure_oblique_mapping(model,h,read,write,pairs):
    """Behavioral interchange audit for a validated oblique mapping."""
    validate_oblique(read,write);h=np.asarray(h)
    with frozen_teacher(model),torch.no_grad():
        def logits(state):
            state=torch.tensor(state,dtype=torch.float32);width=state.shape[1]//2
            return model.from_hidden(state[:,:width],state[:,width:])
        initial=logits(h[pairs.base]).argmax(1).numpy()
        patched=patch_oblique(h[pairs.base],h[pairs.sources],read,write,pairs.masks)
        after=logits(patched).argmax(1).numpy()
    executed=(~pairs.masks|(pairs.source_visited & pairs.target_visited)).all(1)
    meaningful=(initial==pairs.initial)&(pairs.target!=pairs.initial)&executed
    def metrics(ids):
        use=ids&meaningful
        return {"pairs":int(ids.sum()),"all_active_intermediates_executed":int((ids&executed).sum()),
            "symbolic_changes":int(((pairs.target!=pairs.initial)&ids).sum()),
            "baseline_agreement":float(np.mean(initial[ids]==pairs.initial[ids])),
            "interchange_accuracy":float(np.mean(after[ids]==pairs.target[ids])),
            "no_intervention_accuracy":float(np.mean(initial[ids]==pairs.target[ids])),
            "informative_pairs":int(use.sum()),
            "informative_accuracy":float(np.mean(after[use]==pairs.target[use])) if use.any() else None}
    return {"overall":metrics(np.ones(len(after),bool)),
        "by_combination":{",".join(map(str,np.flatnonzero(mask))):metrics(np.all(pairs.masks==mask,axis=1))
                          for mask in np.unique(pairs.masks,axis=0)},
        "base_indices":pairs.base.tolist(),"source_indices":pairs.sources.tolist(),"masks":pairs.masks.tolist(),
        "source_visited":pairs.source_visited.tolist(),"target_visited":pairs.target_visited.tolist(),
        "symbolic_target":pairs.target.tolist(),"neural_before":initial.tolist(),"neural_after":after.tolist(),
        "mapping_diagnostics":validate_oblique(read,write),
        "claim":"joint empirical biorthogonal intervention fidelity; not general equivalence"}
