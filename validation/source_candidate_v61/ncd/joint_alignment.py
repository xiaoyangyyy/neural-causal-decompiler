"""Joint orthogonal mappings for multiple rule intermediates.

Each variable has a disjoint subspace block and its own source world. This
module does not select on test data, retrain the teacher, or claim equivalence.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from .program_trace import ProgramExecutor
from .rules import Rule

@dataclass(frozen=True)
class JointPairs:
    base:np.ndarray
    sources:np.ndarray
    masks:np.ndarray
    target:np.ndarray
    initial:np.ndarray
    source_visited:np.ndarray
    target_visited:np.ndarray

def _check_addresses(executor,addresses):
    if not isinstance(executor.program,Rule):raise ValueError("Joint mapping requires world-row Rule semantics")
    if not addresses or len(set(addresses))!=len(addresses):raise ValueError("Distinct intermediates required")
    for address in addresses:
        prefix,sep,path=address.partition(":")
        if not sep or prefix!=executor.program_id or path not in executor.catalog:raise ValueError("Unknown intermediate")
        if executor.catalog[path]["kind"] not in ("vector","bool"):raise ValueError("Only per-world numeric/boolean intermediates")
    return tuple(addresses)

def make_pairs(executor,features,addresses,combinations,*,count=256,seed=42,disjoint=False):
    addresses=_check_addresses(executor,addresses)
    x=np.asarray(features);k=len(addresses);n=len(x)
    if n<2 or count<1:raise ValueError("Too few worlds or pairs")
    masks=np.asarray(combinations)
    if masks.ndim!=2 or masks.shape[1]!=k or masks.dtype.kind!="b" or not masks.any(1).all():
        raise ValueError("Nonempty Boolean intervention combinations required")
    if len(np.unique(masks,axis=0))!=len(masks):raise ValueError("Duplicate combinations")
    rng=np.random.default_rng(seed)
    if disjoint:
        # Each row consumes k+1 different worlds; none are reused across rows.
        order=rng.permutation(n);count=min(count,n//(k+1))
        if not count:raise ValueError("Insufficient worlds for disjoint groups")
        groups=order[:count*(k+1)].reshape(count,k+1)
        base=groups[:,0];sources=groups[:,1:]
    else:
        base=rng.integers(0,n,count)
        sources=(base[:,None]+rng.integers(1,n,(count,k)))%n
    chosen=masks[np.arange(count)%len(masks)].copy();rng.shuffle(chosen)
    baseline=executor.execute(x)
    values=[]
    for address in addresses:
        value=np.asarray(baseline.records[address]["value"])
        if value.shape!=(n,):raise ValueError("Intermediate is not per-world")
        values.append(value)
    target=np.empty(count,dtype=int)
    source_visited=np.column_stack([np.broadcast_to(baseline.records[a]["visited"],(n,))[sources[:,j]] for j,a in enumerate(addresses)])
    target_visited=np.zeros((count,k),bool)
    for mask in np.unique(chosen,axis=0):
        selected=np.flatnonzero(np.all(chosen==mask,axis=1))
        patches={a:values[j][sources[selected,j]] for j,a in enumerate(addresses) if mask[j]}
        trace=executor.execute(x[base[selected]],patches)
        target[selected]=trace.output
        for j,address in enumerate(addresses):
            if address in trace.records:
                target_visited[selected,j]=trace.records[address]["visited"]
    return JointPairs(base,sources,chosen,target,np.asarray(baseline.output)[base],source_visited,target_visited)

def patch_blocks(base,sources,basis,masks):
    """Torch or numpy; basis is [hidden, variables, rank]."""
    if basis.ndim!=3 or sources.ndim!=3 or sources.shape[:2]!=masks.shape:
        raise ValueError("Joint patch dimensions")
    if sources.shape[0]!=base.shape[0] or sources.shape[2]!=base.shape[1] or basis.shape[:2]!=(base.shape[1],sources.shape[1]):
        raise ValueError("Incompatible hidden/basis shapes")
    out=base
    for j in range(basis.shape[1]):
        q=basis[:,j,:]
        out=out+((sources[:,j,:]-base)@q@q.T)*masks[:,j:j+1]
    return out

def _logits(model,h):
    width=h.shape[1]//2
    return model.from_hidden(h[:,:width],h[:,width:])

@contextmanager
def frozen_teacher(model):
    params=list(model.parameters());flags=[p.requires_grad for p in params]
    modes={m:m.training for m in model.modules()}
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    try:
        model.eval()
        for p in params:p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(params,flags):p.requires_grad_(flag)
        for module,mode in modes.items():module.training=mode
        if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):
            raise RuntimeError("Frozen teacher state changed")

def train_joint_mapping(model,h,features,executor,addresses,combinations,*,rank=1,steps=120,seed=42,shuffle_targets=False):
    _check_addresses(executor,addresses);h=np.asarray(h)
    k=len(addresses);d=h.shape[1]
    if len(h)!=len(features) or not isinstance(rank,int) or rank<1 or k*rank>d or steps<1 or not np.isfinite(h).all():
        raise ValueError("Invalid mapping dimensions or budget")
    # Local generator; preserve caller RNG rather than resetting global state.
    generator=torch.Generator().manual_seed(seed)
    parameter=nn.Parameter(torch.randn(d,k*rank,generator=generator)*.1)
    optimizer=torch.optim.Adam([parameter],lr=.025);ht=torch.tensor(h,dtype=torch.float32)
    rng=np.random.default_rng(seed);losses=[];eligible_counts=[]
    with frozen_teacher(model):
        for step in range(steps):
            pairs=make_pairs(executor,features,addresses,combinations,count=128,seed=seed+step)
            eligible=(~pairs.masks|(pairs.source_visited & pairs.target_visited)).all(1)
            eligible_counts.append(int(eligible.sum()))
            if not eligible.any():
                losses.append(None);continue
            targets=pairs.target.copy()
            if shuffle_targets:targets[eligible]=targets[eligible][rng.permutation(int(eligible.sum()))]
            basis=torch.linalg.qr(parameter,mode="reduced").Q.reshape(d,k,rank)
            patched=patch_blocks(ht[pairs.base],ht[pairs.sources],basis,torch.tensor(pairs.masks,dtype=torch.float32))
            logits=_logits(model,patched)
            weights=torch.tensor(np.where(pairs.target!=pairs.initial,3.,1.),dtype=torch.float32)
            per_pair=nn.functional.cross_entropy(logits,torch.tensor(targets),reduction="none")*weights
            loss=per_pair[torch.tensor(eligible)].mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    q=torch.linalg.qr(parameter.detach(),mode="reduced").Q.numpy().astype(float).reshape(d,k,rank)
    return q,{"loss":losses,"steps":steps,"eligible_pairs_per_step":eligible_counts,
              "optimizer_updates":sum(n>0 for n in eligible_counts),"rank":rank,"addresses":list(addresses),
              "training_combinations":np.asarray(combinations).tolist(),"teacher_frozen":True,
              "shuffle_targets":shuffle_targets}

def measure_joint_mapping(model,h,basis,pairs):
    h=np.asarray(h);q=np.asarray(basis);flat=q.reshape(q.shape[0],-1)
    if not np.allclose(flat.T@flat,np.eye(flat.shape[1]),atol=1e-5):
        raise ValueError("Mapping blocks must be jointly orthonormal")
    with frozen_teacher(model),torch.no_grad():
        initial=_logits(model,torch.tensor(h[pairs.base],dtype=torch.float32)).argmax(1).numpy()
        patched=patch_blocks(h[pairs.base],h[pairs.sources],q,pairs.masks)
        after=_logits(model,torch.tensor(patched,dtype=torch.float32)).argmax(1).numpy()
    executed=(~pairs.masks|(pairs.source_visited & pairs.target_visited)).all(1)
    meaningful=(initial==pairs.initial)&(pairs.target!=pairs.initial)&executed
    def metrics(ids):
        use=ids&meaningful
        return {"pairs":int(ids.sum()),"all_active_intermediates_executed":int((ids&executed).sum()),"symbolic_changes":int(((pairs.target!=pairs.initial)&ids).sum()),
                "baseline_agreement":float(np.mean(initial[ids]==pairs.initial[ids])),
                "interchange_accuracy":float(np.mean(after[ids]==pairs.target[ids])),
                "no_intervention_accuracy":float(np.mean(initial[ids]==pairs.target[ids])),
                "informative_pairs":int(use.sum()),
                "informative_accuracy":float(np.mean(after[use]==pairs.target[use])) if use.any() else None}
    return {"overall":metrics(np.ones(len(after),bool)),
            "by_combination":{",".join(map(str,np.flatnonzero(mask))):metrics(np.all(pairs.masks==mask,axis=1))
                              for mask in np.unique(pairs.masks,axis=0)},
            "base_indices":pairs.base.tolist(),"source_indices":pairs.sources.tolist(),
            "masks":pairs.masks.tolist(),"source_visited":pairs.source_visited.tolist(),
            "target_visited":pairs.target_visited.tolist(),"symbolic_target":pairs.target.tolist(),
            "neural_before":initial.tolist(),"neural_after":after.tolist(),
            "claim":"joint empirical intervention fidelity; not general equivalence"}
