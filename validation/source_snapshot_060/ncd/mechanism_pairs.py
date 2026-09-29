"""Compatible symbolic interventions with unique active source worlds.

Acceptance uses symbolic reachability only, never neural predictions or labels.
Rejected groups consume their worlds too. Results describe an execution-
conditioned distribution and are not estimates for unconditioned SCM sampling.
"""
from itertools import combinations
import numpy as np
from .joint_alignment import JointPairs,_check_addresses

def compatible(executor,addresses):
    _check_addresses(executor,addresses)
    paths=[tuple(executor.intervention_paths(a)) if hasattr(executor,"intervention_paths") else (a.split(":",1)[1],) for a in addresses]
    for i,left_paths in enumerate(paths):
        for right_paths in paths[i+1:]:
            for a in left_paths:
                for b in right_paths:
                    if a.startswith(b+"/") or b.startswith(a+"/"):return False
                    for p,q in ((a,b),(b,a)):
                        if p.endswith("/predicate"):
                            expr=p[:-len("predicate")]+"expr"
                            if q==expr or q.startswith(expr+"/"):return False
                    # Rule branches: two incompatible branches cannot both execute.
                    aa=a.split("/");bb=b.split("/")
                    for left,right in zip(aa,bb):
                        if left!=right:
                            if {left,right}=={"left","right"}:return False
                            break
    if hasattr(executor,"interventions_compatible") and not executor.interventions_compatible(addresses):return False
    return True

def compatible_combinations(executor,addresses,max_order=2):
    _check_addresses(executor,addresses)
    if not isinstance(max_order,int) or max_order<1:raise ValueError("Invalid combination order")
    result=[]
    for size in range(1,min(max_order,len(addresses))+1):
        for chosen in combinations(range(len(addresses)),size):
            if not compatible(executor,[addresses[i] for i in chosen]):continue
            mask=np.zeros(len(addresses),bool);mask[list(chosen)]=True;result.append(mask)
    return np.array(result,dtype=bool)

def execution_conditioned_pairs(executor,features,addresses,masks,*,count=256,seed=42):
    addresses=_check_addresses(executor,addresses);x=np.asarray(features);k=len(addresses)
    masks=np.asarray(masks)
    if type(count) is not int or count<1:raise ValueError("Invalid pair budget")
    if masks.ndim!=2 or masks.shape[1]!=k or masks.dtype.kind!="b" or not len(masks) or not masks.any(1).all():
        raise ValueError("Invalid intervention combinations")
    if len(np.unique(masks,axis=0))!=len(masks):raise ValueError("Duplicate combinations")
    if any(not compatible(executor,[a for a,active in zip(addresses,m) if active]) for m in masks):
        raise ValueError("Structurally incompatible combination")
    trace=executor.execute(x);values=[];visits=[]
    for a in addresses:
        value=np.asarray(trace.records[a]["value"])
        if value.shape!=(len(x),):raise ValueError("Per-world intermediate required")
        values.append(value);visits.append(np.broadcast_to(trace.records[a]["visited"],(len(x),)))
    rng=np.random.default_rng(seed);order=rng.permutation(len(x));mask_order=rng.permutation(len(masks))
    accepted=[];attempts=[];cursor=0
    while len(accepted)<count:
        mask_index=int(mask_order[len(attempts)%len(mask_order)])
        mask=masks[mask_index];active=np.flatnonzero(mask);cost=1+len(active)
        if cursor+cost>len(order):break
        ids=order[cursor:cursor+cost];cursor+=cost
        base=int(ids[0]);sources=np.full(k,base,int);sources[active]=ids[1:]
        source_visited=np.array([visits[j][sources[j]] for j in range(k)],bool)
        patches={addresses[j]:values[j][sources[j]:sources[j]+1] for j in active}
        patched=executor.execute(x[base:base+1],patches)
        target_visited=np.array([bool(np.asarray(patched.records[a]["visited"]).reshape(-1)[0])
            if a in patched.records else False for a in addresses])
        eligible=bool((source_visited[active]&target_visited[active]).all())
        attempts.append({"world_indices":ids.tolist(),"mask_index":mask_index,"accepted":eligible})
        if eligible:
            accepted.append((base,sources,mask.copy(),int(patched.output[0]),int(trace.output[base]),
                             source_visited,target_visited))
    if accepted:
        columns=list(zip(*accepted))
        pairs=JointPairs(*(np.asarray(v) for v in columns))
    else:
        pairs=JointPairs(np.empty(0,int),np.empty((0,k),int),np.empty((0,k),bool),
            np.empty(0,int),np.empty(0,int),np.empty((0,k),bool),np.empty((0,k),bool))
    record={"requested_pairs":count,"accepted_pairs":len(accepted),"attempted_groups":len(attempts),
            "worlds_consumed":cursor,"available_worlds":len(x),"attempts":attempts,"seed":seed,
            "addresses":list(addresses),"combinations":masks.tolist(),
            "neural_or_truth_filtering":False,"rejected_worlds_reused":False,
            "distribution":"symbolic execution-conditioned; not unconditional SCM risk",
            "budget_exhausted":len(accepted)<count}
    return pairs,record

def numeric_frontier(executor):
    """Numeric leaf operands, not whole expression/predicate aliases."""
    paths=[p for p,m in executor.catalog.items() if m["kind"]=="vector" and m["op"]=="var"]
    return [executor.address(p) for p in paths]
