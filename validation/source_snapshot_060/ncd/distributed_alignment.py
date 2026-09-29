"""Learn distributed interchange mappings while keeping the teacher frozen.

Mapping optimization uses independent alignment worlds, and is evaluated on
unseen worlds/source pairs. Multiple random subspaces are retained as controls.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .alignment import hidden,logits_from_hidden
from .model import set_seed
from .statistics import FEATURES
from .io import save_json

def patch_subspace(base,source,basis):
    return base+(source-base)@basis@basis.T

def program_features(program):
    if hasattr(program,"used_features"):return sorted(program.used_features())
    # Generic Rule AST uses vector variable indices referring to feature schema.
    used=set()
    def visit_expr(e):
        if e["op"]=="var":used.add(program.names[e["index"]])
        for a in e.get("args",[]):visit_expr(a)
    def visit(t):
        if "label" not in t:visit_expr(t["expr"]);visit(t["left"]);visit(t["right"])
    visit(program.tree)
    return sorted(used)

def pairs(features,program,feature,count,rng):
    base=rng.integers(0,len(features),count);offset=rng.integers(1,len(features),count)
    source=(base+offset)%len(features)
    intervened=features[base].copy();intervened[:,feature]=features[source,feature]
    return base,source,program.predict(intervened)

def train_mapping(model,h,features,program,feature,*,rank=4,steps=160,seed=42,shuffle_targets=False):
    set_seed(seed);rng=np.random.default_rng(seed)
    ht=torch.tensor(h,dtype=torch.float32);d=h.shape[1]
    parameter=nn.Parameter(torch.randn(d,rank)*.1)
    optimizer=torch.optim.Adam([parameter],lr=.025)
    baseline=program.predict(features)
    for _ in range(steps):
        base,source,target=pairs(features,program,feature,128,rng)
        if shuffle_targets:target=target[rng.permutation(len(target))]
        b=torch.tensor(base);s=torch.tensor(source);y=torch.tensor(target)
        q=torch.linalg.qr(parameter,mode="reduced").Q
        patched=patch_subspace(ht[b],ht[s],q)
        width=d//2;logits=model.from_hidden(patched[:,:width],patched[:,width:])
        weights=torch.tensor(np.where(target!=baseline[base],3.,1.),dtype=torch.float32)
        loss=(nn.functional.cross_entropy(logits,y,reduction="none")*weights).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    return torch.linalg.qr(parameter.detach(),mode="reduced").Q.numpy().astype(float)

def measure_mapping(model,h,features,program,feature,basis,*,pair_count=1024,seed=42,disjoint=False):
    rng=np.random.default_rng(seed)
    if disjoint:
        order=rng.permutation(len(features));base,source=order[:len(order)//2*2:2],order[1:len(order)//2*2:2]
        changed=features[base].copy();changed[:,feature]=features[source,feature];target=program.predict(changed)
    else:base,source,target=pairs(features,program,feature,pair_count,rng)
    initial=program.predict(features[base]);n0=logits_from_hidden(model,h[base])
    patched=patch_subspace(h[base],h[source],basis)
    n1=logits_from_hidden(model,patched)
    informative=(initial==n0.argmax(1))&(target!=initial)
    neural_delta=n1-n0
    symbolic_delta=np.eye(4)[target]-np.eye(4)[initial]
    denominator=np.linalg.norm(neural_delta,axis=1)*np.linalg.norm(symbolic_delta,axis=1)
    cosine=np.divide(np.sum(neural_delta*symbolic_delta,axis=1),denominator,out=np.zeros(len(base)),where=denominator>1e-10)
    return {"pairs":len(base),"interchange_accuracy":float(np.mean(n1.argmax(1)==target)),
        "informative_pairs":int(informative.sum()),
        "informative_accuracy":float(np.mean(n1.argmax(1)[informative]==target[informative])) if informative.any() else None,
        "mean_delta_cosine_on_changes":float(np.mean(cosine[target!=initial])) if np.any(target!=initial) else None,
        "mean_displacement":float(np.linalg.norm(patched-h[base],axis=1).mean()),
        "base_indices":base.tolist(),"source_indices":source.tolist(),"symbolic_target":target.tolist(),
        "neural_after":n1.argmax(1).tolist()}

def distributed_audit(model,program,fit_data,fit_features,test_data,test_features,directory,*,steps=160,controls=10,seed=42):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    for p in model.parameters():p.requires_grad_(False)
    fit_h=hidden(model,fit_data);test_h=hidden(model,test_data)
    cut=int(.7*len(fit_h))
    candidates=[]
    for feature in program_features(program):
        if feature not in FEATURES:continue
        index=FEATURES.index(feature)
        for rank in (1,4,8):
            print(f"distributed mapping: {feature}, rank={rank}",flush=True)
            basis=train_mapping(model,fit_h[:cut],fit_features[:cut],program,index,rank=rank,steps=steps,seed=seed+rank)
            val=measure_mapping(model,fit_h[cut:],fit_features[cut:],program,index,basis,pair_count=512,seed=seed+20)
            score=val["informative_accuracy"]
            candidates.append(((-1 if score is None else score),feature,rank,basis,val))
    if not candidates:return {"status":"no_used_features","mapping_used_in_synthesis":False}
    score,feature,rank,basis,validation=max(candidates,key=lambda item:(item[0],-item[2]))
    idx=FEATURES.index(feature)
    # Do not refit after validation selection: preserve the selected mapping.
    result=measure_mapping(model,test_h,test_features,program,idx,basis,seed=seed+100)
    rng=np.random.default_rng(seed+200);random_results=[];random_bases=[]
    for _ in range(controls):
        q=np.linalg.qr(rng.normal(size=basis.shape))[0]
        random_bases.append(q)
        random_results.append(measure_mapping(model,test_h,test_features,program,idx,q,seed=seed+100))
    shuffled_basis=train_mapping(model,fit_h[:cut],fit_features[:cut],program,idx,rank=rank,steps=steps,
                                 seed=seed+400,shuffle_targets=True)
    shuffled_result=measure_mapping(model,test_h,test_features,program,idx,shuffled_basis,seed=seed+100)
    disjoint_result=measure_mapping(model,test_h,test_features,program,idx,basis,seed=seed+100,disjoint=True)
    from .evidence import fidelity_bound
    disjoint_bound=fidelity_bound((np.array(disjoint_result["neural_after"])==np.array(disjoint_result["symbolic_target"])).astype(float))
    for k,v in model.state_dict().items():
        if not torch.equal(v,original[k]):raise RuntimeError("Teacher was modified during mapping optimization")
    np.savez_compressed(directory/"mapping.npz",basis=basis,controls=np.stack(random_bases),feature=np.array(idx),shuffled_basis=shuffled_basis)
    values=[r["informative_accuracy"] for r in random_results if r["informative_accuracy"] is not None]
    record={"status":"measured","feature":feature,"rank":rank,"validation":validation,"test":result,
        "random_controls":random_results,"shuffled_target_control":shuffled_result,
        "disjoint_test":disjoint_result,"disjoint_fidelity_bound":disjoint_bound,
        "random_informative_mean":float(np.mean(values)) if values else None,
        "test_exceeds_all_random":bool(values and result["informative_accuracy"] is not None and result["informative_accuracy"]>max(values)),
        "candidates":[{"feature":f,"rank":r,"validation_informative_accuracy":v["informative_accuracy"]} for _,f,r,_,v in candidates],
        "frozen_teacher_verified":True,"claim":"empirical distributed alignment, not unique algorithm recovery",
        "mapping_used_in_synthesis":False}
    save_json(directory/"result.json",record)
    return record

def fit_guidance(model,program,fit_data,fit_features,*,steps=80,seed=42):
    """Fit-only circuit scores for synthesis. Never takes evaluation data."""
    for p in model.parameters():p.requires_grad_(False)
    h=hidden(model,fit_data);cut=int(.7*len(h))
    guidance=np.zeros(len(FEATURES));records=[]
    for name in program_features(program):
        if name not in FEATURES:continue
        index=FEATURES.index(name)
        basis=train_mapping(model,h[:cut],fit_features[:cut],program,index,rank=4,steps=steps,seed=seed)
        validation=measure_mapping(model,h[cut:],fit_features[cut:],program,index,basis,pair_count=512,seed=seed+1)
        score=validation["informative_accuracy"]
        guidance[index]=0 if score is None else score
        records.append({"feature":name,"score":float(guidance[index]),"basis":basis.tolist(),"validation":validation})
    return guidance,{"records":records,"fit_rows":cut,"validation_rows":len(h)-cut,
                     "evaluation_data_accessed":False,"used_for_synthesis":True}
