"""Budgeted evolutionary search over valid SCM parameters.

Teacher mismatch and shared causal error are separate objectives. Every unique
candidate costs one teacher query; the random control uses the same budget.
"""
from dataclasses import replace
import numpy as np
from .statistics import extract
from .model import predict
from .worlds import World

def mutate(world,rng,scale=.6):
    coefficient=float(np.sign(world.coefficient)*np.clip(abs(world.coefficient)*np.exp(rng.normal(0,scale)),.2,2.))
    return replace(world,scale_x=float(np.clip(world.scale_x*np.exp(rng.normal(0,scale)),.05,20)),
        scale_y=float(np.clip(world.scale_y*np.exp(rng.normal(0,scale)),.05,20)),
        coefficient=coefficient,noise_scale=float(np.clip(world.noise_scale*np.exp(rng.normal(0,scale)),.08,1.5)),
        split="active_search")

def search_scm(model,program,seeds,*,objective="fidelity",budget=96,population=8,seed=42,guided=True,feature_extractor=extract):
    if objective not in ("fidelity","joint_error") or budget<1 or not seeds:raise ValueError("Invalid active search")
    rng=np.random.default_rng(seed);pending=list(seeds)[:min(population,budget)]
    origins={w.identity:w.identity for w in pending}
    all_records=[];seen=set();elite=list(pending);elite_scores=np.zeros(len(elite))
    while len(all_records)<budget:
        needed=min(population,budget-len(all_records));batch=[]
        for w in pending:
            if w.identity not in seen and len(batch)<needed:batch.append(w);seen.add(w.identity)
        attempts=0
        while len(batch)<needed:
            parent=elite[int(rng.integers(len(elite)))]
            w=mutate(parent,rng);attempts+=1
            if attempts>10000:raise RuntimeError("Search proposal space exhausted")
            if w.identity in seen:continue
            seen.add(w.identity);origins[w.identity]=origins[parent.identity];batch.append(w)
        data=np.stack([w.sample() for w in batch]);features=feature_extractor(data)
        probabilities=predict(model,data);symbolic=program.predict(features);neural=probabilities.argmax(1)
        for w,x,pr,p,n in zip(batch,features,probabilities,symbolic,neural):
            mismatch=int(p!=n);joint=int(p==n and n!=w.label)
            score=float(1-pr[p]) if objective=="fidelity" else float((p!=w.label)*pr[p])
            all_records.append({"world":w.metadata(),"origin":origins.get(w.identity,w.identity),
                "program":int(p),"neural":int(n),"truth":w.label,"probabilities":pr.tolist(),
                "fidelity_error":bool(mismatch),"joint_error":bool(joint),"score":score})
        if guided:
            ranked=sorted(all_records,key=lambda r:r["score"],reverse=True)[:max(2,population//2)]
        else:
            idx=rng.choice(len(all_records),min(len(all_records),max(2,population//2)),replace=False)
            ranked=[all_records[i] for i in idx]
        elite=[World(**{k:v for k,v in r["world"].items() if k in World.__dataclass_fields__}) for r in ranked]
        pending=[]
    return {"objective":objective,"guided":guided,"query_budget":budget,"queries":len(all_records),
        "fidelity_errors":sum(r["fidelity_error"] for r in all_records),
        "joint_errors":sum(r["joint_error"] for r in all_records),
        "records":all_records,"structure_preserved":True,
        "causal_truth_used_for_optimization":objective=="joint_error"}

def replicate_counterexamples(model,program,search_result,*,top=8,replicates=8,seed=1001,feature_extractor=extract):
    rng=np.random.default_rng(seed)
    selected=sorted(search_result["records"],key=lambda r:r["score"],reverse=True)[:top];rows=[]
    for record in selected:
        w=World(**{k:v for k,v in record["world"].items() if k in World.__dataclass_fields__})
        variants=[replace(w,seed=int(rng.integers(0,2**63-1))) for _ in range(replicates)]
        data=np.stack([v.sample() for v in variants]);p=program.predict(feature_extractor(data));n=predict(model,data).argmax(1)
        rows.append({"searched_world":w.identity,"replicate_worlds":[v.metadata() for v in variants],
                     "fidelity_error_rate":float(np.mean(p!=n)),
                     "joint_error_rate":float(np.mean((p==n)&(p!=w.label)))})
    return rows
