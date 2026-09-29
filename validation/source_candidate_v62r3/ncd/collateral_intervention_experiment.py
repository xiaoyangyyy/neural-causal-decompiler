"""Collateral-preserving tangent intervention experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
import torch
from .io import save_json,read_json,digest
from .model import load_model,set_seed
from .rules import Rule
from .worlds import generate_worlds,save_dataset,load_worlds
from .raw_program_trace import RawDiscoveryExecutor
from .numeric_audit import fit_readout_values,intervention_numeric_targets
from .mechanism_pairs import compatible_combinations,execution_conditioned_pairs
from .fixed_numeric_mapping import train_fixed_numeric_mapping
from .oblique_mapping import train_fixed_oblique_mapping,patch_oblique,validate_oblique
from .joint_alignment import frozen_teacher
from .neural_sites import SITES,SiteDecoder
from .manifold_intervention import fit_tangent_subspace,random_subspace,patch_tangent
from .collateral_tangent import patch_collateral_tangent
from .raw_numeric_experiment import _save_pairs,_load_pairs,_trace_values,_schema
from .quadratic_intervention_experiment import _addresses


@dataclass
class CollateralInterventionConfig:
    seed:int=3593;samples:int=96;fit_worlds:int=1024;test_worlds:int=8192
    train_pairs:int=384;test_pairs:int=2048;steps:int=120;rank:int=1
    numeric_weight:float=.5;variance_threshold:float=.95
    @classmethod
    def quick(cls):return cls(seed=3592,samples=32,fit_worlds=128,test_worlds=384,train_pairs=48,test_pairs=96,steps=8)
    def validate(self):
        names=("seed","samples","fit_worlds","test_worlds","train_pairs","test_pairs","steps","rank")
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in names):
            raise ValueError("Invalid collateral intervention budget")
        if self.samples<16 or self.numeric_weight<0 or not np.isfinite(self.numeric_weight) or not 0<self.variance_threshold<=1:
            raise ValueError("Invalid collateral intervention config")


def _numeric(readout,h,patched,pairs,original,original_visited,targets,visited):
    predicted=readout.predict(patched);before=readout.predict(h[pairs.base]);natural=readout.predict(h)
    rows=[]
    for j,address in enumerate(readout.addresses):
        variance=float(readout.target_variance[j]);natural_ids=original_visited[:,j]
        def score(mask):
            if not mask.any():return {"n":0,"mse":None,"nmse":None,"no_intervention_nmse":None}
            mse=float(np.mean((predicted[mask,j]-targets[mask,j])**2))
            return {"n":int(mask.sum()),"mse":mse,"nmse":mse/variance,
                    "no_intervention_nmse":float(np.mean((before[mask,j]-targets[mask,j])**2)/variance)}
        rows.append({"address":address,
            "natural_nmse":float(np.mean((natural[natural_ids,j]-original[natural_ids,j])**2)/variance) if natural_ids.any() else None,
            "natural_n":int(natural_ids.sum()),"fit_variance":variance,
            "targeted":score(visited[:,j]&pairs.masks[:,j]),
            "collateral":score(visited[:,j]&~pairs.masks[:,j])})
    return {"status":"measured","pairs":len(pairs.base),"per_node":rows,
            "claim":"held-out numerical intervention audit, not unique mechanism identification"}


def _behavioral(decoder,h,patched,pairs):
    with frozen_teacher(decoder),torch.no_grad():
        initial=decoder.logits(h[pairs.base]).argmax(1).numpy()
        after=decoder.logits(patched).argmax(1).numpy()
    executed=(~pairs.masks|(pairs.source_visited & pairs.target_visited)).all(1)
    meaningful=(initial==pairs.initial)&(pairs.target!=pairs.initial)&executed
    return {"pairs":len(after),"all_active_intermediates_executed":int(executed.sum()),
        "symbolic_changes":int((pairs.target!=pairs.initial).sum()),
        "baseline_agreement":float(np.mean(initial==pairs.initial)),
        "interchange_accuracy":float(np.mean(after==pairs.target)),
        "no_intervention_accuracy":float(np.mean(initial==pairs.target)),
        "informative_pairs":int(meaningful.sum()),
        "informative_accuracy":float(np.mean(after[meaningful]==pairs.target[meaningful])) if meaningful.any() else None}


def _inactive_rms(patched,base,read,masks):
    total=0.;count=0
    for mask in np.unique(masks,axis=0):
        ids=np.flatnonzero(np.all(masks==mask,axis=1));inactive=np.flatnonzero(~mask)
        if not len(inactive):continue
        values=(patched[ids]-base[ids])@read[:,inactive,:].reshape(base.shape[1],-1)
        total+=float(np.sum(values*values));count+=values.size
    return float(np.sqrt(total/count)) if count else 0.


def _method(readout,h,patched,pairs,natural,targets,visited,decoder,diagnostics):
    return {"numeric":_numeric(readout,h,patched,pairs,*natural,targets,visited),
            "behavioral":_behavioral(decoder,h,patched,pairs),"diagnostics":diagnostics}


def _execute_site(root,site,hf,ht,addresses,natural,train,train_targets,train_visited,
                  test,test_targets,test_visited,c,save):
    folder=root/"sites"/site
    if save:folder.mkdir(parents=True)
    readout=fit_readout_values(hf,addresses,*natural["alignment_fit"])
    tangent=fit_tangent_subspace(hf,c.variance_threshold)
    random=random_subspace(hf.shape[1],tangent.basis.shape[1],c.seed+12000+SITES.index(site))
    decoder=SiteDecoder(load_model(root/"teacher.pt"),site)
    initial,orth_record=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,readout,
        rank=c.rank,steps=c.steps,seed=c.seed+3000,numeric_weight=c.numeric_weight)
    read,write,training=train_fixed_oblique_mapping(decoder,hf,train,train_targets,train_visited,readout,
        rank=c.rank,steps=c.steps,seed=c.seed,numeric_weight=c.numeric_weight,initial_write=initial.reshape(hf.shape[1],-1))
    validate_oblique(read,write)
    if save:
        save_json(folder/"probe.json",readout.to_dict());save_json(folder/"tangent.json",tangent.to_dict())
        save_json(folder/"orthogonal_training.json",orth_record);save_json(folder/"oblique_training.json",training)
        np.savez_compressed(folder/"mapping.npz",read=read,write=write,random_basis=random)
    else:
        if readout.to_dict()!=read_json(folder/"probe.json") or tangent.to_dict()!=read_json(folder/"tangent.json"):
            raise ValueError("Collateral probe/subspace replay mismatch")
        if orth_record!=read_json(folder/"orthogonal_training.json") or training!=read_json(folder/"oblique_training.json"):
            raise ValueError("Collateral training replay mismatch")
        with np.load(folder/"mapping.npz") as z:
            np.testing.assert_allclose(z["read"],read,rtol=0,atol=1e-7)
            np.testing.assert_allclose(z["write"],write,rtol=0,atol=1e-7)
            np.testing.assert_allclose(z["random_basis"],random,rtol=0,atol=1e-12)
    base=ht[test.base];sources=ht[test.sources]
    ordinary=patch_oblique(base,sources,read,write,test.masks)
    constrained,cdiag=patch_tangent(base,sources,read,test.masks,tangent.basis)
    cdiag["inactive_coordinate_rms"]=_inactive_rms(constrained,base,read,test.masks)
    random_states,rdiag=patch_tangent(base,sources,read,test.masks,random)
    preserved,pdiag=patch_collateral_tangent(base,sources,read,test.masks,tangent.basis)
    ordinary_delta=np.linalg.norm(ordinary-base,axis=1)
    odiag={"mean_displacement_norm":float(ordinary_delta.mean()),"max_displacement_norm":float(ordinary_delta.max()),
           **validate_oblique(read,write)}
    artifacts={
        "ordinary_biorthogonal":_method(readout,ht,ordinary,test,natural["alignment_test"],test_targets,test_visited,decoder,odiag),
        "natural_pca":_method(readout,ht,constrained,test,natural["alignment_test"],test_targets,test_visited,decoder,cdiag),
        "collateral_preserving_pca":_method(readout,ht,preserved,test,natural["alignment_test"],test_targets,test_visited,decoder,pdiag),
        "random_subspace":_method(readout,ht,random_states,test,natural["alignment_test"],test_targets,test_visited,decoder,rdiag)}
    meta={"rank":int(tangent.basis.shape[1]),"explained_variance":tangent.explained_variance,
          "variance_threshold":tangent.threshold,"hidden_dimension":int(hf.shape[1])}
    if save:
        for name,value in artifacts.items():save_json(folder/(name+".json"),value)
        save_json(folder/"subspace_summary.json",meta)
    else:
        for name,value in artifacts.items():
            if value!=read_json(folder/(name+".json")):raise ValueError("Collateral audit replay mismatch")
        if meta!=read_json(folder/"subspace_summary.json"):raise ValueError("Collateral metadata replay mismatch")
    return {"subspace":meta,"methods":{name:{"numeric_nodes":a["numeric"]["per_node"],
        "behavioral":a["behavioral"],"diagnostics":a["diagnostics"]} for name,a in artifacts.items()}}


def run_collateral_intervention(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty collateral intervention directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"});set_seed(config.seed)
    try:
        shutil.copy2(source/"model"/"discoverer.pt",root/"teacher.pt")
        shutil.copy2(source/"program.json",root/"program.json")
        cfg={**asdict(config),"source":str(source),"source_model_sha256":digest(source/"model"/"discoverer.pt"),
             "source_program_sha256":digest(source/"program.json"),"sites":list(SITES),
             "scope":"all 54 scalar groups; ordinary, minimum-norm tangent, and collateral-preserving tangent writes"}
        save_json(root/"config.json",cfg);model=load_model(root/"teacher.pt")
        executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/"program.json")),trace_dependence=True,trace_regression=True)
        addresses=_addresses(executor);data={};natural={};seen=set()
        for split,count in (("alignment_fit",config.fit_worlds),("alignment_test",config.test_worlds)):
            worlds=generate_worlds(split,count,config.seed,config.samples);ids={w.identity for w in worlds}
            if len(ids)!=count or ids&seen:raise ValueError("Collateral world leakage")
            seen|=ids;save_dataset(root/"datasets"/split,worlds);data[split]=np.stack([w.sample() for w in worlds])
            trace=executor.execute(data[split]);natural[split]=_trace_values(trace,addresses)
            np.savez_compressed(root/"datasets"/split/"raw_trace.npz",features=trace.features,values=natural[split][0],
                                visited=natural[split][1],output=trace.output)
        single=np.eye(len(addresses),dtype=bool);masks=compatible_combinations(executor,addresses,max_order=2)
        train,tr=execution_conditioned_pairs(executor,data["alignment_fit"],addresses,single,count=config.train_pairs,seed=config.seed+4000)
        train_targets,train_visited=intervention_numeric_targets(executor,data["alignment_fit"],addresses,train)
        test,te=execution_conditioned_pairs(executor,data["alignment_test"],addresses,masks,count=config.test_pairs,seed=config.seed+5000)
        if config.test_pairs>=len(masks) and (len(test.base)!=config.test_pairs or len(np.unique(test.masks,axis=0))!=len(masks)):
            raise ValueError("Collateral full mask coverage exhausted")
        test_targets,test_visited=intervention_numeric_targets(executor,data["alignment_test"],addresses,test)
        _save_pairs(root/"train_pairs.npz",train,train_targets,train_visited);_save_pairs(root/"test_pairs.npz",test,test_targets,test_visited)
        save_json(root/"pairing.json",{"train":tr,"test":te})
        summary={"config":cfg,"groups":_schema(executor,addresses),"test_combinations":masks.tolist(),"sites":{}}
        for site in SITES:
            decoder=SiteDecoder(model,site)
            summary["sites"][site]=_execute_site(root,site,decoder.extract(data["alignment_fit"]),
                decoder.extract(data["alignment_test"]),addresses,natural,train,train_targets,train_visited,
                test,test_targets,test_visited,config,True)
        save_json(root/"summary.json",summary);snap=root/"source";snap.mkdir()
        [shutil.copy2(p,snap/p.name) for p in Path(__file__).parent.glob("*.py")]
        save_json(root/"status.json",{"state":"completed"})
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise


def verify_collateral_intervention(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json");cfg=read_json(root/"config.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete manifold run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Collateral artifact mismatch")
    c=CollateralInterventionConfig(**{k:v for k,v in cfg.items() if k in CollateralInterventionConfig.__dataclass_fields__})
    c.validate();set_seed(c.seed)
    if cfg!=summary["config"] or digest(root/"teacher.pt")!=cfg["source_model_sha256"] or digest(root/"program.json")!=cfg["source_program_sha256"]:
        raise ValueError("Collateral source mismatch")
    model=load_model(root/"teacher.pt")
    executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/"program.json")),trace_dependence=True,trace_regression=True)
    addresses=_addresses(executor);data={};natural={};seen=set()
    for split,count in (("alignment_fit",c.fit_worlds),("alignment_test",c.test_worlds)):
        worlds=load_worlds(root/"datasets"/split)
        if worlds!=generate_worlds(split,count,c.seed,c.samples):raise ValueError("Collateral world mismatch")
        ids={w.identity for w in worlds}
        if len(ids)!=count or ids&seen:raise ValueError("Collateral replay leakage")
        seen|=ids;data[split]=np.stack([w.sample() for w in worlds]);trace=executor.execute(data[split]);natural[split]=_trace_values(trace,addresses)
        with np.load(root/"datasets"/split/"raw_trace.npz") as z:
            np.testing.assert_allclose(z["features"],trace.features,atol=1e-10)
            np.testing.assert_allclose(z["values"],natural[split][0],atol=1e-10)
            np.testing.assert_array_equal(z["visited"],natural[split][1]);np.testing.assert_array_equal(z["output"],trace.output)
    if _schema(executor,addresses)!=summary["groups"]:raise ValueError("Collateral schema mismatch")
    single=np.eye(len(addresses),dtype=bool);masks=compatible_combinations(executor,addresses,max_order=2)
    train,tr=execution_conditioned_pairs(executor,data["alignment_fit"],addresses,single,count=c.train_pairs,seed=c.seed+4000)
    train_targets,train_visited=intervention_numeric_targets(executor,data["alignment_fit"],addresses,train)
    test,te=execution_conditioned_pairs(executor,data["alignment_test"],addresses,masks,count=c.test_pairs,seed=c.seed+5000)
    test_targets,test_visited=intervention_numeric_targets(executor,data["alignment_test"],addresses,test)
    if {"train":tr,"test":te}!=read_json(root/"pairing.json"):raise ValueError("Collateral pairing mismatch")
    for path,expected in ((root/"train_pairs.npz",(train,train_targets,train_visited)),(root/"test_pairs.npz",(test,test_targets,test_visited))):
        actual=_load_pairs(path)
        for a,b in zip(actual[0].__dict__.values(),expected[0].__dict__.values()):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(actual[1],expected[1]);np.testing.assert_array_equal(actual[2],expected[2])
    for site in SITES:
        decoder=SiteDecoder(model,site)
        actual=_execute_site(root,site,decoder.extract(data["alignment_fit"]),decoder.extract(data["alignment_test"]),
            addresses,natural,train,train_targets,train_visited,test,test_targets,test_visited,c,False)
        if actual!=summary["sites"][site]:raise ValueError("Collateral summary mismatch")
    return {"status":"verified","worlds":len(seen),"groups":len(addresses),"sites":list(SITES),
            "train_pairs":len(train.base),"test_pairs":len(test.base),"science_not_certified":True}
