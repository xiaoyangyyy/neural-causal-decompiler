"""Candidate-independent causal support for the 14 discovery features."""
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np

from .io import save_json,read_json
from .rules import Rule
from .statistics import FEATURES
from .raw_program_trace import RawDiscoveryExecutor,feature_root_addresses
from .numeric_audit import (numeric_values,fit_readout_values,intervention_numeric_targets,
    audit_numeric_values,audit_oblique_numeric_values,NumericReadout)
from .mechanism_pairs import compatible_combinations,execution_conditioned_pairs
from .fixed_numeric_mapping import train_fixed_numeric_mapping
from .oblique_mapping import train_fixed_oblique_mapping,biorthogonal_bases
from .joint_alignment import measure_joint_mapping
from .neural_sites import SITES,SiteDecoder


@dataclass
class FeatureGuidanceConfig:
    seed:int=1393
    steps:int=120
    train_pairs:int=384
    validation_pairs:int=768
    def validate(self):
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in ("seed","steps","train_pairs","validation_pairs")):
            raise ValueError("Invalid feature-guidance budget")


def _constant_executor():
    return RawDiscoveryExecutor(Rule(tuple(FEATURES),{"label":0}))


def _save_pairs(path,pairs,targets,visited):
    np.savez_compressed(path,base=pairs.base,sources=pairs.sources,masks=pairs.masks,target=pairs.target,
        initial=pairs.initial,source_visited=pairs.source_visited,target_visited=pairs.target_visited,
        numeric_targets=targets,numeric_visited=visited)


def _weighted(row,kind,key="nmse"):
    value=row[kind]
    return value[key] if value["n"] and value[key] is not None else None


def feature_support(oblique,shuffled,random):
    """Per-feature control-corrected intervention support in [0,1]."""
    result=[]
    for o,s,r in zip(oblique["per_node"],shuffled["per_node"],random["per_node"]):
        error=_weighted(o,"targeted");baseline=_weighted(o,"targeted","no_intervention_nmse")
        controls=[_weighted(s,"targeted"),_weighted(r,"targeted")]
        collateral=_weighted(o,"collateral")
        if any(v is None or not np.isfinite(v) for v in [error,baseline,collateral,*controls]) or o["natural_nmse"] is None:
            result.append(0.);continue
        reference=min(baseline,*controls)
        gain=max(0.,(reference-error)/(1.+reference))
        result.append(float(gain/(1.+o["natural_nmse"])/(1.+collateral)))
    return np.asarray(result)


def fit_causal_feature_guidance(model,fit_data,validation_data,directory,c):
    c.validate();root=Path(directory);root.mkdir(parents=True,exist_ok=False);save_json(root/"config.json",asdict(c))
    executor=_constant_executor();addresses=feature_root_addresses(executor)
    fit_trace=executor.execute(fit_data);val_trace=executor.execute(validation_data)
    np.testing.assert_allclose(fit_trace.features,np.column_stack([fit_trace.records[a]["value"] for a in addresses]))
    natural_fit=numeric_values(executor,fit_data,addresses);natural_val=numeric_values(executor,validation_data,addresses)
    singles=np.eye(len(addresses),dtype=bool);masks=compatible_combinations(executor,addresses,max_order=2)
    train,train_record=execution_conditioned_pairs(executor,fit_data,addresses,singles,count=c.train_pairs,seed=c.seed+4000)
    train_targets,train_visited=intervention_numeric_targets(executor,fit_data,addresses,train)
    val,val_record=execution_conditioned_pairs(executor,validation_data,addresses,masks,count=c.validation_pairs,seed=c.seed+5000)
    if len(val.base)!=c.validation_pairs or len(np.unique(val.masks,axis=0))!=len(masks):
        raise ValueError("Feature-guidance validation coverage budget exhausted")
    val_targets,val_visited=intervention_numeric_targets(executor,validation_data,addresses,val)
    _save_pairs(root/"train_pairs.npz",train,train_targets,train_visited)
    _save_pairs(root/"validation_pairs.npz",val,val_targets,val_visited)
    save_json(root/"pairing.json",{"train":train_record,"validation":val_record,"compatible_masks":masks.tolist()})
    summary={"features":list(FEATURES),"addresses":addresses,"sites":{}}
    for site in SITES:
        decoder=SiteDecoder(model,site);hfit=decoder.extract(fit_data);hval=decoder.extract(validation_data)
        folder=root/site;folder.mkdir();probe=fit_readout_values(hfit,addresses,*natural_fit);save_json(folder/"probe.json",probe.to_dict())
        orth,training=train_fixed_numeric_mapping(decoder,hfit,train,train_targets,train_visited,probe,steps=c.steps,
            seed=c.seed+3000,numeric_weight=1.,behavioral_weight=0.)
        save_json(folder/"orthogonal_training.json",training)
        methods={}
        for name,shuffle,offset,warm in (("oblique",False,0,orth.reshape(hfit.shape[1],-1)),
                                          ("shuffled",True,1000,None)):
            read,write,record=train_fixed_oblique_mapping(decoder,hfit,train,train_targets,train_visited,probe,steps=c.steps,
                seed=c.seed+offset,numeric_weight=1.,behavioral_weight=0.,shuffle_targets=shuffle,initial_write=warm)
            methods[name]=(read,write);save_json(folder/(name+"_training.json"),record)
        rng=np.random.default_rng(c.seed+2000)
        methods["random"]=biorthogonal_bases(rng.normal(size=(hfit.shape[1],len(addresses))),len(addresses),1)
        packed={"orthogonal":orth}
        for name,(read,write) in methods.items():packed[name+"_read"]=read;packed[name+"_write"]=write
        np.savez_compressed(folder/"mappings.npz",**packed)
        audits={"orthogonal":audit_numeric_values(probe,hval,orth,val,*natural_val,val_targets,val_visited)}
        for name,(read,write) in methods.items():
            audits[name]=audit_oblique_numeric_values(probe,hval,read,write,val,*natural_val,val_targets,val_visited)
        support=feature_support(audits["oblique"],audits["shuffled"],audits["random"])
        for name,audit in audits.items():save_json(folder/(name+"_validation.json"),audit)
        save_json(folder/"support.json",{"raw":support.tolist(),"mean":float(support.mean())})
        summary["sites"][site]={"support":support.tolist(),"mean_support":float(support.mean())}
    chosen=max(SITES,key=lambda s:(summary["sites"][s]["mean_support"],-SITES.index(s)))
    raw=np.asarray(summary["sites"][chosen]["support"]);normalized=raw/raw.max() if raw.max()>0 else raw
    summary.update({"selected_site":chosen,"raw_support":raw.tolist(),"normalized_support":normalized.tolist(),
        "positive_features":int((raw>0).sum()),"selection_uses_validation_only":True,
        "claim":"candidate-independent empirical feature intervention support; not causal truth"})
    save_json(root/"summary.json",summary)
    return normalized,summary
