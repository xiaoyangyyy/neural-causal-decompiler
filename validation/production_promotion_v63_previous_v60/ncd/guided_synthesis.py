"""Choose programs and neural sites using extraction/fit/validation worlds only.

This is finite candidate reranking, not an unrestricted whole-program search.
All controls use the same validation pairs as their candidate/site.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import torch
from .io import save_json
from .worlds import save_dataset
from .model import predict
from .cdir import composed_features
from .statistics import FEATURES
from .rules import fit_rule
from .program_trace import ProgramExecutor
from .neural_sites import SITES,SiteDecoder
from .mechanism_pairs import numeric_frontier,compatible_combinations,execution_conditioned_pairs
from .numeric_audit import numeric_values,fit_readout,audit_numeric
from .numeric_mapping import train_numeric_mapping
from .joint_alignment import frozen_teacher,measure_joint_mapping

@dataclass
class GuidedConfig:
    seed:int=391
    steps:int=40
    max_splits:int=6
    beam_width:int=3
    penalties:tuple=(0.,.001,.005)
    complexity_weight:float=.001
    mechanism_weight:float=.05
    min_informative:int=8

def average_error(nodes,category,key):
    rows=[n[category] for n in nodes if n[category]["n"] and n[category][key] is not None]
    count=sum(r["n"] for r in rows)
    return sum(r[key]*r["n"] for r in rows)/count if count else None

def mechanism_support(numeric,behavioral,min_informative=8):
    if numeric["status"]!="measured" or behavioral.get("overall",{}).get("informative_pairs",0)<min_informative:return 0.
    nodes=numeric["per_node"]
    error=average_error(nodes,"targeted","nmse");baseline=average_error(nodes,"targeted","no_intervention_nmse")
    collateral=average_error(nodes,"collateral","nmse")
    natural=[r for r in nodes if r["natural_n"] and r["natural_nmse"] is not None]
    n=sum(r["natural_n"] for r in natural)
    if error is None or baseline is None or not n:return 0.
    natural_error=sum(r["natural_nmse"]*r["natural_n"] for r in natural)/n
    gain=max(0.,(baseline-error)/(1.+baseline))
    preservation=1./(1.+(collateral if collateral is not None else 0.))
    accuracy=behavioral["overall"]["informative_accuracy"] or 0.
    return float(gain*preservation*accuracy/(1.+natural_error))

def select_records(records,c):
    if not records:raise ValueError("No candidate programs")
    def base(r):return r["validation_fidelity"]-c.complexity_weight*r["complexity"]
    plain=max(records,key=lambda r:(base(r),-r["complexity"],r["id"]))
    guided=max(records,key=lambda r:(base(r)+c.mechanism_weight*r["mechanism_support"],-r["complexity"],r["id"]))
    return {"without_internal":plain["id"],"with_internal":guided["id"],
            "score_definition":"validation_fidelity - complexity_weight*complexity + mechanism_weight*control_corrected_support",
            "base_scores":{r["id"]:base(r) for r in records},
            "guided_scores":{r["id"]:base(r)+c.mechanism_weight*r["mechanism_support"] for r in records}}

def _check_worlds(extraction,fit,validation):
    seen=set()
    for role,worlds,split in (("extraction",extraction,"extraction"),("fit",fit,"alignment_fit"),("validation",validation,"alignment_fit")):
        if not worlds or any(w.split!=split for w in worlds):raise ValueError("Unexpected world role: "+role)
        ids={w.identity for w in worlds}
        if len(ids)!=len(worlds) or ids&seen:raise ValueError("World identity leakage")
        seen|=ids

def _guided_synthesis(model,extraction,fit,validation,directory,c=None):
    c=c or GuidedConfig();_check_worlds(extraction,fit,validation)
    from .model import set_seed
    set_seed(c.seed)
    if c.steps<1 or c.max_splits<1 or c.beam_width<1 or c.min_informative<1:
        raise ValueError("Invalid synthesis budget")
    if not c.penalties or any(not np.isfinite(p) or p<0 for p in c.penalties):
        raise ValueError("Invalid candidate penalties")
    if not np.isfinite(c.mechanism_weight) or c.mechanism_weight<0 or not np.isfinite(c.complexity_weight) or c.complexity_weight<0:
        raise ValueError("Invalid objective weights")
    root=Path(directory)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use empty synthesis directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"config.json",asdict(c))
    data={};features={}
    for role,worlds in (("extraction",extraction),("fit",fit),("validation",validation)):
        save_dataset(root/"datasets"/role,worlds)
        data[role]=np.stack([w.sample() for w in worlds]);features[role]=composed_features(data[role])
        np.savez_compressed(root/"datasets"/role/"features.npz",features=features[role])
    with frozen_teacher(model):
        probabilities={role:predict(model,value) for role,value in data.items()}
        torch.save({"width":model.width,"state_dict":model.state_dict()},root/"teacher.pt")
        y=probabilities["extraction"].argmax(1);vy=probabilities["validation"].argmax(1)
        candidates={};origins={}
        for penalty in c.penalties:
            rule,trace=fit_rule(features["extraction"],y,FEATURES,penalty=penalty,beam_width=c.beam_width,max_splits=c.max_splits)
            identifier=ProgramExecutor(rule).program_id
            candidates[identifier]=rule;origins.setdefault(identifier,[]).append(penalty)
        hidden={}
        for site in SITES:
            decoder=SiteDecoder(model,site)
            hidden[site]={role:decoder.extract(data[role]) for role in ("fit","validation")}
        records=[]
        for identifier,rule in candidates.items():
            print("mechanism-guided candidate "+identifier[:12],flush=True)
            path=root/"candidates"/identifier;path.mkdir(parents=True)
            save_json(path/"program.json",rule.to_dict())
            executor=ProgramExecutor(rule);frontier=numeric_frontier(executor)
            record={"id":identifier,"complexity":rule.complexity,"origin_penalties":origins[identifier],
                "validation_fidelity":float(np.mean(rule.predict(features["validation"])==vy)),
                "mechanism_support":0.,"selected_site":None,"sites":{}}
            if frontier:
                values,visited=numeric_values(executor,features["fit"],frontier)
                addresses=[a for j,a in enumerate(frontier) if visited[:,j].sum()>=8 and values[visited[:,j],j].var()>1e-8]
            else:addresses=[]
            record["supported_addresses"]=addresses;record["unsupported_addresses"]=[a for a in frontier if a not in addresses]
            if len(addresses)<2:
                record["mechanism_status"]="insufficient_numeric_operands";records.append(record);continue
            masks=compatible_combinations(executor,addresses)
            pairs,pairing=execution_conditioned_pairs(executor,features["validation"],addresses,masks,count=1024,seed=c.seed+900)
            save_json(path/"validation_pairs.json",pairing)
            if not len(pairs.base):
                record["mechanism_status"]="no_executed_validation_pairs";records.append(record);continue
            record["mechanism_status"]="measured"
            for site in SITES:
                decoder=SiteDecoder(model,site);hfit=hidden[site]["fit"];hval=hidden[site]["validation"]
                probe=fit_readout(hfit,executor,features["fit"],addresses)
                target=path/site;target.mkdir();save_json(target/"probe.json",probe.to_dict())
                bases={}
                for name,shuffled,offset in (("fitted",False,0),("shuffled",True,1000)):
                    q,training=train_numeric_mapping(decoder,hfit,features["fit"],executor,addresses,
                        np.eye(len(addresses),dtype=bool),probe,steps=c.steps,seed=c.seed+offset,
                        numeric_weight=.5,shuffle_targets=shuffled)
                    bases[name]=q;save_json(target/(name+"_training.json"),training)
                rng=np.random.default_rng(c.seed+2000);shape=bases["fitted"].shape
                bases["random"]=np.linalg.qr(rng.normal(size=(shape[0],len(addresses))))[0].reshape(shape)
                np.savez_compressed(target/"bases.npz",**bases)
                supports={}
                for name,basis in bases.items():
                    numeric=audit_numeric(probe,executor,features["validation"],hval,basis,pairs)
                    behavioral=measure_joint_mapping(decoder,hval,basis,pairs)
                    support=mechanism_support(numeric,behavioral,c.min_informative);supports[name]=support
                    save_json(target/(name+"_validation.json"),{"numeric":numeric,"behavioral":behavioral,"support":support})
                corrected=max(0.,supports["fitted"]-max(supports["random"],supports["shuffled"]))
                record["sites"][site]={"raw_support":supports,"corrected_support":corrected}
            # Deterministic site choice depends only on this validation partition.
            site=max(SITES,key=lambda site:(record["sites"][site]["corrected_support"],-SITES.index(site)))
            record["mechanism_support"]=record["sites"][site]["corrected_support"]
            record["selected_site"]=site if record["mechanism_support"]>0 else None
            records.append(record)
    selection=select_records(records,c)
    selection.update({"candidates":records,"world_ids":{r:[w.identity for w in ws] for r,ws in
                     (("extraction",extraction),("fit",fit),("validation",validation))},
        "test_worlds_accessed":False,"teacher_labels_only":True,"teacher_frozen":True,
        "comparison":"same candidate programs and validation worlds; internal method uses extra computation",
        "search_scope":"finite penalty-generated candidate set; selected neural site and support rerank programs"})
    save_json(root/"selection.json",selection)
    return candidates[selection["with_internal"]],candidates[selection["without_internal"]],selection


def guided_synthesis(model,extraction,fit,validation,directory,c=None):
    from .io import digest
    import shutil
    root=Path(directory)
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use empty synthesis directory")
    try:
        result=_guided_synthesis(model,extraction,fit,validation,root,c)
        snapshot=root/"source";snapshot.mkdir()
        for source in Path(__file__).parent.glob("*.py"):shutil.copy2(source,snapshot/source.name)
        save_json(root/"status.json",{"state":"completed"})
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p)
            for p in sorted(root.rglob("*")) if p.is_file()}})
        return result
    except BaseException as exc:
        if root.exists():save_json(root/"status.json",{"state":"failed","error":str(exc)})
        raise

def verify_guided(directory):
    from .io import read_json,digest
    from .worlds import load_worlds
    from .model import load_model,set_seed
    from .rules import Rule
    root=Path(directory).resolve()
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Guided artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete synthesis")
    c=GuidedConfig(**read_json(root/"config.json"));set_seed(c.seed)
    selection=read_json(root/"selection.json")
    worlds={role:load_worlds(root/"datasets"/role) for role in ("extraction","fit","validation")}
    _check_worlds(worlds["extraction"],worlds["fit"],worlds["validation"])
    model=load_model(root/"teacher.pt");data={};features={}
    if {role:[w.identity for w in ws] for role,ws in worlds.items()}!=selection["world_ids"]:raise ValueError("World provenance mismatch")
    for role,ws in worlds.items():
        path=root/"datasets"/role;data[role]=np.stack([w.sample() for w in ws]);features[role]=composed_features(data[role])
        if read_json(path/"worlds.json")!=[w.metadata() for w in ws]:raise ValueError("World metadata mismatch")
        with np.load(path/"samples.npz") as a:np.testing.assert_array_equal(data[role],a["data"])
        with np.load(path/"features.npz") as a:np.testing.assert_allclose(features[role],a["features"],atol=1e-9)
    y=predict(model,data["extraction"]).argmax(1);vy=predict(model,data["validation"]).argmax(1)
    candidates={};origins={}
    for penalty in c.penalties:
        rule,_=fit_rule(features["extraction"],y,FEATURES,penalty=penalty,beam_width=c.beam_width,max_splits=c.max_splits)
        identifier=ProgramExecutor(rule).program_id;candidates[identifier]=rule;origins.setdefault(identifier,[]).append(penalty)
    if list(candidates)!=[r["id"] for r in selection["candidates"]]:raise ValueError("Candidate-generation replay mismatch")
    hidden={}
    for site in SITES:
        decoder=SiteDecoder(model,site);hidden[site]={role:decoder.extract(data[role]) for role in ("fit","validation")}
    records=[]
    for saved in selection["candidates"]:
        identifier=saved["id"];rule=candidates[identifier];path=root/"candidates"/identifier
        if rule.to_dict()!=read_json(path/"program.json"):raise ValueError("Program mismatch")
        executor=ProgramExecutor(rule);frontier=numeric_frontier(executor);addresses=[]
        if frontier:
            values,visited=numeric_values(executor,features["fit"],frontier)
            addresses=[a for j,a in enumerate(frontier) if visited[:,j].sum()>=8 and values[visited[:,j],j].var()>1e-8]
        record={"id":identifier,"complexity":rule.complexity,"origin_penalties":origins[identifier],
                "validation_fidelity":float(np.mean(rule.predict(features["validation"])==vy)),
                "mechanism_support":0.,"selected_site":None,"sites":{},
                "supported_addresses":addresses,"unsupported_addresses":[a for a in frontier if a not in addresses]}
        if len(addresses)<2:record["mechanism_status"]="insufficient_numeric_operands"
        else:
            masks=compatible_combinations(executor,addresses)
            pairs,pairing=execution_conditioned_pairs(executor,features["validation"],addresses,masks,count=1024,seed=c.seed+900)
            if pairing!=read_json(path/"validation_pairs.json"):raise ValueError("Validation pairing mismatch")
            if not len(pairs.base):record["mechanism_status"]="no_executed_validation_pairs"
            else:
                record["mechanism_status"]="measured"
                for site in SITES:
                    target=path/site;decoder=SiteDecoder(model,site);hfit=hidden[site]["fit"];hval=hidden[site]["validation"]
                    probe=fit_readout(hfit,executor,features["fit"],addresses)
                    if probe.to_dict()!=read_json(target/"probe.json"):raise ValueError("Readout provenance mismatch")
                    with np.load(target/"bases.npz") as a:bases={k:a[k] for k in a.files}
                    if set(bases)!={"fitted","shuffled","random"}:raise ValueError("Control coverage mismatch")
                    rng=np.random.default_rng(c.seed+2000);shape=bases["fitted"].shape
                    np.testing.assert_array_equal(bases["random"],np.linalg.qr(rng.normal(size=(shape[0],len(addresses))))[0].reshape(shape))
                    supports={}
                    for name,basis in bases.items():
                        numeric=audit_numeric(probe,executor,features["validation"],hval,basis,pairs)
                        behavioral=measure_joint_mapping(decoder,hval,basis,pairs)
                        support=mechanism_support(numeric,behavioral,c.min_informative);supports[name]=support
                        if {"numeric":numeric,"behavioral":behavioral,"support":support}!=read_json(target/(name+"_validation.json")):raise ValueError("Internal-score replay mismatch")
                    record["sites"][site]={"raw_support":supports,
                        "corrected_support":max(0.,supports["fitted"]-max(supports["random"],supports["shuffled"]))}
                site=max(SITES,key=lambda site:(record["sites"][site]["corrected_support"],-SITES.index(site)))
                record["mechanism_support"]=record["sites"][site]["corrected_support"]
                record["selected_site"]=site if record["mechanism_support"]>0 else None
        if record!=saved:raise ValueError("Candidate-record replay mismatch")
        records.append(record)
    expected=select_records(records,c)
    for key,value in expected.items():
        if selection[key]!=value:raise ValueError("Final selection replay mismatch")
    return {"status":"verified","worlds":sum(len(ws) for ws in worlds.values()),"candidates":len(records),
            "selection_changed":selection["with_internal"]!=selection["without_internal"],
            "positive_internal_support":sum(r["mechanism_support"]>0 for r in records),
            "scope":"fit/validation program selection; final test effectiveness not established"}
