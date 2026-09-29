"""Multi-site numerical audit of a fixed, previously trained teacher/program.

All three sites are reported. No site is selected using final test outcomes.
"""
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .model import load_model,set_seed
from .rules import Rule
from .program_trace import ProgramExecutor
from .worlds import generate_worlds,save_dataset,load_worlds
from .cdir import composed_features
from .neural_sites import SITES,SiteDecoder
from .mechanism_pairs import numeric_frontier,compatible_combinations,execution_conditioned_pairs
from .numeric_audit import numeric_values,fit_readout,audit_numeric,NumericReadout
from .numeric_mapping import train_numeric_mapping
from .joint_alignment import measure_joint_mapping

def run_numeric(directory,source,seed=291,quick=False):
    root=Path(directory).resolve();source=Path(source).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use a new numeric audit directory")
    root.mkdir(parents=True,exist_ok=True);set_seed(seed)
    c={"seed":seed,"samples":96,"fit_worlds":512 if quick else 1024,"test_worlds":1024 if quick else 4096,
       "steps":10 if quick else 120,"rank":1,"numeric_weight":.5,"pairs":256 if quick else 1024,
       "source_model_sha256":digest(source/"model"/"discoverer.pt"),"source_program_sha256":digest(source/"program.json"),
       "sites":list(SITES),"quick":quick,"selection":"all sites reported, no final-test selection"}
    save_json(root/"config.json",c);save_json(root/"status.json",{"state":"running"})
    try:
        shutil.copy2(source/"model"/"discoverer.pt",root/"teacher.pt");shutil.copy2(source/"program.json",root/"program.json")
        model=load_model(root/"teacher.pt");executor=ProgramExecutor(Rule.from_dict(read_json(root/"program.json")))
        x={};data={};identities=set()
        for split,n in (("alignment_fit",c["fit_worlds"]),("alignment_test",c["test_worlds"])):
            ws=generate_worlds(split,n,seed,c["samples"])
            ids={w.identity for w in ws}
            if ids&identities:raise ValueError("World leakage")
            identities|=ids;save_dataset(root/"datasets"/split,ws)
            data[split]=np.stack([w.sample() for w in ws]);x[split]=composed_features(data[split])
            np.savez_compressed(root/"datasets"/split/"features.npz",features=x[split])
        frontier=numeric_frontier(executor);values,visited=numeric_values(executor,x["alignment_fit"],frontier)
        supported=[j for j in range(len(frontier)) if visited[:,j].sum()>=8 and values[visited[:,j],j].var()>1e-8]
        addresses=[frontier[j] for j in supported]
        if len(addresses)<2:raise ValueError("Fewer than two supported numeric operands")
        masks=compatible_combinations(executor,addresses)
        # Singles only in fitting; all compatible pairs are unseen combinations.
        fit_masks=np.eye(len(addresses),dtype=bool)
        pairs,pair_record=execution_conditioned_pairs(executor,x["alignment_test"],addresses,masks,count=c["pairs"],seed=seed+5000)
        save_json(root/"test_pairs.json",pair_record)
        summary={"config":c,"addresses":addresses,"unsupported_addresses":[a for a in frontier if a not in addresses],
            "fit_counts":{a:int(visited[:,j].sum()) for j,a in enumerate(frontier)},"test_combinations":masks.tolist(),
            "held_out_combinations":[m.tolist() for m in masks if m.sum()>1],"sites":{},
            "scope":"numeric operands of an extracted rule; not automatic recovery of raw statistical primitive internals"}
        for site in SITES:
            print("numeric site: "+site,flush=True);directory=root/"sites"/site;directory.mkdir(parents=True)
            decoder=SiteDecoder(model,site);hf=decoder.extract(data["alignment_fit"]);ht=decoder.extract(data["alignment_test"])
            probe=fit_readout(hf,executor,x["alignment_fit"],addresses)
            save_json(directory/"probe.json",probe.to_dict())
            q,record=train_numeric_mapping(decoder,hf,x["alignment_fit"],executor,addresses,fit_masks,probe,
                rank=c["rank"],steps=c["steps"],seed=seed,numeric_weight=c["numeric_weight"])
            save_json(directory/"training.json",record)
            control,record=train_numeric_mapping(decoder,hf,x["alignment_fit"],executor,addresses,fit_masks,probe,
                rank=c["rank"],steps=c["steps"],seed=seed+1000,numeric_weight=c["numeric_weight"],shuffle_targets=True)
            save_json(directory/"shuffled_training.json",record)
            behavior,record=train_numeric_mapping(decoder,hf,x["alignment_fit"],executor,addresses,fit_masks,probe,
                rank=c["rank"],steps=c["steps"],seed=seed,numeric_weight=0.)
            save_json(directory/"behavior_only_training.json",record)
            rng=np.random.default_rng(seed+2000);random=np.linalg.qr(rng.normal(size=(q.shape[0],len(addresses))))[0].reshape(q.shape)
            bases={"numeric":q,"behavior_only":behavior,"shuffled":control,"random":random}
            np.savez_compressed(directory/"bases.npz",**bases)
            metrics={}
            for name,basis in bases.items():
                numeric=audit_numeric(probe,executor,x["alignment_test"],ht,basis,pairs)
                behavioral=measure_joint_mapping(decoder,ht,basis,pairs) if len(pairs.base) else {"status":"no_executed_pairs"}
                save_json(directory/(name+".json"),{"numeric":numeric,"behavioral":behavioral})
                metrics[name]={"numeric_nodes":numeric["per_node"],"behavioral":behavioral.get("overall",behavioral)}
            summary["sites"][site]=metrics
        save_json(root/"summary.json",summary);save_json(root/"status.json",{"state":"completed"})
        snapshot=root/"source";snapshot.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snapshot/p.name)
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","error":str(exc)});raise

def verify_numeric(directory):
    root=Path(directory).resolve()
    for name,expected in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=expected:raise ValueError("Numeric artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete run")
    c=read_json(root/"config.json");s=read_json(root/"summary.json");set_seed(c["seed"])
    if c!=s["config"] or set(s["sites"])!=set(SITES):raise ValueError("Protocol coverage mismatch")
    if digest(root/"teacher.pt")!=c["source_model_sha256"] or digest(root/"program.json")!=c["source_program_sha256"]:raise ValueError("Source mismatch")
    model=load_model(root/"teacher.pt");executor=ProgramExecutor(Rule.from_dict(read_json(root/"program.json")))
    data={};x={};seen=set()
    for split,n in (("alignment_fit",c["fit_worlds"]),("alignment_test",c["test_worlds"])):
        path=root/"datasets"/split;ws=load_worlds(path)
        if ws!=generate_worlds(split,n,c["seed"],c["samples"]):raise ValueError("World protocol mismatch")
        ids={w.identity for w in ws}
        if seen&ids or len(ids)!=n:raise ValueError("World reuse")
        seen|=ids;data[split]=np.stack([w.sample() for w in ws]);x[split]=composed_features(data[split])
        with np.load(path/"samples.npz") as a:np.testing.assert_array_equal(data[split],a["data"])
        with np.load(path/"features.npz") as a:np.testing.assert_allclose(x[split],a["features"],atol=1e-9)
    frontier=numeric_frontier(executor);values,visited=numeric_values(executor,x["alignment_fit"],frontier)
    addresses=[a for j,a in enumerate(frontier) if visited[:,j].sum()>=8 and values[visited[:,j],j].var()>1e-8]
    if addresses!=s["addresses"]:raise ValueError("Fit-only operand selection mismatch")
    masks=compatible_combinations(executor,addresses)
    if masks.tolist()!=s["test_combinations"] or [m.tolist() for m in masks if m.sum()>1]!=s["held_out_combinations"]:raise ValueError("Combination mismatch")
    pairs,record=execution_conditioned_pairs(executor,x["alignment_test"],addresses,masks,count=c["pairs"],seed=c["seed"]+5000)
    if record!=read_json(root/"test_pairs.json"):raise ValueError("Pairing replay mismatch")
    for site in SITES:
        directory=root/"sites"/site;decoder=SiteDecoder(model,site)
        hf=decoder.extract(data["alignment_fit"]);ht=decoder.extract(data["alignment_test"])
        probe=fit_readout(hf,executor,x["alignment_fit"],addresses);saved=NumericReadout.from_dict(read_json(directory/"probe.json"))
        np.testing.assert_allclose(probe.coefficients,saved.coefficients,atol=1e-8)
        if probe.to_dict()!=saved.to_dict():raise ValueError("Probe replay mismatch")
        with np.load(directory/"bases.npz") as a:bases={k:a[k] for k in a.files}
        if set(bases)!={"numeric","behavior_only","shuffled","random"}:raise ValueError("Control coverage mismatch")
        rng=np.random.default_rng(c["seed"]+2000);shape=bases["numeric"].shape
        np.testing.assert_array_equal(bases["random"],np.linalg.qr(rng.normal(size=(shape[0],len(addresses))))[0].reshape(shape))
        for name,basis in bases.items():
            numeric=audit_numeric(probe,executor,x["alignment_test"],ht,basis,pairs)
            behavioral=measure_joint_mapping(decoder,ht,basis,pairs) if len(pairs.base) else {"status":"no_executed_pairs"}
            if {"numeric":numeric,"behavioral":behavioral}!=read_json(directory/(name+".json")):raise ValueError("Numerical audit replay mismatch")
            expected={"numeric_nodes":numeric["per_node"],"behavioral":behavioral.get("overall",behavioral)}
            if expected!=s["sites"][site][name]:raise ValueError("Summary mismatch")
    return {"status":"verified","worlds":len(seen),"sites":list(SITES),"accepted_pairs":len(pairs.base),
            "source_teacher_unchanged":True,"science_not_certified":True}

def main():
    import argparse
    import json
    p=argparse.ArgumentParser();p.add_argument("command",choices=("run","verify"));p.add_argument("directory",type=Path)
    p.add_argument("--source",type=Path);p.add_argument("--seed",type=int,default=291);p.add_argument("--quick",action="store_true")
    a=p.parse_args()
    if a.command=="verify":result=verify_numeric(a.directory)
    else:
        if a.source is None:p.error("--source is required")
        s=run_numeric(a.directory,a.source,a.seed,a.quick);result={"output":str(a.directory),"sites":list(s["sites"])}
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
