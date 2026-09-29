"""Independent multi-intermediate experiment on newly trained causal networks."""
from dataclasses import dataclass,asdict,fields
from itertools import combinations
from pathlib import Path
import shutil
import time
import numpy as np
from .io import save_json,read_json,digest
from .worlds import generate_worlds,save_dataset,load_worlds
from .model import train,predict,load_model,set_seed
from .alignment import hidden
from .cdir import composed_features
from .statistics import FEATURES
from .rules import Rule,fit_rule
from .program_trace import ProgramExecutor
from .joint_alignment import JointPairs,make_pairs,train_joint_mapping,measure_joint_mapping

@dataclass
class JointConfig:
    seed:int=191
    samples:int=96
    train_worlds:int=1800
    dev_worlds:int=256
    extraction_worlds:int=512
    fit_worlds:int=384
    validation_worlds:int=256
    test_worlds:int=384
    epochs:int=40
    steps:int=120
    pairs:int=1024
    controls:int=10
    splits:int=6
    @classmethod
    def quick(cls):
        return cls(samples=32,train_worlds=128,dev_worlds=48,extraction_worlds=96,
                   fit_worlds=64,validation_worlds=48,test_worlds=64,epochs=3,
                   steps=8,pairs=96,controls=2,splits=3)

def protocol(executor,mode):
    suffix="/expr" if mode=="expression" else "/predicate"
    addresses=[executor.address(p) for p in executor.catalog if p.endswith(suffix)]
    k=len(addresses)
    singles=np.eye(k,dtype=bool)
    pairs=[]
    for i,j in combinations(range(k),2):
        mask=np.zeros(k,bool);mask[[i,j]]=True;pairs.append(mask)
    # At least one pair is always held out; pair allocation fixed before fitting.
    held=pairs[::2];seen=pairs[1::2]
    fit=np.array(list(singles)+seen,dtype=bool)
    test=np.array(list(singles)+seen+held,dtype=bool)
    return addresses,fit,test,[m.tolist() for m in held]

def _subset(pairs,ids,column=None):
    values={}
    for f in fields(JointPairs):
        value=getattr(pairs,f.name)[ids]
        if column is not None and value.ndim==2:value=value[:,column:column+1]
        values[f.name]=value
    return JointPairs(**values)

def worlds_for(split,n,c):
    # Separate namespace through a fixed seed offset; generator split semantics unchanged.
    return generate_worlds("alignment_fit" if split=="alignment_validation" else split,n,
                           c.seed+10000 if split=="alignment_validation" else c.seed,c.samples)

def counts(c):
    return {"train":c.train_worlds,"dev":c.dev_worlds,"extraction":c.extraction_worlds,
            "alignment_fit":c.fit_worlds,"alignment_validation":c.validation_worlds,
            **{s:c.test_worlds for s in ("alignment_test","test_function","test_noise","test_scale","test_intervention")}}

def _finish(root,summary):
    save_json(root/"summary.json",summary)
    save_json(root/"status.json",{"state":"completed"})
    source=root/"source";source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,source/p.name)
    save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p)
        for p in sorted(root.rglob("*")) if p.is_file()}})
    return summary

def run_joint(directory,c):
    root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use a new joint experiment directory")
    if any(type(v) is not int or v<1 for v in asdict(c).values()):raise ValueError("Invalid config")
    root.mkdir(parents=True,exist_ok=True);start=time.time()
    save_json(root/"config.json",asdict(c));save_json(root/"status.json",{"state":"running"})
    try:
        data={};features={};truth={}
        for split,n in counts(c).items():
            ws=worlds_for(split,n,c);save_dataset(root/"datasets"/split,ws)
            data[split]=np.stack([w.sample() for w in ws]);truth[split]=np.array([w.label for w in ws])
            if split not in ("train","dev"):
                features[split]=composed_features(data[split])
                np.savez_compressed(root/"datasets"/split/"features.npz",features=features[split])
        print("train new frozen teacher",flush=True)
        model=train(data["train"],truth["train"],data["dev"],truth["dev"],root/"model",epochs=c.epochs,seed=c.seed)
        y=predict(model,data["extraction"]).argmax(1)
        rule,search=fit_rule(features["extraction"],y,FEATURES,max_splits=c.splits,beam_width=3)
        save_json(root/"program.json",rule.to_dict());save_json(root/"synthesis.json",search)
        executor=ProgramExecutor(rule)
        save_json(root/"program_trace.json",executor.execute(features["extraction"][:8]).to_dict())
        representations={s:hidden(model,data[s]) for s in features}
        for s,h in representations.items():np.savez_compressed(root/"datasets"/s/"hidden.npz",hidden=h)
        summary={"config":asdict(c),"teacher_sha256":digest(root/"model"/"discoverer.pt"),
                 "program_id":executor.program_id,"modes":{},"behavior":{},
                 "claim":"joint empirical audit on a free-trained teacher; no full algorithm recovery claim",
                 "multi_layer_selection_implemented":False,"joint_guidance_in_synthesis":False}
        for split in ("alignment_test","test_function","test_noise","test_scale","test_intervention"):
            p=predict(model,data[split]).argmax(1);r=rule.predict(features[split])
            summary["behavior"][split]={"neural_accuracy":float(np.mean(p==truth[split])),
                "program_accuracy":float(np.mean(r==truth[split])),"fidelity":float(np.mean(p==r))}
        for mode in ("expression","predicate"):
            addresses,fit_masks,test_masks,held=protocol(executor,mode);k=len(addresses)
            if k<2:
                summary["modes"][mode]={"status":"insufficient_intermediates","count":k};continue
            print(f"joint mapping {mode}: {k} intermediates",flush=True)
            directory=root/"mappings"/mode;directory.mkdir(parents=True)
            validation_pairs=make_pairs(executor,features["alignment_validation"],addresses,fit_masks,count=c.pairs,seed=c.seed+800)
            choices=[]
            for rank in (1,2):
                q,record=train_joint_mapping(model,representations["alignment_fit"],features["alignment_fit"],
                    executor,addresses,fit_masks,rank=rank,steps=c.steps,seed=c.seed+rank)
                result=measure_joint_mapping(model,representations["alignment_validation"],q,validation_pairs)
                np.savez_compressed(directory/f"rank{rank}.npz",basis=q)
                save_json(directory/f"rank{rank}.json",{"training":record,"validation":result})
                score=result["overall"]["informative_accuracy"]
                choices.append((-1 if score is None else score,-rank,rank,q))
            _,_,rank,q=max(choices,key=lambda v:(v[0],v[1]))
            bases={"joint":q,"permuted":q[:,np.roll(np.arange(k),1),:]}
            shuffled,record=train_joint_mapping(model,representations["alignment_fit"],features["alignment_fit"],
                executor,addresses,fit_masks,rank=rank,steps=c.steps,seed=c.seed+700,shuffle_targets=True)
            bases["shuffled"]=shuffled;save_json(directory/"shuffled_training.json",record)
            rng=np.random.default_rng(c.seed+900)
            for i in range(c.controls):
                bases[f"random{i}"]=np.linalg.qr(rng.normal(size=(q.shape[0],k*rank)))[0].reshape(q.shape)
            singles={}
            for j,address in enumerate(addresses):
                single,record=train_joint_mapping(model,representations["alignment_fit"],features["alignment_fit"],
                    executor,[address],np.array([[True]]),rank=rank,steps=c.steps,seed=c.seed+1000+j)
                singles[f"single{j}"]=single
                save_json(directory/f"single{j}_training.json",record)
            np.savez_compressed(directory/"bases.npz",**bases,**singles)
            tests={}
            for split in ("alignment_test","test_function","test_noise","test_scale","test_intervention"):
                for disjoint in (False,True):
                    key=split+("_disjoint" if disjoint else "")
                    pairs=make_pairs(executor,features[split],addresses,test_masks,count=c.pairs,seed=c.seed+2000,disjoint=disjoint)
                    measurements={name:measure_joint_mapping(model,representations[split],basis,pairs) for name,basis in bases.items()}
                    single_results={}
                    for j in range(k):
                        ids=(pairs.masks.sum(1)==1)&pairs.masks[:,j]
                        if not ids.any():continue
                        single_results[str(j)]={
                            "independent":measure_joint_mapping(model,representations[split],singles[f"single{j}"],_subset(pairs,ids,j)),
                            "joint":measure_joint_mapping(model,representations[split],q,_subset(pairs,ids))}
                    save_json(root/"evaluations"/mode/f"{key}.json",{"methods":measurements,"single_controls":single_results})
                    tests[key]={name:value["overall"] for name,value in measurements.items()}
            summary["modes"][mode]={"status":"measured","addresses":addresses,"selected_rank":rank,
                "fit_combinations":fit_masks.tolist(),"test_combinations":test_masks.tolist(),
                "held_out_combinations":held,"selection_worlds":"alignment_validation",
                "tests":tests,"single_controls":"evaluations/"+mode+"/"}
        summary["runtime_seconds"]=time.time()-start
        return _finish(root,summary)
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","error":str(exc)});raise

def verify_joint(directory):
    root=Path(directory).resolve()
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path)!=value:raise ValueError("Joint artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete joint experiment")
    summary=read_json(root/"summary.json");c=JointConfig(**summary["config"]);set_seed(c.seed)
    if read_json(root/"config.json")!=asdict(c):raise ValueError("Config mismatch")
    if set(summary["modes"])!={"expression","predicate"}:raise ValueError("Mode coverage mismatch")
    if {p.name for p in (root/"datasets").iterdir()}!=set(counts(c)):raise ValueError("Dataset coverage mismatch")
    model=load_model(root/"model"/"discoverer.pt")
    if digest(root/"model"/"discoverer.pt")!=summary["teacher_sha256"]:raise ValueError("Teacher mismatch")
    rule=Rule.from_dict(read_json(root/"program.json"));executor=ProgramExecutor(rule)
    if executor.program_id!=summary["program_id"]:raise ValueError("Program mismatch")
    features={};representations={};seen=set();total=0
    for split,n in counts(c).items():
        path=root/"datasets"/split;ws=load_worlds(path)
        if ws!=worlds_for(split,n,c):raise ValueError("Dataset protocol mismatch")
        ids={w.identity for w in ws}
        if len(ids)!=len(ws) or seen&ids:raise ValueError("World leakage")
        seen|=ids;total+=len(ws)
        data=np.stack([w.sample() for w in ws])
        with np.load(path/"samples.npz") as a:
            np.testing.assert_array_equal(data,a["data"])
            np.testing.assert_array_equal([w.label for w in ws],a["labels"])
        if split in ("train","dev"):continue
        features[split]=composed_features(data);representations[split]=hidden(model,data)
        with np.load(path/"features.npz") as a:np.testing.assert_allclose(features[split],a["features"],atol=1e-9)
        with np.load(path/"hidden.npz") as a:np.testing.assert_allclose(representations[split],a["hidden"],atol=1e-6)
        if split in summary["behavior"]:
            p=predict(model,data).argmax(1);r=rule.predict(features[split]);truth=np.array([w.label for w in ws])
            actual={"neural_accuracy":float(np.mean(p==truth)),"program_accuracy":float(np.mean(r==truth)),"fidelity":float(np.mean(p==r))}
            if actual!=summary["behavior"][split]:raise ValueError("Behavior mismatch")
    if executor.execute(features["extraction"][:8]).to_dict()!=read_json(root/"program_trace.json"):raise ValueError("Trace mismatch")
    for mode,item in summary["modes"].items():
        addresses,fit_masks,test_masks,held=protocol(executor,mode)
        if item["status"]!="measured":
            if len(addresses)>=2:raise ValueError("Incorrect insufficient coverage")
            continue
        if addresses!=item["addresses"] or held!=item["held_out_combinations"] or fit_masks.tolist()!=item["fit_combinations"] or test_masks.tolist()!=item["test_combinations"]:
            raise ValueError("Combination protocol mismatch")
        expected_tests={s+suffix for s in ("alignment_test","test_function","test_noise","test_scale","test_intervention") for suffix in ("","_disjoint")}
        if set(item["tests"])!=expected_tests:raise ValueError("Evaluation coverage mismatch")
        directory=root/"mappings"/mode
        vp=make_pairs(executor,features["alignment_validation"],addresses,fit_masks,count=c.pairs,seed=c.seed+800)
        scores=[]
        for rank in (1,2):
            with np.load(directory/f"rank{rank}.npz") as a:q=a["basis"]
            result=measure_joint_mapping(model,representations["alignment_validation"],q,vp)
            if result!=read_json(directory/f"rank{rank}.json")["validation"]:raise ValueError("Selection replay mismatch")
            score=result["overall"]["informative_accuracy"];scores.append((-1 if score is None else score,-rank,rank))
        if max(scores)[2]!=item["selected_rank"]:raise ValueError("Rank selection mismatch")
        with np.load(directory/"bases.npz") as a:bases={k:a[k] for k in a.files}
        with np.load(directory/f"rank{item['selected_rank']}.npz") as a:np.testing.assert_array_equal(a["basis"],bases["joint"])
        expected_bases={"joint","permuted","shuffled"}|{f"random{i}" for i in range(c.controls)}|{f"single{j}" for j in range(len(addresses))}
        if set(bases)!=expected_bases:raise ValueError("Control coverage mismatch")
        np.testing.assert_array_equal(bases["permuted"],bases["joint"][:,np.roll(np.arange(len(addresses)),1),:])
        rng=np.random.default_rng(c.seed+900);shape=bases["joint"].shape
        for i in range(c.controls):
            expected=np.linalg.qr(rng.normal(size=(shape[0],shape[1]*shape[2])))[0].reshape(shape)
            np.testing.assert_array_equal(bases[f"random{i}"],expected)
        for key,expected in item["tests"].items():
            disjoint=key.endswith("_disjoint");split=key[:-9] if disjoint else key
            pairs=make_pairs(executor,features[split],addresses,test_masks,count=c.pairs,seed=c.seed+2000,disjoint=disjoint)
            saved=read_json(root/"evaluations"/mode/f"{key}.json")
            if disjoint and len(np.unique(np.c_[pairs.base,pairs.sources]))!=len(pairs.base)*(len(addresses)+1):raise ValueError("Disjoint reuse")
            for name,value in bases.items():
                if name.startswith("single"):continue
                result=measure_joint_mapping(model,representations[split],value,pairs)
                if result!=saved["methods"][name] or result["overall"]!=expected[name]:raise ValueError("Mapping replay mismatch")
            for j in range(len(addresses)):
                ids=(pairs.masks.sum(1)==1)&pairs.masks[:,j]
                if not ids.any():continue
                actual={"independent":measure_joint_mapping(model,representations[split],bases[f"single{j}"],_subset(pairs,ids,j)),
                        "joint":measure_joint_mapping(model,representations[split],bases["joint"],_subset(pairs,ids))}
                if actual!=saved["single_controls"][str(j)]:raise ValueError("Single control mismatch")
    return {"status":"verified","worlds":total,"modes":{m:i["status"] for m,i in summary["modes"].items()},
            "science_not_certified":True}


def main():
    import argparse
    import json
    parser=argparse.ArgumentParser(description="Multi-intermediate causal mechanism experiment")
    parser.add_argument("command",choices=("run","verify"))
    parser.add_argument("directory",type=Path)
    parser.add_argument("--seed",type=int,default=191)
    parser.add_argument("--quick",action="store_true")
    args=parser.parse_args()
    if args.command=="verify":result=verify_joint(args.directory)
    else:
        c=JointConfig.quick() if args.quick else JointConfig();c.seed=args.seed
        summary=run_joint(args.directory,c)
        result={"output":str(args.directory),"runtime_seconds":summary["runtime_seconds"],
                "modes":{k:v["status"] for k,v in summary["modes"].items()}}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
