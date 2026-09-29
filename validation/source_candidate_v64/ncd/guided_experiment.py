"""Final-test ablation for fit-only mechanism-guided program/site selection."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
import torch
from .io import save_json,read_json,digest
from .model import load_model,predict,set_seed
from .worlds import generate_worlds,save_dataset,load_worlds
from .cdir import composed_features
from .rules import Rule
from .metrics import evaluate
from .guided_synthesis import GuidedConfig,guided_synthesis,verify_guided

TESTS=("test_id","test_function","test_noise","test_scale","test_intervention")

@dataclass
class GuidedExperimentConfig:
    seed:int=393
    samples:int=96
    extraction_worlds:int=512
    fit_worlds:int=512
    validation_worlds:int=1024
    test_worlds:int=512
    steps:int=40
    splits:int=6
    @classmethod
    def quick(cls):
        return cls(seed=391,samples=32,extraction_worlds=96,fit_worlds=96,validation_worlds=128,test_worlds=64,steps=2,splits=2)

def selection_worlds(c):
    return (generate_worlds("extraction",c.extraction_worlds,c.seed,c.samples),
            generate_worlds("alignment_fit",c.fit_worlds,c.seed,c.samples),
            generate_worlds("alignment_fit",c.validation_worlds,c.seed+10000,c.samples))

def run_guided(directory,source,c):
    root=Path(directory).resolve();source=Path(source).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use a new guided experiment directory")
    if any(type(v) is not int or v<1 for v in asdict(c).values()):raise ValueError("Invalid experiment config")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    try:
        save_json(root/"config.json",asdict(c))
        shutil.copy2(source/"model"/"discoverer.pt",root/"source_teacher.pt")
        source_hash=digest(root/"source_teacher.pt");model=load_model(root/"source_teacher.pt")
        extraction,fit,validation=selection_worlds(c)
        gc=GuidedConfig(seed=c.seed,steps=c.steps,max_splits=c.splits)
        guided,plain,selection=guided_synthesis(model,extraction,fit,validation,root/"selection",gc)
        # No final world is generated until candidate and site selection finish.
        programs={"with_internal":guided,"without_internal":plain}
        used={w.identity for ws in (extraction,fit,validation) for w in ws}
        results={}
        for split in TESTS:
            worlds=generate_worlds(split,c.test_worlds,c.seed,c.samples)
            ids={w.identity for w in worlds}
            if used&ids:raise ValueError("Final-test identity leakage")
            used|=ids;save_dataset(root/"tests"/split,worlds)
            data=np.stack([w.sample() for w in worlds]);features=composed_features(data)
            truth=np.array([w.label for w in worlds]);probabilities=predict(model,data)
            metrics={};predictions={}
            for name,program in programs.items():
                predicted=program.predict(features);predictions[name]=predicted
                metrics[name]=evaluate(truth,probabilities,predicted)
            np.savez_compressed(root/"tests"/split/"predictions.npz",features=features,neural=probabilities,**predictions)
            save_json(root/"tests"/split/"metrics.json",metrics);results[split]=metrics
        summary={"config":asdict(c),"source_teacher_sha256":source_hash,"evaluation":results,
                 "selected_programs":{name:selection[name] for name in programs},
                 "selection_changed":selection["with_internal"]!=selection["without_internal"],
                 "candidate_count":len(selection["candidates"]),
                 "positive_internal_support":sum(r["mechanism_support"]>0 for r in selection["candidates"]),
                 "candidate_sites":{r["id"]:r["selected_site"] for r in selection["candidates"]},
                 "scope":"finite candidate reranking with fit/validation internal evidence",
                 "no_full_algorithm_recovery_claim":True}
        save_json(root/"summary.json",summary);save_json(root/"status.json",{"state":"completed"})
        snapshot=root/"source";snapshot.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snapshot/p.name)
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","error":str(exc)});raise

def verify_guided_experiment(directory):
    root=Path(directory).resolve()
    for name,expected in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=expected:raise ValueError("Experiment artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete guided experiment")
    c=GuidedExperimentConfig(**read_json(root/"config.json"));set_seed(c.seed)
    summary=read_json(root/"summary.json")
    if asdict(c)!=summary["config"] or set(summary["evaluation"])!=set(TESTS):raise ValueError("Protocol mismatch")
    if digest(root/"source_teacher.pt")!=summary["source_teacher_sha256"]:raise ValueError("Source teacher mismatch")
    audit=verify_guided(root/"selection")
    selection=read_json(root/"selection"/"selection.json")
    expected_cfg=asdict(GuidedConfig(seed=c.seed,steps=c.steps,max_splits=c.splits));expected_cfg["penalties"]=list(expected_cfg["penalties"])
    if expected_cfg!=read_json(root/"selection"/"config.json"):raise ValueError("Selection config mismatch")
    used=set()
    for role,worlds in zip(("extraction","fit","validation"),selection_worlds(c)):
        if worlds!=load_worlds(root/"selection"/"datasets"/role):raise ValueError("Selection-world protocol mismatch")
        used|={w.identity for w in worlds}
    model=load_model(root/"source_teacher.pt");selected_model=load_model(root/"selection"/"teacher.pt")
    for name,value in model.state_dict().items():torch.testing.assert_close(value,selected_model.state_dict()[name],rtol=0,atol=0)
    programs={name:Rule.from_dict(read_json(root/"selection"/"candidates"/selection[name]/"program.json"))
              for name in ("with_internal","without_internal")}
    if summary["selected_programs"]!={name:selection[name] for name in programs}:raise ValueError("Selected-program mismatch")
    if summary["selection_changed"]!=audit["selection_changed"] or summary["positive_internal_support"]!=audit["positive_internal_support"] or summary["candidate_count"]!=audit["candidates"]:raise ValueError("Selection summary mismatch")
    if summary["candidate_sites"]!={r["id"]:r["selected_site"] for r in selection["candidates"]}:raise ValueError("Site summary mismatch")
    if {p.name for p in (root/"tests").iterdir()}!=set(TESTS):raise ValueError("Test coverage mismatch")
    for split in TESTS:
        path=root/"tests"/split;worlds=load_worlds(path)
        if worlds!=generate_worlds(split,c.test_worlds,c.seed,c.samples):raise ValueError("Test-world protocol mismatch")
        ids={w.identity for w in worlds}
        if used&ids or len(ids)!=c.test_worlds:raise ValueError("Test leakage")
        used|=ids
        data=np.stack([w.sample() for w in worlds]);truth=np.array([w.label for w in worlds])
        with np.load(path/"samples.npz") as a:
            np.testing.assert_array_equal(data,a["data"]);np.testing.assert_array_equal(truth,a["labels"])
        features=composed_features(data);probabilities=predict(model,data)
        expected={}
        with np.load(path/"predictions.npz") as a:
            np.testing.assert_allclose(features,a["features"],atol=1e-9);np.testing.assert_allclose(probabilities,a["neural"],atol=1e-7)
            for name,program in programs.items():
                predictions=program.predict(features);np.testing.assert_array_equal(predictions,a[name])
                expected[name]=evaluate(truth,probabilities,predictions)
        if expected!=read_json(path/"metrics.json") or expected!=summary["evaluation"][split]:raise ValueError("Final-test metric mismatch")
    return {"status":"verified","worlds":len(used),"selection":audit,"final_environments":list(TESTS),
            "science_not_certified":True}

def main():
    import argparse
    import json
    p=argparse.ArgumentParser();p.add_argument("command",choices=("run","verify"));p.add_argument("directory",type=Path)
    p.add_argument("--source",type=Path);p.add_argument("--seed",type=int);p.add_argument("--quick",action="store_true")
    args=p.parse_args()
    if args.command=="verify":result=verify_guided_experiment(args.directory)
    else:
        if args.source is None:p.error("--source is required")
        c=GuidedExperimentConfig.quick() if args.quick else GuidedExperimentConfig()
        if args.seed is not None:c.seed=args.seed
        s=run_guided(args.directory,args.source,c)
        result={"directory":str(args.directory),"selection_changed":s["selection_changed"],"candidates":s["candidate_count"]}
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
