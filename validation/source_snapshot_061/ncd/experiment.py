"""Reproducible experiment orchestration with explicit split boundaries."""
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import scipy
import sklearn
import torch
from .io import save_json,digest
from .worlds import generate_worlds,save_dataset
from .statistics import extract,FEATURES
from .model import train,predict
from .synthesis import synthesize
from .counterexamples import refine,compare_search
from .alignment import fit_and_test
from .baselines import decision_tree,calibrate_anm,anm_predict
from .metrics import evaluate
from .report import render

@dataclass
class Config:
    seed: int = 42
    samples: int = 96
    train_worlds: int = 2400
    dev_worlds: int = 384
    extraction_worlds: int = 768
    refinement_worlds: int = 512
    alignment_worlds: int = 384
    test_worlds: int = 384
    epochs: int = 40
    width: int = 48
    beam_width: int = 4
    max_splits: int = 4
    penalty: float = .001
    cegis_rounds: int = 2
    query_budget: int = 64

    def validate(self):
        for name,value in asdict(self).items():
            if name=="penalty":
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
                    raise ValueError("penalty must be finite numeric")
            elif isinstance(value,bool) or not isinstance(value,int):
                raise ValueError(f"{name} must be an integer")
        if not 0<=self.seed<2**32: raise ValueError("seed must be a 32-bit nonnegative integer")
        for k in ("samples","train_worlds","dev_worlds","extraction_worlds","refinement_worlds","alignment_worlds","test_worlds"):
            if getattr(self,k) < 32: raise ValueError(f"{k} must be >= 32")
        for k in ("epochs","width","beam_width","max_splits","query_budget"):
            if getattr(self,k)<1: raise ValueError(f"{k} must be positive")
        if self.penalty<0 or self.cegis_rounds<0: raise ValueError("Invalid search configuration")

    @classmethod
    def quick(cls, seed=42):
        return cls(seed=seed,samples=48,train_worlds=192,dev_worlds=64,extraction_worlds=96,
                   refinement_worlds=96,alignment_worlds=64,test_worlds=64,
                   epochs=3,width=24,beam_width=2,max_splits=2,cegis_rounds=1,query_budget=16)

def run(directory, config):
    config.validate()
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()): raise FileExistsError("Run directory must be empty; previous evidence is never overwritten")
    root.mkdir(parents=True,exist_ok=True)
    start = time.time()
    save_json(root/"config.json",asdict(config))
    save_json(root/"status.json",{"state":"running","stage":"datasets"})
    try:
        return _execute(root,config,start)
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","error_type":type(exc).__name__,"message":str(exc)})
        raise

def _execute(root,c,start):
    counts = {"train":c.train_worlds,"dev":c.dev_worlds,"extraction":c.extraction_worlds,
              "refinement":c.refinement_worlds,"alignment_fit":c.alignment_worlds,
              "alignment_test":c.test_worlds,**{s:c.test_worlds for s in
              ("test_id","test_function","test_noise","test_scale","test_intervention")}}
    datasets,worlds = {},{}
    for split,count in counts.items():
        print(f"dataset {split}: {count} worlds",flush=True)
        ws = generate_worlds(split,count,c.seed,c.samples)
        worlds[split] = ws
        save_dataset(root/"datasets"/split,ws)
        with np.load(root/"datasets"/split/"samples.npz") as f:
            datasets[split] = {"data":f["data"],"labels":f["labels"]}
    print("training frozen teacher",flush=True)
    save_json(root/"status.json",{"state":"running","stage":"training"})
    model = train(datasets["train"]["data"],datasets["train"]["labels"],
                  datasets["dev"]["data"],datasets["dev"]["labels"],root/"model",
                  epochs=c.epochs,width=c.width,seed=c.seed)
    teacher_hash = digest(root/"model"/"discoverer.pt")
    for split,d in datasets.items():
        if split == "train": continue
        print(f"statistical primitives: {split}",flush=True)
        d["features"] = extract(d["data"])
        d["probabilities"] = predict(model,d["data"])
        np.savez_compressed(root/"datasets"/split/"derived.npz",features=d["features"],probabilities=d["probabilities"])
    search = {"penalty":c.penalty,"beam_width":c.beam_width,"max_splits":c.max_splits}
    ex,ref = datasets["extraction"],datasets["refinement"]
    teacher = ex["probabilities"].argmax(1)
    print("program synthesis: teacher only",flush=True)
    initial,trace = synthesize(ex["features"],teacher,**search)
    save_json(root/"programs"/"search_trace.json",trace)
    print("counterexample-guided refinement",flush=True)
    final,cegis = refine(initial,ex["features"],teacher,ref["features"],ref["probabilities"].argmax(1),
                        rounds=c.cegis_rounds,per_round=c.query_budget,**search)
    save_json(root/"programs"/"cegis.json",cegis)
    print("baselines: same DSL truth, no MDL, decision tree",flush=True)
    truth_program,truth_trace = synthesize(ex["features"],ex["labels"],**search)
    no_mdl_initial,_ = synthesize(ex["features"],teacher,**{**search,"penalty":0.})
    no_mdl,no_mdl_cegis = refine(no_mdl_initial,ex["features"],teacher,ref["features"],
                                ref["probabilities"].argmax(1),rounds=c.cegis_rounds,
                                per_round=c.query_budget,**{**search,"penalty":0.})
    save_json(root/"programs"/"without_mdl_cegis.json",no_mdl_cegis)
    programs = {"full":final,"without_cegis":initial,"truth_supervised_same_dsl":truth_program,
                "without_mdl":no_mdl,"decision_tree":decision_tree(ex["features"],teacher,c.seed,c.max_splits+1)}
    for name,p in programs.items():
        save_json(root/"programs"/f"{name}.json",{"program":p.to_dict(),"complexity":p.complexity,
            "teacher_sha256":teacher_hash if name!="truth_supervised_same_dsl" else None,
            "supervision":"ground_truth" if name=="truth_supervised_same_dsl" else "frozen_teacher",
            "features":list(FEATURES)})
        (root/"programs"/f"{name}.txt").write_text(p.text()+"\n",encoding="utf-8")
    dev = datasets["dev"]
    anm = calibrate_anm(dev["features"],dev["labels"])
    save_json(root/"programs"/"anm.json",anm)
    print("independent counterexample-search benchmark",flush=True)
    # A new seed namespace; these worlds never refine the program.
    attack_ws = generate_worlds("refinement",c.refinement_worlds,c.seed+1000003,c.samples)
    save_dataset(root/"datasets"/"attack",attack_ws)
    attack_data = np.stack([w.sample() for w in attack_ws])
    attack_x,attack_y = extract(attack_data),np.array([w.label for w in attack_ws])
    attacks = compare_search(final,attack_x,attack_y,lambda ids:predict(model,attack_data[ids]),
                            budget=c.query_budget,seed=c.seed)
    save_json(root/"counterexamples.json",attacks)
    np.savez_compressed(root/"datasets"/"attack"/"derived.npz",features=attack_x)
    print("held-out interchange interventions",flush=True)
    af,at = datasets["alignment_fit"],datasets["alignment_test"]
    alignment,state = fit_and_test(model,final,af["data"],af["features"],at["data"],at["features"],c.seed)
    save_json(root/"alignment.json",alignment)
    if state is not None: np.savez_compressed(root/"alignment_state.npz",**state)
    evaluation = {}
    for split,d in datasets.items():
        if not split.startswith("test_"): continue
        methods = {name:evaluate(d["labels"],d["probabilities"],p.predict(d["features"])) for name,p in programs.items()}
        methods["anm"] = evaluate(d["labels"],d["probabilities"],anm_predict(d["features"],
                                    anm["independence_threshold"],anm["ambiguity_threshold"]))
        evaluation[split] = methods
        save_json(root/"evaluations"/f"{split}.json",{
            "world_ids":[w.identity for w in worlds[split]],"truth":d["labels"].tolist(),
            "neural":d["probabilities"].argmax(1).tolist(),
            "programs":{name:p.predict(d["features"]).tolist() for name,p in programs.items()},
            "metrics":methods})
    save_json(root/"data_policy.json",{
        "network_train":["train"],"network_selection":["dev"],
        "program_synthesis":["extraction"],"program_refinement":["refinement"],
        "anm_calibration":["dev"],"alignment_selection_and_fit":["alignment_fit"],
        "alignment_evaluation":["alignment_test"],"counterexample_benchmark":["attack"],
        "final_evaluation":[s for s in counts if s.startswith("test_")],
        "split_unit":"whole SCM world, not rows","all_test_worlds_untouched_by_fitting":True,
        "generator_family_overlap_is_intentional_for_id":True})
    summary = {"config":asdict(c),"teacher_sha256":teacher_hash,"evaluation":evaluation,
               "counterexamples":attacks,"alignment":alignment,
               "program_complexity":{name:p.complexity for name,p in programs.items()},
               "runtime_seconds":time.time()-start,
               "versions":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
                           "torch":torch.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
               "claim":"behavioral extraction with empirical alignment audit; not guaranteed mechanism recovery"}
    save_json(root/"summary.json",summary)
    (root/"report.html").write_text(render(summary,programs),encoding="utf-8")
    # Preserve the exact executed source alongside evidence.
    source_root = Path(__file__).parent
    snapshot = root/"source"/"ncd"
    snapshot.mkdir(parents=True,exist_ok=True)
    for path in source_root.glob("*.py"): shutil.copy2(path,snapshot/path.name)
    project_file = source_root.parent/"pyproject.toml"
    if project_file.is_file():
        shutil.copy2(project_file,root/"source"/"pyproject.toml")
    else:
        from importlib.metadata import metadata
        save_json(root/"source"/"installed_package_metadata.json",dict(metadata("neural-causal-decompiler")))
    save_json(root/"status.json",{"state":"completed","runtime_seconds":time.time()-start})
    manifest = {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    save_json(root/"manifest.json",{"algorithm":"sha256","artifacts":manifest,
                                 "purpose":"integrity, not authenticity against coordinated tampering"})
    print(f"completed: {root}",flush=True)
    return summary
