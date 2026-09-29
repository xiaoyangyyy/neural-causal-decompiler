"""Expanded bivariate research: composed IR, circuit-guided synthesis, active CEGIS,
real PySR and environment-augmentation ablations. Final tests never feed search.
"""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
from html import escape
import shutil
import time
import numpy as np
from .worlds import generate_worlds,save_dataset,World
from .model import train,predict,load_model
from .cdir import composed_features,discovery_feature_nodes
from .statistics import FEATURES
from .rules import fit_rule,Rule
from .baselines import decision_tree
from .active_search import search_scm,replicate_counterexamples
from .distributed_alignment import fit_guidance,distributed_audit,measure_mapping
from .alignment import hidden
from .metrics import evaluate
from .io import save_json,read_json,digest
from .pysr_baseline import fit_pysr
from .evidence import fidelity_bound,empirical_equivalence
from .traces import TorchCircuit

@dataclass
class ResearchConfig:
    seed:int=91
    samples:int=96
    train_worlds:int=1800
    dev_worlds:int=256
    extraction_worlds:int=512
    alignment_worlds:int=256
    test_worlds:int=384
    epochs:int=40
    max_splits:int=6
    beam_width:int=3
    cegis_rounds:int=2
    search_budget:int=64
    mapping_steps:int=100
    random_controls:int=10
    pysr_iterations:int=12

    @classmethod
    def quick(cls):
        return cls(samples=32,train_worlds=128,dev_worlds=48,extraction_worlds=64,alignment_worlds=64,
                   test_worlds=64,epochs=3,max_splits=2,beam_width=2,cegis_rounds=1,
                   search_budget=12,mapping_steps=8,random_controls=3,pysr_iterations=2)

    def validate(self):
        for name,value in asdict(self).items():
            if not isinstance(value,int) or value<1:raise ValueError(f"Invalid {name}")
        if min(self.samples,self.train_worlds,self.dev_worlds,self.extraction_worlds,self.alignment_worlds,self.test_worlds)<16:
            raise ValueError("Datasets too small")

def training_worlds(c,augment):
    ws=generate_worlds("train",c.train_worlds,c.seed,c.samples)
    if not augment:return ws
    # Fixed, training-only moderate shifts. Final function and heavy-tail noise
    # families remain held out; no test world or metadata enters augmentation.
    rng=np.random.default_rng(c.seed+88)
    return [replace(w,split="train_augmented",scale_x=float(np.exp(rng.uniform(-.5,.5))),
        scale_y=float(np.exp(rng.uniform(-.5,.5))),
        effect_noise="uniform" if w.family not in ("linear_gaussian","linear_nongaussian") else w.effect_noise)
        if i%2 else w for i,w in enumerate(ws)]

def active_refine(model,initial,x,teacher,c,directory,*,alignment=None,penalty=.001,rounds=None):
    current=initial;all_x=x.copy();all_y=teacher.copy();records=[]
    rounds=c.cegis_rounds if rounds is None else rounds
    for iteration in range(rounds):
        seeds=generate_worlds("refinement",6,c.seed+100+iteration,c.samples)
        result=search_scm(model,current,seeds,objective="fidelity",budget=c.search_budget,
                          seed=c.seed+iteration,feature_extractor=composed_features)
        result["program_before"]=current.to_dict()
        failures=[r for r in result["records"] if r["fidelity_error"]]
        if failures:
            ws=[World(**{k:v for k,v in r["world"].items() if k in World.__dataclass_fields__}) for r in failures]
            all_x=np.r_[all_x,composed_features(np.stack([w.sample() for w in ws]))]
            all_y=np.r_[all_y,[r["neural"] for r in failures]]
            current,_=fit_rule(all_x,all_y,FEATURES,max_splits=c.max_splits,beam_width=c.beam_width,
                               penalty=penalty,feature_alignment=alignment,alignment_weight=.015 if alignment is not None else 0.)
        records.append(result)
    save_json(Path(directory)/"active_cegis.json",{"rounds":records,"teacher_only_objective":True})
    return current

def research_report(summary,programs):
    rows=[]
    for env,methods in summary["evaluation"].items():
        for name,m in methods.items():
            rows.append("<tr>"+"".join("<td>"+escape(str(v))+"</td>" for v in
                (env,name,round(m["neural_accuracy"],3),round(m["program_accuracy"],3),round(m["fidelity"],3)))+"</tr>")
    programs_html="".join("<h3>"+escape(n)+"</h3><pre>"+escape(p.text())+"</pre>" for n,p in programs.items() if isinstance(p,Rule))
    compact_alignment={k:v for k,v in summary["alignment"].items() if k not in ("validation","test","random_controls","shuffled_target_control","disjoint_test")}
    if "test" in summary["alignment"]:
        compact_alignment["test"]={k:v for k,v in summary["alignment"]["test"].items() if k not in ("base_indices","source_indices","symbolic_target","neural_after")}
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整因果反编译研究</title>
<style>body{max-width:1200px;margin:40px auto;font:16px/1.6 system-ui}td,th{padding:6px;border-bottom:1px solid #ccc}pre{background:#edf2f7;padding:16px;white-space:pre-wrap}</style>
<h1>组合式 IR · 内部对齐 · 主动反例 · 完整消融</h1>
<p>真实 PySR 对照、相同预算的无对齐/无MDL/无CEGIS，以及无训练环境扩增对照。因果准确率与网络一致率分开。
对齐仅在拟合集选择；全部最终测试不参与合成。</p>
<p>程序类别：0=X→Y，1=Y→X，2=独立，3=未定。</p><table><tr><th>环境</th><th>方法</th><th>网络准确率</th><th>程序准确率</th><th>一致率</th></tr>"""+"".join(rows)+"</table>"+programs_html+"<h2>内部对齐</h2><pre>"+escape(__import__("json").dumps(compact_alignment,ensure_ascii=False,indent=2))+"</pre></html>"

def run_research(directory,c):
    c.validate();root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Research directory must be empty")
    root.mkdir(parents=True,exist_ok=True);start=time.time()
    save_json(root/"config.json",asdict(c));save_json(root/"status.json",{"state":"running"})
    try:return _execute(root,c,start)
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def _execute(root,c,start):
    datasets={};all_worlds={}
    counts={"dev":c.dev_worlds,"extraction":c.extraction_worlds,"alignment_fit":c.alignment_worlds,
            "alignment_test":c.test_worlds,**{s:c.test_worlds for s in
            ("test_id","test_function","test_noise","test_scale","test_intervention")}}
    for split,count in counts.items():
        ws=generate_worlds(split,count,c.seed,c.samples);all_worlds[split]=ws
        save_dataset(root/"datasets"/split,ws)
        data=np.stack([w.sample() for w in ws])
        datasets[split]={"data":data,"truth":np.array([w.label for w in ws]),"features":composed_features(data)}
        np.savez_compressed(root/"datasets"/split/"features.npz",features=datasets[split]["features"])
    save_json(root/"cdir.json",{"names":list(FEATURES),"nodes":[n.to_dict() for n in discovery_feature_nodes()],
                               "semantics":"composed_raw_dataset_statistics_v2"})
    teachers={}
    for label,augment in (("full",True),("without_environment_augmentation",False)):
        print(f"training research teacher: {label}",flush=True)
        ws=training_worlds(c,augment);save_dataset(root/"datasets"/("train_"+label),ws)
        data=np.stack([w.sample() for w in ws]);truth=np.array([w.label for w in ws])
        teachers[label]=train(data,truth,datasets["dev"]["data"],datasets["dev"]["truth"],root/"models"/label,
                              epochs=c.epochs,seed=c.seed)
    import torch
    trace_root=root/"circuits";trace_root.mkdir()
    trace=TorchCircuit(teachers["full"],["encoder.0","encoder.2","head.0"]).capture(torch.tensor(datasets["alignment_fit"]["data"][:4]))
    save_json(trace_root/"trace.json",trace.save(trace_root/"activations.npz"))
    for d in datasets.values():d["probabilities"]=predict(teachers["full"],d["data"])
    ex,fit=datasets["extraction"],datasets["alignment_fit"];x=ex["features"];y=ex["probabilities"].argmax(1)
    kwargs={"max_splits":c.max_splits,"beam_width":c.beam_width}
    initial,_=fit_rule(x,y,FEATURES,**kwargs)
    print("fit-only circuit guidance",flush=True)
    guidance,guidance_record=fit_guidance(teachers["full"],initial,fit["data"],fit["features"],steps=c.mapping_steps,seed=c.seed)
    save_json(root/"guidance.json",{"scores":guidance.tolist(),**guidance_record})
    guided_initial,_=fit_rule(x,y,FEATURES,feature_alignment=guidance,alignment_weight=.015,**kwargs)
    programs={}
    programs["full"]=active_refine(teachers["full"],guided_initial,x,y,c,root/"variants"/"full",alignment=guidance)
    programs["without_cegis"]=guided_initial
    programs["without_alignment"]=active_refine(teachers["full"],initial,x,y,c,root/"variants"/"without_alignment")
    no_mdl,_=fit_rule(x,y,FEATURES,penalty=0,feature_alignment=guidance,alignment_weight=.015,**kwargs)
    programs["without_mdl"]=active_refine(teachers["full"],no_mdl,x,y,c,root/"variants"/"without_mdl",alignment=guidance,penalty=0)
    noaug_y=predict(teachers["without_environment_augmentation"],ex["data"]).argmax(1)
    noaug_initial,_=fit_rule(x,noaug_y,FEATURES,**kwargs)
    noaug_guidance,noaug_record=fit_guidance(teachers["without_environment_augmentation"],noaug_initial,fit["data"],fit["features"],steps=c.mapping_steps,seed=c.seed)
    save_json(root/"guidance_without_augmentation.json",{"scores":noaug_guidance.tolist(),**noaug_record})
    noaug_initial,_=fit_rule(x,noaug_y,FEATURES,feature_alignment=noaug_guidance,alignment_weight=.015,**kwargs)
    programs["without_environment_augmentation"]=active_refine(teachers["without_environment_augmentation"],
        noaug_initial,x,noaug_y,c,root/"variants"/"without_environment_augmentation",alignment=noaug_guidance)
    programs["decision_tree"]=decision_tree(x,y,c.seed,c.max_splits+1)
    print("actual PySR distillation",flush=True)
    programs["pysr"]=fit_pysr(x,ex["probabilities"],FEATURES,root/"pysr",iterations=c.pysr_iterations,seed=c.seed)
    for name,p in programs.items():
        save_json(root/"programs"/f"{name}.json",{"type":type(p).__name__,"program":p.to_dict()})
    print("final held-out distributed alignment",flush=True)
    at=datasets["alignment_test"]
    alignment=distributed_audit(teachers["full"],programs["full"],fit["data"],fit["features"],at["data"],at["features"],
                               root/"alignment",steps=c.mapping_steps,controls=c.random_controls,seed=c.seed)
    evaluation={};environment_alignment={};bounds={}
    basis_state=np.load(root/"alignment"/"mapping.npz") if alignment["status"]=="measured" else None
    for split,d in datasets.items():
        if not split.startswith("test_"):continue
        evaluation[split]={};bounds[split]={}
        saved_predictions={}
        for name,p in programs.items():
            probs=predict(teachers["without_environment_augmentation"],d["data"]) if name=="without_environment_augmentation" else d["probabilities"]
            pred=p.predict(d["features"]);saved_predictions[name]=pred.tolist()
            evaluation[split][name]=evaluate(d["truth"],probs,pred)
            bounds[split][name]=fidelity_bound((pred==probs.argmax(1)).astype(float))
        if basis_state is not None:
            environment_alignment[split]=measure_mapping(teachers["full"],hidden(teachers["full"],d["data"]),
                d["features"],programs["full"],int(basis_state["feature"]),basis_state["basis"],seed=c.seed+100)
        save_json(root/"evaluations"/f"{split}.json",{"predictions":saved_predictions,"metrics":evaluation[split]})
    if basis_state is not None:basis_state.close()
    print("independent active counterexample benchmark",flush=True)
    seeds=generate_worlds("refinement",6,c.seed+999,c.samples);attacks={}
    for guided in (True,False):
        label="guided" if guided else "random"
        result=search_scm(teachers["full"],programs["full"],seeds,objective="joint_error",budget=c.search_budget,
                          guided=guided,seed=c.seed+333,feature_extractor=composed_features)
        result["replicates"]=replicate_counterexamples(teachers["full"],programs["full"],result,top=4,replicates=8,
                                                       feature_extractor=composed_features)
        save_json(root/"attacks"/f"{label}.json",result)
        attacks[label]={k:v for k,v in result.items() if k not in ("records","replicates")}
    summary={"config":asdict(c),"evaluation":evaluation,"alignment":alignment,"environment_alignment":environment_alignment,
             "active_search":attacks,"runtime_seconds":time.time()-start,
             "program_complexity":{n:(p.complexity if hasattr(p,"complexity") else p.to_dict()["complexity"]) for n,p in programs.items()},
             "teacher_hashes":{name:digest(root/"models"/name/"discoverer.pt") for name in teachers},
             "ablation_without_ood_mapping":"without_environment_augmentation; no final test environments used as training data",
             "alignment_guides_synthesis":True,"fidelity_bounds":bounds,
             "empirical_program_equivalence":empirical_equivalence(programs,datasets["test_id"]["features"]),"scientific_claim":"partial empirical mechanism audit, not universal identification"}
    save_json(root/"summary.json",summary);(root/"report.html").write_text(research_report(summary,programs),encoding="utf-8")
    source=root/"source";source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,source/p.name)
    save_json(root/"status.json",{"state":"completed"})
    save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
    print(f"research suite complete: {root}",flush=True);return summary
