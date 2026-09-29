"""End-to-end 3/5/8-variable discovery, local programs and SCM recovery."""
from dataclasses import dataclass,asdict
from pathlib import Path
from html import escape
import shutil
import time
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .graph_model import (pair_features,graph_labels,train_graph_model,load_graph_model,graph_probabilities,
                         decode_graph,dag_completion,graph_metrics,GRAPH_FEATURES,SWAP_LABELS)
from .graphs import pc_stable
from .rules import fit_rule,Rule
from .mechanisms import recover_mechanisms,evaluate_recovery,ExplicitSCM,load_mechanism,neural_values

@dataclass
class GraphConfig:
    seed:int=71
    nodes:tuple=(3,5,8)
    samples:int=96
    train_worlds:int=96
    dev_worlds:int=24
    extraction_worlds:int=48
    refinement_worlds:int=24
    test_worlds:int=24
    epochs:int=30
    width:int=48
    max_splits:int=6
    beam_width:int=3
    mechanism_worlds:int=1
    mechanism_rows:int=512
    mechanism_epochs:int=160

    @classmethod
    def quick(cls):
        return cls(train_worlds=8,dev_worlds=4,extraction_worlds=8,refinement_worlds=4,test_worlds=4,
                   samples=32,epochs=2,width=16,max_splits=2,beam_width=2,mechanism_rows=128,mechanism_epochs=12)

    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Expected 3/5/8-variable tasks")
        if len(set(self.nodes))!=len(self.nodes):raise ValueError("Duplicate node tasks")
        for name,value in asdict(self).items():
            if name=="nodes":continue
            if not isinstance(value,int) or value<1:raise ValueError(f"Invalid {name}")
        if self.samples<16 or self.mechanism_rows<64 or self.width%4:raise ValueError("Invalid model/data sizes")

def symbolic_probabilities(rule,features):
    shape=features.shape[:-1]
    labels=rule.predict(features.reshape(-1,features.shape[-1])).reshape(shape)
    p=np.eye(4)[labels]
    p=.5*(p+p.transpose(0,2,1,3)[...,SWAP_LABELS])
    tied=(p[...,1]==p[...,2])&(p[...,1]>p[...,0])&(p[...,1]>p[...,3])
    p[tied]=[0,0,0,1]
    return p

def flatten_groups(groups,field):
    result=[]
    for g in groups:
        n=g["features"].shape[1];mask=~np.eye(n,dtype=bool)
        values=g[field][:,mask]
        result.append(values.reshape(-1,values.shape[-1]) if field in ("features","probabilities") else values.ravel())
    return np.concatenate(result)

def graph_report(summary,rule):
    rows=[]
    for key,m in summary["evaluation"].items():
        rows.append("<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in [key,m["worlds"],round(m["pair_fidelity"],3),
            round(m["macro_f1"],3),round(m["skeleton_f1"],3),round(m["mean_pair_shd"],3)])+"</tr>")
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多变量与机制恢复</title>
<style>body{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui}td,th{padding:8px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;background:#eef2f6;padding:18px}</style>
<h1>3 / 5 / 8 节点因果图与显式 SCM</h1><p>图推断、图约束、符号机制和真实 SCM 分别评估。
高斯目标为 CPDAG。推断的不确定边用于机制拟合时采用显式标记的一个 DAG completion，不声称方向已被识别。</p>
<p>局部边类别：0=无边，1=i→j，2=j→i，3=方向未定。</p><table><tr><th>环境</th><th>世界数</th><th>图对一致率</th><th>Macro F1</th><th>骨架 F1</th><th>平均 SHD</th></tr>"""+"".join(rows)+"""</table><h2>局部边程序</h2><pre>"""+escape(rule.text())+"""</pre><h2>机制恢复</h2><pre>"""+escape(__import__("json").dumps(summary["mechanisms"],ensure_ascii=False,indent=2))+"</pre></html>"

def run_graphs(directory,config):
    config.validate();root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty graph experiment directory")
    root.mkdir(parents=True,exist_ok=True);start=time.time()
    save_json(root/"config.json",asdict(config));save_json(root/"status.json",{"state":"running"})
    try:return _execute(root,config,start)
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def _execute(root,c,start):
    counts={"train":c.train_worlds,"dev":c.dev_worlds,"extraction":c.extraction_worlds,"refinement":c.refinement_worlds,
            **{name:c.test_worlds for name in ("test_id","test_function","test_noise","test_scale","test_intervention")}}
    groups={name:[] for name in counts};worlds={}
    for nodes in c.nodes:
        for split,count in counts.items():
            print(f"graph dataset n={nodes} {split}: {count}",flush=True)
            ws=generate_graph_worlds(split,count,nodes,c.seed,c.samples);worlds[(nodes,split)]=ws
            path=root/"datasets"/f"n{nodes}_{split}";save_graph_dataset(path,ws)
            features=np.stack([pair_features(w.sample()) for w in ws])
            target=np.stack([w.target_graph for w in ws]);labels=np.stack([graph_labels(t) for t in target])
            np.savez_compressed(path/"features.npz",features=features,labels=labels)
            groups[split].append({"nodes":nodes,"features":features,"labels":labels,"target":target})
    teacher=train_graph_model([(g["features"],g["labels"]) for g in groups["train"]],
                              [(g["features"],g["labels"]) for g in groups["dev"]],root/"model",
                              epochs=c.epochs,width=c.width,seed=c.seed)
    teacher_hash=digest(root/"model"/"graph_teacher.pt")
    for split,gs in groups.items():
        if split=="train":continue
        for g in gs:g["probabilities"]=graph_probabilities(teacher,g["features"])
    x=flatten_groups(groups["extraction"],"features")
    y=flatten_groups(groups["extraction"],"probabilities").argmax(1)
    print("local graph program synthesis",flush=True)
    kwargs={"max_splits":c.max_splits,"beam_width":c.beam_width}
    initial,trace=fit_rule(x,y,GRAPH_FEATURES,**kwargs)
    rx=flatten_groups(groups["refinement"],"features");ry=flatten_groups(groups["refinement"],"probabilities").argmax(1)
    ids=np.flatnonzero(initial.predict(rx)!=ry)[:256]
    rule,refinement_trace=fit_rule(np.r_[x,rx[ids]],np.r_[y,ry[ids]],GRAPH_FEATURES,**kwargs)
    save_json(root/"local_rule.json",{**rule.to_dict(),"teacher_sha256":teacher_hash})
    save_json(root/"initial_rule.json",initial.to_dict())
    save_json(root/"synthesis.json",{"trace":trace,"refinement_trace":refinement_trace,
                                    "refinement_indices":ids.tolist(),"supervision":"teacher_only"})
    evaluation={};pc_evaluation={}
    for split,gs in groups.items():
        if not split.startswith("test_"):continue
        for g in gs:
            symbolic_p=symbolic_probabilities(rule,g["features"])
            neural_graphs=np.stack([decode_graph(p)[0] for p in g["probabilities"]])
            symbolic_graphs=np.stack([decode_graph(p)[0] for p in symbolic_p])
            key=f"n{g['nodes']}_{split}"
            evaluation[key]=graph_metrics(g["target"],neural_graphs,symbolic_graphs)
            pc_records=[pc_stable(w.sample()) for w in worlds[(g["nodes"],split)]]
            pc_graphs=np.stack([r["pdag"] for r in pc_records])
            pc_evaluation[key]=graph_metrics(g["target"],neural_graphs,pc_graphs)
            np.savez_compressed(root/"datasets"/key/"predictions.npz",neural_probabilities=g["probabilities"],
                                symbolic_probabilities=symbolic_p,neural=neural_graphs,symbolic=symbolic_graphs,pc=pc_graphs)
            save_json(root/"datasets"/key/"pc_trace.json",[{**r,"pdag":r["pdag"].astype(int).tolist()} for r in pc_records])
    mechanisms=[]
    for g in groups["test_id"]:
        ws=worlds[(g["nodes"],"test_id")]
        # Preselected non-Gaussian worlds, before examining graph correctness.
        for index in range(1,min(1+c.mechanism_worlds,len(ws))):
            world=ws[index]
            print(f"recover mechanisms n={world.nodes}, index={index}",flush=True)
            partial,projection=decode_graph(g["probabilities"][index]);graph,choices=dag_completion(partial)
            for mode,parents in (("inferred",graph),("oracle_graph_diagnostic",np.array(world.graph,bool))):
                path=root/"mechanisms"/f"n{world.nodes}_{index}_{mode}"
                observations=world.sample(seed=world.seed^1234567,samples=c.mechanism_rows)
                scm,models,records=recover_mechanisms(observations,parents,path,seed=c.seed,epochs=c.mechanism_epochs,query_count=c.mechanism_rows)
                metrics=evaluate_recovery(world,scm,models,records,seed=c.seed+1234,samples=256)
                save_json(path/"world.json",world.metadata());save_json(path/"evaluation.json",metrics)
                save_json(path/"graph_provenance.json",{"mode":mode,"completion_choices":choices if mode=="inferred" else [],
                    "projection":projection if mode=="inferred" else None,"world_index":index,"nodes":world.nodes})
                mechanisms.append({"path":path.relative_to(root).as_posix(),"mode":mode,"nodes":world.nodes,
                    **{k:v for k,v in metrics.items() if k not in ("nodes","interventions")}})
    summary={"config":asdict(c),"teacher_sha256":teacher_hash,"evaluation":evaluation,"pc_baseline":pc_evaluation,
             "mechanisms":mechanisms,"rule_complexity":rule.complexity,"runtime_seconds":time.time()-start,
             "claim":"multivariate implementation with explicit uncertainty; empirical results not mechanism-identification proof"}
    save_json(root/"summary.json",summary);(root/"report.html").write_text(graph_report(summary,rule),encoding="utf-8")
    source=root/"source";source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,source/p.name)
    save_json(root/"status.json",{"state":"completed"})
    save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
    print(f"graph experiment complete: {root}",flush=True);return summary

def verify_graphs(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Graph artifact integrity mismatch")
    c=GraphConfig(**{**summary["config"],"nodes":tuple(summary["config"]["nodes"])})
    from .model import set_seed
    set_seed(c.seed)
    graph_teacher=load_graph_model(root/"model"/"graph_teacher.pt");rule=Rule.from_dict(read_json(root/"local_rule.json"))
    seen=set();world_count=0
    expected_paths={f"n{n}_{split}" for n in c.nodes for split in ("train","dev","extraction","refinement","test_id","test_function","test_noise","test_scale","test_intervention")}
    if {p.name for p in (root/"datasets").iterdir()}!=expected_paths:raise ValueError("Graph split coverage mismatch")
    for path in sorted((root/"datasets").iterdir()):
        ws=load_graph_worlds(path);ids={w.identity for w in ws}
        # Normalize nested tuples to their JSON array representation.
        import json
        expected_metadata=json.loads(json.dumps([w.metadata() for w in ws],allow_nan=False))
        if read_json(path/"worlds.json")!=expected_metadata:raise ValueError("Graph world metadata mismatch")
        nodes=int(path.name.split("_",1)[0][1:]);split=path.name.split("_",1)[1]
        count={"train":c.train_worlds,"dev":c.dev_worlds,"extraction":c.extraction_worlds,"refinement":c.refinement_worlds}.get(split,c.test_worlds)
        if ws!=generate_graph_worlds(split,count,nodes,c.seed,c.samples):raise ValueError("Graph dataset differs from config")
        if len(ids)!=len(ws) or seen&ids:raise ValueError("World leakage")
        seen|=ids;world_count+=len(ws)
        with np.load(path/"samples.npz") as a:
            for i,w in enumerate(ws):
                np.testing.assert_array_equal(w.sample(),a["data"][i])
                np.testing.assert_array_equal(w.sample(interventions={0:1.}),a["hard_intervention"][i])
            np.testing.assert_array_equal(a["target_graph"],np.stack([w.target_graph for w in ws]))
        with np.load(path/"features.npz") as a:features=a["features"]
        np.testing.assert_allclose(features,np.stack([pair_features(w.sample()) for w in ws]),atol=1e-9)
        if (path/"predictions.npz").exists():
            with np.load(path/"predictions.npz") as a:
                p=graph_probabilities(graph_teacher,features);sp=symbolic_probabilities(rule,features)
                np.testing.assert_allclose(p,a["neural_probabilities"],atol=1e-6)
                np.testing.assert_array_equal(sp,a["symbolic_probabilities"])
                ng=np.stack([decode_graph(v)[0] for v in p]);sg=np.stack([decode_graph(v)[0] for v in sp])
                np.testing.assert_array_equal(ng,a["neural"]);np.testing.assert_array_equal(sg,a["symbolic"])
                pc_graphs=np.stack([pc_stable(w.sample())["pdag"] for w in ws])
                np.testing.assert_array_equal(pc_graphs,a["pc"])
                if graph_metrics(np.stack([w.target_graph for w in ws]),ng,pc_graphs)!=summary["pc_baseline"][path.name]:
                    raise ValueError("PC baseline metric mismatch")
                actual=graph_metrics(np.stack([w.target_graph for w in ws]),ng,sg)
                if actual!=summary["evaluation"][path.name]:raise ValueError("Graph metric mismatch")
    from .multiverse import GraphWorld
    for item in summary["mechanisms"]:
        path=root/item["path"];world=GraphWorld.from_dict(read_json(path/"world.json"))
        scm=ExplicitSCM.from_dict(read_json(path/"explicit_scm.json"))
        models=[load_mechanism(path/f"mechanism_{j}.pt") for j in range(world.nodes)]
        provenance=read_json(path/"graph_provenance.json")
        reference=load_graph_worlds(root/"datasets"/f"n{world.nodes}_test_id")[provenance["world_index"]]
        if world!=reference:raise ValueError("Mechanism world provenance mismatch")
        if provenance["mode"]=="inferred":
            pp=graph_probabilities(graph_teacher,np.array([pair_features(world.sample())]))[0]
            expected_graph,choices=dag_completion(decode_graph(pp)[0])
            if choices!=provenance["completion_choices"]:raise ValueError("Ambiguous graph completion mismatch")
        elif provenance["mode"]=="oracle_graph_diagnostic":expected_graph=np.array(world.graph,bool)
        else:raise ValueError("Unknown graph source")
        np.testing.assert_array_equal(scm.source_graph,expected_graph)
        with np.load(path/"observations.npz") as a:
            np.testing.assert_array_equal(a["data"],world.sample(seed=world.seed^1234567,samples=c.mechanism_rows))
        for j,neural_model in enumerate(models):
            if tuple(np.flatnonzero(expected_graph[:,j]))!=neural_model.parents:raise ValueError("Mechanism parent mismatch")
        for j,model in enumerate(models):
            with np.load(path/f"distillation_{j}.npz") as a:
                np.testing.assert_allclose(neural_values(model,a["queries"]),a["teacher"],atol=1e-7)
                from .mechanisms import sparse_symbolic_fit
                recovered,_,_=sparse_symbolic_fit(a["queries"],a["teacher"],model.parents)
                np.testing.assert_allclose(recovered.evaluate(a["queries"]),scm.equations[j].evaluate(a["queries"]),atol=1e-7)
        actual=evaluate_recovery(world,scm,models,read_json(path/"recovery.json"),seed=c.seed+1234,samples=256)
        if actual!=read_json(path/"evaluation.json"):raise ValueError("Mechanism metrics mismatch")
    if (root/"report.html").read_text(encoding="utf-8")!=graph_report(summary,rule):raise ValueError("Graph report mismatch")
    return {"status":"verified","worlds":world_count,"node_sizes":list(c.nodes),"mechanism_recoveries":len(summary["mechanisms"])}
