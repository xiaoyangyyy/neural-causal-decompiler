"""Paired incidence-attention experiment: graphs, programs, parents and SCMs."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .model import set_seed
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds,GraphWorld
from .graph_model import pair_features,graph_labels,graph_probabilities,decode_graph,dag_completion,graph_metrics,GRAPH_FEATURES
from .graph_experiment import symbolic_probabilities,flatten_groups
from .relational_graph import train_relational_graph,load_relational_graph
from .rules import Rule,fit_rule
from .mechanisms import recover_mechanisms,evaluate_recovery,ExplicitSCM,load_mechanism,neural_values,sparse_symbolic_fit

MODES=("without_relations","with_relations")
TESTS=("test_id","test_function","test_noise","test_scale","test_intervention")

@dataclass
class RelationalConfig:
    seed:int=493
    nodes:tuple=(3,5,8)
    samples:int=96
    train_worlds:int=256
    dev_worlds:int=64
    extraction_worlds:int=128
    refinement_worlds:int=64
    test_worlds:int=96
    epochs:int=40
    width:int=48
    splits:int=6
    mechanism_worlds:int=2
    mechanism_rows:int=512
    mechanism_epochs:int=160
    @classmethod
    def quick(cls):
        return cls(seed=491,samples=32,train_worlds=8,dev_worlds=4,extraction_worlds=8,refinement_worlds=4,
                   test_worlds=4,epochs=2,width=16,splits=2,mechanism_worlds=1,mechanism_rows=128,mechanism_epochs=12)

def split_counts(c):
    return {"train":c.train_worlds,"dev":c.dev_worlds,"extraction":c.extraction_worlds,
            "refinement":c.refinement_worlds,**{s:c.test_worlds for s in TESTS}}

def decoded(probabilities):
    projected=[];raw=[];logs=[]
    for p in probabilities:
        graph,record=decode_graph(p);projected.append(graph);raw.append(np.array(record["raw_graph"],bool));logs.append(record)
    return np.stack(raw),np.stack(projected),logs

def metrics(truth,neural,neural_raw,symbolic,symbolic_raw,nlogs,slogs):
    return {"neural_raw":graph_metrics(truth,neural_raw,neural_raw),
            "neural_projected":graph_metrics(truth,neural,neural),
            "symbolic_raw":graph_metrics(truth,neural_raw,symbolic_raw),
            "symbolic_projected":graph_metrics(truth,neural,symbolic),
            "neural_removed_cycle_edges":sum(len(r["removed_cycle_edges"]) for r in nlogs),
            "symbolic_removed_cycle_edges":sum(len(r["removed_cycle_edges"]) for r in slogs)}

def run_relational(directory,c):
    root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use a new relational run")
    if tuple(c.nodes)!=(3,5,8) or c.width%4 or c.mechanism_worlds>=c.test_worlds:raise ValueError("Invalid graph protocol")
    if any(type(v) is not int or v<1 for k,v in asdict(c).items() if k!="nodes"):raise ValueError("Invalid budget")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    save_json(root/"config.json",asdict(c));set_seed(c.seed)
    try:
        groups={split:[] for split in split_counts(c)};worlds={}
        for n in c.nodes:
            for split,count in split_counts(c).items():
                print(f"shared worlds n={n} {split}: {count}",flush=True)
                ws=generate_graph_worlds(split,count,n,c.seed,c.samples);worlds[(n,split)]=ws
                path=root/"datasets"/f"n{n}_{split}";save_graph_dataset(path,ws)
                features=np.stack([pair_features(w.sample()) for w in ws])
                target=np.stack([w.target_graph for w in ws]);labels=np.stack([graph_labels(t) for t in target])
                np.savez_compressed(path/"features.npz",features=features,labels=labels)
                groups[split].append({"nodes":n,"features":features,"labels":labels,"target":target})
        models={};rules={};evaluation={};predicted={}
        for mode in MODES:
            print("train controlled model: "+mode,flush=True)
            model=train_relational_graph([(g["features"],g["labels"]) for g in groups["train"]],
                [(g["features"],g["labels"]) for g in groups["dev"]],root/"models"/mode,
                relations=mode=="with_relations",epochs=c.epochs,width=c.width,seed=c.seed)
            models[mode]=model
            local={}
            for split in ("extraction","refinement"):
                local[split]=[{**g,"probabilities":graph_probabilities(model,g["features"])} for g in groups[split]]
            x=flatten_groups(local["extraction"],"features");y=flatten_groups(local["extraction"],"probabilities").argmax(1)
            kwargs={"max_splits":c.splits,"beam_width":3}
            initial,trace=fit_rule(x,y,GRAPH_FEATURES,**kwargs)
            rx=flatten_groups(local["refinement"],"features");ry=flatten_groups(local["refinement"],"probabilities").argmax(1)
            ids=np.flatnonzero(initial.predict(rx)!=ry)[:256]
            rule,refinement=fit_rule(np.r_[x,rx[ids]],np.r_[y,ry[ids]],GRAPH_FEATURES,**kwargs);rules[mode]=rule
            save_json(root/"programs"/mode/"initial.json",initial.to_dict());save_json(root/"programs"/mode/"final.json",rule.to_dict())
            save_json(root/"programs"/mode/"synthesis.json",{"trace":trace,"refinement":refinement,
                "refinement_indices":ids.tolist(),"supervision":"frozen_teacher_only"})
            evaluation[mode]={}
            for split in TESTS:
                for g in groups[split]:
                    n=g["nodes"];key=f"n{n}_{split}";p=graph_probabilities(model,g["features"])
                    sp=symbolic_probabilities(rule,g["features"])
                    nr,ng,nlog=decoded(p);sr,sg,slog=decoded(sp)
                    path=root/"evaluations"/mode/key;path.mkdir(parents=True)
                    np.savez_compressed(path/"predictions.npz",neural_probabilities=p,symbolic_probabilities=sp,
                                        neural_raw=nr,neural=ng,symbolic_raw=sr,symbolic=sg)
                    save_json(path/"projection.json",{"neural":nlog,"symbolic":slog})
                    evaluation[mode][key]=metrics(g["target"],ng,nr,sg,sr,nlog,slog)
                    predicted[(mode,n,split)]=p
        mechanisms=[]
        for n in c.nodes:
            for index in range(1,1+c.mechanism_worlds):
                world=worlds[(n,"test_id")][index]
                cases=[("oracle_graph_diagnostic",np.array(world.graph,bool),[],None)]
                for mode in MODES:
                    partial,projection=decode_graph(predicted[(mode,n,"test_id")][index])
                    graph,choices=dag_completion(partial);cases.append((mode,graph,choices,projection))
                for mode,parents,choices,projection in cases:
                    print(f"mechanism n={n} world={index} source={mode}",flush=True)
                    path=root/"mechanisms"/f"n{n}_{index}_{mode}"
                    observations=world.sample(seed=world.seed^1234567,samples=c.mechanism_rows)
                    scm,neural,records=recover_mechanisms(observations,parents,path,seed=c.seed,
                        epochs=c.mechanism_epochs,query_count=c.mechanism_rows)
                    result=evaluate_recovery(world,scm,neural,records,seed=c.seed+1234,samples=256)
                    save_json(path/"world.json",world.metadata());save_json(path/"evaluation.json",result)
                    save_json(path/"graph_provenance.json",{"mode":mode,"world_index":index,"nodes":n,
                        "completion_choices":choices,"projection":projection})
                    mechanisms.append({"path":path.relative_to(root).as_posix(),"mode":mode,"nodes":n,
                        **{k:v for k,v in result.items() if k not in ("nodes","interventions")}})
        summary={"config":asdict(c),"evaluation":evaluation,"mechanisms":mechanisms,
            "parameter_counts":{m:sum(p.numel() for p in model.parameters()) for m,model in models.items()},
            "teacher_hashes":{m:digest(root/"models"/m/"graph_teacher.pt") for m in MODES},
            "rule_complexity":{m:r.complexity for m,r in rules.items()},
            "control":"same worlds, initial shared weights, allocated parameters, optimizer, epochs and decoding; relation bias gated on/off",
            "effective_capacity_note":"enabled relation parameters receive informative gradients; disabled relation biases do not",
            "claim":"new-architecture comparison, not recovered mechanisms of historical frozen teachers"}
        save_json(root/"summary.json",summary);save_json(root/"status.json",{"state":"completed"})
        snapshot=root/"source";snapshot.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snapshot/p.name)
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","error":str(exc)});raise

def verify_relational(directory):
    root=Path(directory).resolve()
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Relational artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete run")
    summary=read_json(root/"summary.json");cfg=read_json(root/"config.json");c=RelationalConfig(**{**cfg,"nodes":tuple(cfg["nodes"])})
    if cfg!=summary["config"] or set(summary["evaluation"])!=set(MODES):raise ValueError("Protocol mismatch")
    set_seed(c.seed);models={m:load_relational_graph(root/"models"/m/"graph_teacher.pt") for m in MODES}
    rules={m:Rule.from_dict(read_json(root/"programs"/m/"final.json")) for m in MODES}
    worlds={};groups={s:[] for s in split_counts(c)};seen=set()
    expected_paths={f"n{n}_{s}" for n in c.nodes for s in split_counts(c)}
    if {p.name for p in (root/"datasets").iterdir()}!=expected_paths:raise ValueError("Dataset coverage mismatch")
    for n in c.nodes:
        for split,count in split_counts(c).items():
            path=root/"datasets"/f"n{n}_{split}";ws=load_graph_worlds(path)
            if ws!=generate_graph_worlds(split,count,n,c.seed,c.samples):raise ValueError("World protocol mismatch")
            if read_json(path/"worlds.json")!=json.loads(json.dumps([w.metadata() for w in ws])):raise ValueError("World metadata mismatch")
            ids={w.identity for w in ws}
            if len(ids)!=len(ws) or seen&ids:raise ValueError("World leakage")
            seen|=ids;worlds[(n,split)]=ws
            target=np.stack([w.target_graph for w in ws]);data=np.stack([w.sample() for w in ws])
            features=np.stack([pair_features(d) for d in data]);labels=np.stack([graph_labels(t) for t in target])
            with np.load(path/"samples.npz") as a:
                np.testing.assert_array_equal(data,a["data"]);np.testing.assert_array_equal(target,a["target_graph"])
                np.testing.assert_array_equal(np.stack([w.sample(interventions={0:1.}) for w in ws]),a["hard_intervention"])
            with np.load(path/"features.npz") as a:
                np.testing.assert_allclose(features,a["features"],atol=1e-9);np.testing.assert_array_equal(labels,a["labels"])
            groups[split].append({"nodes":n,"features":features,"labels":labels,"target":target})
    if len(set(summary["parameter_counts"].values()))!=1:raise ValueError("Unmatched parameter allocation")
    flat=np.concatenate([g["features"][:,~np.eye(g["nodes"],dtype=bool)].reshape(-1,len(GRAPH_FEATURES)) for g in groups["train"]])
    predicted={}
    for mode,model in models.items():
        if model.width!=c.width:raise ValueError("Model width mismatch")
        np.testing.assert_array_equal(model.mean.numpy(),flat.mean(0).astype(np.float32))
        np.testing.assert_array_equal(model.std.numpy(),flat.std(0).clip(.03).astype(np.float32))
        training=read_json(root/"models"/mode/"training.json");history=training["history"]
        if training["seed"]!=c.seed or training["relations"]!=(mode=="with_relations") or [r["epoch"] for r in history]!=list(range(1,c.epochs+1)):
            raise ValueError("Training protocol mismatch")
        best=1+int(np.argmin([r["dev_cross_entropy"] for r in history]))
        if best!=training["best_epoch"]:raise ValueError("Dev checkpoint selection mismatch")
        dev_losses=[]
        for g in groups["dev"]:
            p=graph_probabilities(model,g["features"]);mask=~np.eye(g["nodes"],dtype=bool)
            dev_losses.append(float(-np.log(np.take_along_axis(p[:,mask],g["labels"][:,mask,None],axis=-1).clip(1e-9)).mean()))
        np.testing.assert_allclose(np.mean(dev_losses),history[best-1]["dev_cross_entropy"],rtol=1e-7,atol=1e-8)
        np.testing.assert_array_equal(model.relation_bias.detach().numpy(),training["relation_bias"])
        if mode=="without_relations" and np.any(model.relation_bias.detach().numpy()!=0):raise ValueError("Disabled relation control changed")

        if model.relations_enabled!=(mode=="with_relations") or digest(root/"models"/mode/"graph_teacher.pt")!=summary["teacher_hashes"][mode]:
            raise ValueError("Teacher/control mismatch")
        if sum(p.numel() for p in model.parameters())!=summary["parameter_counts"][mode]:raise ValueError("Parameter mismatch")
        local={s:[{**g,"probabilities":graph_probabilities(model,g["features"])} for g in groups[s]] for s in ("extraction","refinement")}
        x=flatten_groups(local["extraction"],"features");y=flatten_groups(local["extraction"],"probabilities").argmax(1)
        kwargs={"max_splits":c.splits,"beam_width":3}
        initial,trace=fit_rule(x,y,GRAPH_FEATURES,**kwargs)
        rx=flatten_groups(local["refinement"],"features");ry=flatten_groups(local["refinement"],"probabilities").argmax(1)
        ids=np.flatnonzero(initial.predict(rx)!=ry)[:256]
        recovered,refinement=fit_rule(np.r_[x,rx[ids]],np.r_[y,ry[ids]],GRAPH_FEATURES,**kwargs)
        if initial.to_dict()!=read_json(root/"programs"/mode/"initial.json") or recovered.to_dict()!=rules[mode].to_dict():raise ValueError("Program synthesis replay mismatch")
        expected={"trace":trace,"refinement":refinement,"refinement_indices":ids.tolist(),"supervision":"frozen_teacher_only"}
        if expected!=read_json(root/"programs"/mode/"synthesis.json"):raise ValueError("Synthesis provenance mismatch")
        if rules[mode].complexity!=summary["rule_complexity"][mode]:raise ValueError("Rule complexity mismatch")
        if set(summary["evaluation"][mode])!={f"n{n}_{s}" for n in c.nodes for s in TESTS}:raise ValueError("Evaluation coverage mismatch")
        for split in TESTS:
            for g in groups[split]:
                n=g["nodes"];key=f"n{n}_{split}";path=root/"evaluations"/mode/key
                p=graph_probabilities(model,g["features"]);sp=symbolic_probabilities(rules[mode],g["features"])
                nr,ng,nlog=decoded(p);sr,sg,slog=decoded(sp)
                with np.load(path/"predictions.npz") as a:
                    for name,value in (("neural_probabilities",p),("symbolic_probabilities",sp),("neural_raw",nr),("neural",ng),("symbolic_raw",sr),("symbolic",sg)):
                        np.testing.assert_allclose(value,a[name],atol=1e-7)
                if {"neural":nlog,"symbolic":slog}!=read_json(path/"projection.json"):raise ValueError("Projection replay mismatch")
                if metrics(g["target"],ng,nr,sg,sr,nlog,slog)!=summary["evaluation"][mode][key]:raise ValueError("Graph metric mismatch")
                predicted[(mode,n,split)]=p
    expected_mechanisms={f"mechanisms/n{n}_{i}_{m}" for n in c.nodes for i in range(1,1+c.mechanism_worlds) for m in (*MODES,"oracle_graph_diagnostic")}
    if {m["path"] for m in summary["mechanisms"]}!=expected_mechanisms:raise ValueError("Mechanism coverage mismatch")
    for item in summary["mechanisms"]:
        path=root/item["path"];provenance=read_json(path/"graph_provenance.json")
        n=provenance["nodes"];i=provenance["world_index"];mode=provenance["mode"]
        world=worlds[(n,"test_id")][i]
        if GraphWorld.from_dict(read_json(path/"world.json"))!=world:raise ValueError("Mechanism world mismatch")
        if mode=="oracle_graph_diagnostic":graph=np.array(world.graph,bool);choices=[];projection=None
        else:
            partial,projection=decode_graph(predicted[(mode,n,"test_id")][i]);graph,choices=dag_completion(partial)
        if choices!=provenance["completion_choices"] or projection!=provenance["projection"]:raise ValueError("Graph source mismatch")
        scm=ExplicitSCM.from_dict(read_json(path/"explicit_scm.json"))
        np.testing.assert_array_equal(scm.source_graph,graph)
        neural=[load_mechanism(path/f"mechanism_{j}.pt") for j in range(n)]
        with np.load(path/"observations.npz") as a:np.testing.assert_array_equal(a["data"],world.sample(seed=world.seed^1234567,samples=c.mechanism_rows))
        for j,model in enumerate(neural):
            if model.parents!=tuple(np.flatnonzero(graph[:,j])):raise ValueError("Mechanism parents mismatch")
            with np.load(path/f"distillation_{j}.npz") as a:
                np.testing.assert_allclose(neural_values(model,a["queries"]),a["teacher"],atol=1e-7)
                expression,_,_=sparse_symbolic_fit(a["queries"],a["teacher"],model.parents)
                np.testing.assert_allclose(expression.evaluate(a["queries"]),scm.equations[j].evaluate(a["queries"]),atol=1e-7)
        result=evaluate_recovery(world,scm,neural,read_json(path/"recovery.json"),seed=c.seed+1234,samples=256)
        if result!=read_json(path/"evaluation.json"):raise ValueError("Mechanism metric mismatch")
        expected={"path":item["path"],"mode":mode,"nodes":n,**{k:v for k,v in result.items() if k not in ("nodes","interventions")}}
        if expected!=item:raise ValueError("Mechanism summary mismatch")
    return {"status":"verified","worlds":len(seen),"modes":list(MODES),"node_sizes":list(c.nodes),
            "mechanism_recoveries":len(summary["mechanisms"]),"science_not_certified":True}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("command",choices=("run","verify"));p.add_argument("directory",type=Path)
    p.add_argument("--quick",action="store_true");p.add_argument("--seed",type=int)
    a=p.parse_args()
    if a.command=="verify":result=verify_relational(a.directory)
    else:
        c=RelationalConfig.quick() if a.quick else RelationalConfig()
        if a.seed is not None:c.seed=a.seed
        s=run_relational(a.directory,c);result={"output":str(a.directory),"mechanism_recoveries":len(s["mechanisms"])}
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
