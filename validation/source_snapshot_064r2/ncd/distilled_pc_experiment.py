"""Teacher-distilled executable PC programs with explicit orientation traces."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .relational_graph import load_relational_graph
from .graph_model import pair_features,graph_probabilities,decode_graph,graph_labels
from .graph_experiment import symbolic_probabilities
from .graphs import pc_stable
from .rules import Rule

MODES=("without_relations","with_relations")
ENVIRONMENTS=("test_id","test_function","test_noise","test_scale","test_intervention")
ALPHAS=(.0001,.0003,.001,.003,.01,.03,.1,.2)

@dataclass
class DistilledPCConfig:
    seed:int=2193
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=32
    samples:int=96
    alphas:tuple=ALPHAS
    max_conditions:tuple=(0,1,2)
    @classmethod
    def quick(cls):return cls(seed=2192,nodes=(3,),environments=("test_id",),worlds_per_cell=2,samples=32,alphas=(.003,.01,.03),max_conditions=(0,1))
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid PC nodes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid PC environments")
        if self.worlds_per_cell<1 or self.samples<16 or not self.alphas or not self.max_conditions:raise ValueError("Invalid PC budget")
        if any(not 0<a<1 for a in self.alphas) or any(k not in (0,1,2) for k in self.max_conditions):raise ValueError("Invalid PC candidates")

def _labels(graphs):
    a=np.asarray(graphs,bool);n=a.shape[1];mask=np.triu(np.ones((n,n),bool),1)
    return np.stack([graph_labels(g)[mask] for g in a])

def _metrics(teacher,program,truth):
    a,b,t=_labels(teacher),_labels(program),_labels(truth);active=(a!=0)|(b!=0)
    return {"worlds":len(a),"all_pair_fidelity":float(np.mean(a==b)),
            "active_pair_fidelity":float(np.mean(a[active]==b[active])) if active.any() else 1.,
            "exact_graph_fidelity":float(np.mean(np.all(a==b,axis=1))),
            "teacher_exact_graph_accuracy":float(np.mean(np.all(a==t,axis=1))),
            "program_exact_graph_accuracy":float(np.mean(np.all(b==t,axis=1)))}

def _json_trace(record):
    return {**record,"pdag":record["pdag"].astype(int).tolist(),"skeleton":record["skeleton"].astype(int).tolist(),
            "collider_pdag":record["collider_pdag"].astype(int).tolist()}

def _source_parts(source,n,split):
    worlds=load_graph_worlds(source/"datasets"/f"n{n}_{split}")
    with np.load(source/"datasets"/f"n{n}_{split}"/"features.npz") as z:features=z["features"]
    with np.load(source/"datasets"/f"n{n}_{split}"/"samples.npz") as z:data=z["data"]
    return worlds,features,data

def _select_programs(source,models,c):
    selected={m:{} for m in MODES};traces={m:{} for m in MODES}
    for n in c.nodes:
        chunks=[_source_parts(source,n,s) for s in ("extraction","refinement")]
        features=np.concatenate([x[1] for x in chunks]);data=np.concatenate([x[2] for x in chunks])
        teachers={m:np.stack([decode_graph(p)[0] for p in graph_probabilities(models[m],features)]) for m in MODES}
        candidates=[]
        for order in c.max_conditions:
            if order>n-2:continue
            for alpha in c.alphas:
                graphs=np.stack([pc_stable(x,alpha=alpha,max_condition=order)["pdag"] for x in data])
                candidates.append((alpha,order,graphs))
        for mode in MODES:
            rows=[]
            for alpha,order,graphs in candidates:
                metric=_metrics(teachers[mode],graphs,graphs)
                rows.append({"alpha":alpha,"max_condition":order,"exact_graph_fidelity":metric["exact_graph_fidelity"],"all_pair_fidelity":metric["all_pair_fidelity"]})
            chosen=max(rows,key=lambda x:(x["exact_graph_fidelity"],x["all_pair_fidelity"],-x["max_condition"],-abs(np.log10(x["alpha"])-np.log10(.01))))
            selected[mode][str(n)]={"alpha":chosen["alpha"],"max_condition":chosen["max_condition"],
                "semantics":"Fisher partial-correlation CI -> sepsets -> collider orientation -> Meek R1/R2/R3"}
            traces[mode][str(n)]=rows
    return selected,traces

def _source_integrity(source,c):
    manifest=read_json(source/"manifest.json")["artifacts"];hashes={}
    for mode in MODES:
        for rel in (f"models/{mode}/graph_teacher.pt",f"programs/{mode}/final.json"):
            p=source/rel
            if digest(p)!=manifest[rel]:raise ValueError("PC source integrity mismatch")
            hashes[rel]=manifest[rel]
    for n in c.nodes:
        for split in ("extraction","refinement"):
            for name in ("features.npz","samples.npz","worlds.json"):
                rel=f"datasets/n{n}_{split}/{name}";p=source/rel
                if digest(p)!=manifest[rel]:raise ValueError("PC selection source mismatch")
                hashes[rel]=manifest[rel]
    return hashes

def run_distilled_pc(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty distilled-PC directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    try:return _execute(root,source,config)
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def _execute(root,source,c):
    hashes=_source_integrity(source,c);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};rules={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES}
    selected,selection_trace=_select_programs(source,models,c);records=[];seen=set()
    for n in c.nodes:
        for environment in c.environments:
            worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples);ids={w.identity for w in worlds}
            if len(ids)!=len(worlds) or seen&ids:raise ValueError("PC world leakage")
            seen|=ids;path=root/"datasets"/f"n{n}_{environment}";save_graph_dataset(path,worlds)
            data=np.stack([w.sample() for w in worlds]);features=np.stack([pair_features(x) for x in data]);truth=np.stack([w.target_graph for w in worlds]);np.savez_compressed(path/"features.npz",features=features)
            for mode in MODES:
                teacher_p=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in teacher_p]);local_p=symbolic_probabilities(rules[mode],features);local=np.stack([decode_graph(x)[0] for x in local_p])
                spec=selected[mode][str(n)];pc_records=[pc_stable(x,alpha=spec["alpha"],max_condition=spec["max_condition"]) for x in data];pc=np.stack([x["pdag"] for x in pc_records])
                np.savez_compressed(path/f"{mode}.npz",teacher_probabilities=teacher_p,teacher=teacher,local=local,pc=pc)
                save_json(path/f"{mode}_pc_trace.json",[_json_trace(x) for x in pc_records])
                families=sorted({w.family for w in worlds});family={f:{"local":_metrics(teacher[[w.family==f for w in worlds]],local[[w.family==f for w in worlds]],truth[[w.family==f for w in worlds]]),"pc":_metrics(teacher[[w.family==f for w in worlds]],pc[[w.family==f for w in worlds]],truth[[w.family==f for w in worlds]])} for f in families}
                records.append({"nodes":n,"environment":environment,"mode":mode,"program":spec,"local":_metrics(teacher,local,truth),"pc":_metrics(teacher,pc,truth),"family":family})
    aggregate={method:{key:float(np.mean([r[method][key] for r in records])) for key in ("all_pair_fidelity","active_pair_fidelity","exact_graph_fidelity","program_exact_graph_accuracy")} for method in ("local","pc")}
    summary={"config":{**asdict(c),"source":str(source)},"source_hashes":hashes,"selected_programs":selected,"selection_trace":selection_trace,"world_count":len(seen),"records":records,"aggregate":aggregate,"teacher_fidelity_is_primary":True,"truth_is_diagnostic":True};save_json(root/"summary.json",summary)
    src=root/"source";src.mkdir();[shutil.copy2(p,src/p.name) for p in Path(__file__).parent.glob("*.py")]
    save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary

def _close(a,b,path="root"):
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError(f"PC keys differ at {path}")
        for k in a:_close(a[k],b[k],path+"."+str(k))
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError(f"PC length differs at {path}")
        for i,(x,y) in enumerate(zip(a,b)):_close(x,y,f"{path}[{i}]")
    elif isinstance(a,float) or isinstance(b,float):
        if not np.isclose(a,b,rtol=1e-7,atol=1e-9):raise ValueError(f"PC float differs at {path}")
    elif a!=b:raise ValueError(f"PC value differs at {path}")

def verify_distilled_pc(directory):
    root=Path(directory).resolve();saved=read_json(root/"summary.json")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        if digest(root/name)!=value:raise ValueError("PC artifact mismatch")
    raw=saved["config"];source=Path(raw.pop("source"));c=DistilledPCConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"]),"alphas":tuple(raw["alphas"]),"max_conditions":tuple(raw["max_conditions"])})
    models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};rules={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};selected,trace=_select_programs(source,models,c);_close(selected,saved["selected_programs"]);_close(trace,saved["selection_trace"]);records=[];seen=set()
    for n in c.nodes:
        for environment in c.environments:
            path=root/"datasets"/f"n{n}_{environment}";worlds=load_graph_worlds(path)
            if worlds!=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples):raise ValueError("PC world mismatch")
            seen|={w.identity for w in worlds};data=np.stack([w.sample() for w in worlds]);features=np.stack([pair_features(x) for x in data]);truth=np.stack([w.target_graph for w in worlds])
            for mode in MODES:
                tp=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in tp]);local=np.stack([decode_graph(x)[0] for x in symbolic_probabilities(rules[mode],features)]);spec=selected[mode][str(n)];pcs=[pc_stable(x,alpha=spec["alpha"],max_condition=spec["max_condition"]) for x in data];pc=np.stack([x["pdag"] for x in pcs])
                with np.load(path/f"{mode}.npz") as z:np.testing.assert_allclose(z["teacher_probabilities"],tp,rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(z["teacher"],teacher);np.testing.assert_array_equal(z["local"],local);np.testing.assert_array_equal(z["pc"],pc)
                _close([_json_trace(x) for x in pcs],read_json(path/f"{mode}_pc_trace.json"),"pc_trace")
                families=sorted({w.family for w in worlds});family={f:{"local":_metrics(teacher[[w.family==f for w in worlds]],local[[w.family==f for w in worlds]],truth[[w.family==f for w in worlds]]),"pc":_metrics(teacher[[w.family==f for w in worlds]],pc[[w.family==f for w in worlds]],truth[[w.family==f for w in worlds]])} for f in families};records.append({"nodes":n,"environment":environment,"mode":mode,"program":spec,"local":_metrics(teacher,local,truth),"pc":_metrics(teacher,pc,truth),"family":family})
    _close(records,saved["records"],"records");return {"status":"verified","worlds":len(seen),"strata":len(records),"science_not_certified":True}