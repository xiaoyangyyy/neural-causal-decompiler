"""Teacher-only factorized skeleton/orientation graph-program experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .relational_graph import load_relational_graph
from .graph_model import pair_features,graph_probabilities,decode_graph,graph_labels,GRAPH_FEATURES
from .graph_experiment import symbolic_probabilities
from .factorized_graph_program import FactorizedGraphProgram
from .rules import Rule,fit_rule

MODES=("without_relations","with_relations")
ENVIRONMENTS=("test_id","test_function","test_noise","test_scale","test_intervention")

@dataclass
class FactorizedGraphConfig:
    seed:int=2393
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=32
    samples:int=96
    max_splits:int=6
    beam_width:int=3
    penalty:float=.001
    @classmethod
    def quick(cls):return cls(seed=2392,nodes=(3,),environments=("test_id",),worlds_per_cell=2,samples=32,max_splits=2)
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid factorized nodes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid factorized environments")
        if self.worlds_per_cell<1 or self.samples<16 or self.max_splits<1 or self.beam_width<1 or self.penalty<0:raise ValueError("Invalid factorized budget")

def _labels(graphs):
    a=np.asarray(graphs,bool);n=a.shape[1];mask=np.triu(np.ones((n,n),bool),1)
    return np.stack([graph_labels(g)[mask] for g in a])

def _metrics(teacher,program,truth):
    a,b,t=_labels(teacher),_labels(program),_labels(truth);active=(a!=0)|(b!=0)
    return {"worlds":len(a),"pairs":int(a.size),"all_pair_fidelity":float(np.mean(a==b)),
      "active_pair_fidelity":float(np.mean(a[active]==b[active])) if active.any() else 1.,
      "exact_graph_fidelity":float(np.mean(np.all(a==b,axis=1))),
      "teacher_exact_graph_accuracy":float(np.mean(np.all(a==t,axis=1))),
      "program_exact_graph_accuracy":float(np.mean(np.all(b==t,axis=1)))}

def _pooled_metrics(teachers,programs,truths):
    triples=[(_labels(a),_labels(b),_labels(t)) for a,b,t in zip(teachers,programs,truths)]
    af=np.concatenate([a.ravel() for a,b,t in triples]);bf=np.concatenate([b.ravel() for a,b,t in triples]);tf=np.concatenate([t.ravel() for a,b,t in triples]);active=(af!=0)|(bf!=0)
    return {"worlds":int(sum(len(a) for a,b,t in triples)),"pairs":int(len(af)),"all_pair_fidelity":float(np.mean(af==bf)),
      "active_pair_fidelity":float(np.mean(af[active]==bf[active])) if active.any() else 1.,
      "exact_graph_fidelity":float(np.mean(np.concatenate([np.all(a==b,axis=1) for a,b,t in triples]))),
      "teacher_exact_graph_accuracy":float(np.mean(np.concatenate([np.all(a==t,axis=1) for a,b,t in triples]))),
      "program_exact_graph_accuracy":float(np.mean(np.concatenate([np.all(b==t,axis=1) for a,b,t in triples])))}
def _source_features(source,n,split):
    worlds=load_graph_worlds(source/"datasets"/f"n{n}_{split}")
    with np.load(source/"datasets"/f"n{n}_{split}"/"features.npz") as z:features=z["features"]
    return worlds,features

def _source_integrity(source,c):
    manifest=read_json(source/"manifest.json")["artifacts"];hashes={}
    for mode in MODES:
        for rel in (f"models/{mode}/graph_teacher.pt",f"programs/{mode}/final.json"):
            if digest(source/rel)!=manifest[rel]:raise ValueError("Factorized source integrity mismatch")
            hashes[rel]=manifest[rel]
    for n in c.nodes:
        for split in ("extraction","refinement"):
            for name in ("features.npz","worlds.json"):
                rel=f"datasets/n{n}_{split}/{name}"
                if digest(source/rel)!=manifest[rel]:raise ValueError("Factorized source data mismatch")
                hashes[rel]=manifest[rel]
    return hashes

def _flatten(features):
    n=features.shape[1];return features[:,~np.eye(n,dtype=bool)].reshape(-1,len(GRAPH_FEATURES))

def _train_programs(source,models,c):
    extraction=[];refinement=[]
    for n in c.nodes:
        extraction.append(_source_features(source,n,"extraction")[1]);refinement.append(_source_features(source,n,"refinement")[1])
    programs={};traces={}
    for mode in MODES:
        xs=[];ys=[]
        for features in extraction:
            labels=graph_probabilities(models[mode],features).argmax(-1);n=features.shape[1];mask=~np.eye(n,dtype=bool)
            xs.append(features[:,mask].reshape(-1,len(GRAPH_FEATURES)));ys.append(labels[:,mask].reshape(-1))
        x=np.concatenate(xs);y=np.concatenate(ys);kwargs=dict(max_splits=c.max_splits,beam_width=c.beam_width,penalty=c.penalty,arithmetic=True)
        skeleton,skeleton_trace=fit_rule(x,(y!=0).astype(int),GRAPH_FEATURES,**kwargs)
        orientation,orientation_trace=fit_rule(x[y!=0],y[y!=0],GRAPH_FEATURES,**kwargs)
        rows=[]
        for aggregation in ("and","or"):
            p=FactorizedGraphProgram(skeleton,orientation,aggregation);teacher=[];pred=[]
            for features in refinement:
                teacher.append(np.stack([decode_graph(z)[0] for z in graph_probabilities(models[mode],features)]));pred.append(np.stack([p.predict(z) for z in features]))
            metric=_pooled_metrics(teacher,pred,teacher)
            rows.append({"aggregation":aggregation,"exact_graph_fidelity":metric["exact_graph_fidelity"],"all_pair_fidelity":metric["all_pair_fidelity"]})
        chosen=max(rows,key=lambda z:(z["exact_graph_fidelity"],z["all_pair_fidelity"],z["aggregation"]=="and"))
        programs[mode]=FactorizedGraphProgram(skeleton,orientation,chosen["aggregation"])
        traces[mode]={"skeleton":skeleton_trace,"orientation":orientation_trace,"orientation_training_rows":int(np.sum(y!=0)),"aggregation_candidates":rows,"selected":chosen["aggregation"],"supervision":"frozen_teacher_only"}
    return programs,traces

def run_factorized_graph(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty factorized-graph directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    try:return _execute(root,source,config)
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def _execute(root,source,c):
    hashes=_source_integrity(source,c);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};baseline={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES}
    programs,traces=_train_programs(source,models,c)
    for mode,p in programs.items():
        path=root/"programs"/mode;path.mkdir(parents=True);save_json(path/"factorized.json",p.to_dict());save_json(path/"synthesis.json",traces[mode])
    records=[];pooled={m:{k:[] for k in ("teacher","local","factorized","truth")} for m in MODES};seen=set()
    for n in c.nodes:
      for environment in c.environments:
        worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples);ids={w.identity for w in worlds}
        if len(ids)!=len(worlds) or seen&ids:raise ValueError("Factorized world leakage")
        seen|=ids;path=root/"datasets"/f"n{n}_{environment}";save_graph_dataset(path,worlds)
        data=np.stack([w.sample() for w in worlds]);features=np.stack([pair_features(x) for x in data]);truth=np.stack([w.target_graph for w in worlds]);np.savez_compressed(path/"features.npz",features=features)
        for mode in MODES:
          tp=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in tp]);local=np.stack([decode_graph(x)[0] for x in symbolic_probabilities(baseline[mode],features)]);factorized=np.stack([programs[mode].predict(x) for x in features])
          np.savez_compressed(path/f"{mode}.npz",teacher_probabilities=tp,teacher=teacher,local=local,factorized=factorized)
          for key,value in (("teacher",teacher),("local",local),("factorized",factorized),("truth",truth)):pooled[mode][key].append(value)
          family={}
          for f in sorted({w.family for w in worlds}):
            select=np.array([w.family==f for w in worlds]);family[f]={"local":_metrics(teacher[select],local[select],truth[select]),"factorized":_metrics(teacher[select],factorized[select],truth[select])}
          records.append({"nodes":n,"environment":environment,"mode":mode,"local":_metrics(teacher,local,truth),"factorized":_metrics(teacher,factorized,truth),"family":family})
    aggregate={}
    for mode,parts in pooled.items():aggregate[mode]={method:_pooled_metrics(parts["teacher"],parts[method],parts["truth"]) for method in ("local","factorized")}
    overall={method:_pooled_metrics([x for m in MODES for x in pooled[m]["teacher"]],[x for m in MODES for x in pooled[m][method]],[x for m in MODES for x in pooled[m]["truth"]]) for method in ("local","factorized")}
    decision={"per_mode_non_decline":all(aggregate[m]["factorized"]["exact_graph_fidelity"]>=aggregate[m]["local"]["exact_graph_fidelity"] for m in MODES),"pooled_exact_gain":overall["factorized"]["exact_graph_fidelity"]-overall["local"]["exact_graph_fidelity"],"pooled_active_non_decline":overall["factorized"]["active_pair_fidelity"]>=overall["local"]["active_pair_fidelity"]}
    summary={"config":{**asdict(c),"source":str(source)},"source_hashes":hashes,"programs":{m:programs[m].to_dict() for m in MODES},"synthesis":traces,"world_count":len(seen),"records":records,"aggregate":aggregate,"overall":overall,"decision_components":decision,"teacher_fidelity_is_primary":True,"truth_is_diagnostic":True};save_json(root/"summary.json",summary)
    snap=root/"source";snap.mkdir();[shutil.copy2(p,snap/p.name) for p in Path(__file__).parent.glob("*.py")]
    save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary

def _close(a,b,path="root"):
    if isinstance(a,dict):
      if a.keys()!=b.keys():raise ValueError(f"Factorized keys differ at {path}")
      for k in a:_close(a[k],b[k],path+"."+str(k))
    elif isinstance(a,list):
      if len(a)!=len(b):raise ValueError(f"Factorized length differs at {path}")
      for i,(x,y) in enumerate(zip(a,b)):_close(x,y,f"{path}[{i}]")
    elif isinstance(a,float) or isinstance(b,float):
      if not np.isclose(a,b,rtol=1e-7,atol=1e-9):raise ValueError(f"Factorized float differs at {path}")
    elif a!=b:raise ValueError(f"Factorized value differs at {path}")

def verify_factorized_graph(directory):
    root=Path(directory).resolve();saved=read_json(root/"summary.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete factorized run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
      p=(root/name).resolve()
      if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Factorized artifact mismatch")
    raw=dict(saved["config"]);source=Path(raw.pop("source"));c=FactorizedGraphConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"])})
    _source_integrity(source,c);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};baseline={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};programs,traces=_train_programs(source,models,c)
    _close(traces,saved["synthesis"],"synthesis");records=[];pooled={m:{k:[] for k in ("teacher","local","factorized","truth")} for m in MODES};seen=set()
    for n in c.nodes:
      for environment in c.environments:
        path=root/"datasets"/f"n{n}_{environment}";worlds=load_graph_worlds(path)
        if worlds!=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples):raise ValueError("Factorized world mismatch")
        ids={w.identity for w in worlds}
        if len(ids)!=len(worlds) or seen&ids:raise ValueError("Factorized replay leakage")
        seen|=ids;features=np.stack([pair_features(w.sample()) for w in worlds]);truth=np.stack([w.target_graph for w in worlds])
        with np.load(path/"features.npz") as z:np.testing.assert_allclose(z["features"],features,atol=1e-9)
        for mode in MODES:
          tp=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in tp]);local=np.stack([decode_graph(x)[0] for x in symbolic_probabilities(baseline[mode],features)]);factorized=np.stack([programs[mode].predict(x) for x in features])
          with np.load(path/f"{mode}.npz") as z:np.testing.assert_allclose(z["teacher_probabilities"],tp,rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(z["teacher"],teacher);np.testing.assert_array_equal(z["local"],local);np.testing.assert_array_equal(z["factorized"],factorized)
          for key,value in (("teacher",teacher),("local",local),("factorized",factorized),("truth",truth)):pooled[mode][key].append(value)
          family={}
          for f in sorted({w.family for w in worlds}):
            select=np.array([w.family==f for w in worlds]);family[f]={"local":_metrics(teacher[select],local[select],truth[select]),"factorized":_metrics(teacher[select],factorized[select],truth[select])}
          records.append({"nodes":n,"environment":environment,"mode":mode,"local":_metrics(teacher,local,truth),"factorized":_metrics(teacher,factorized,truth),"family":family})
    aggregate={m:{method:_pooled_metrics(pooled[m]["teacher"],pooled[m][method],pooled[m]["truth"]) for method in ("local","factorized")} for m in MODES};overall={method:_pooled_metrics([x for m in MODES for x in pooled[m]["teacher"]],[x for m in MODES for x in pooled[m][method]],[x for m in MODES for x in pooled[m]["truth"]]) for method in ("local","factorized")}
    decision={"per_mode_non_decline":all(aggregate[m]["factorized"]["exact_graph_fidelity"]>=aggregate[m]["local"]["exact_graph_fidelity"] for m in MODES),"pooled_exact_gain":overall["factorized"]["exact_graph_fidelity"]-overall["local"]["exact_graph_fidelity"],"pooled_active_non_decline":overall["factorized"]["active_pair_fidelity"]>=overall["local"]["active_pair_fidelity"]}
    _close(records,saved["records"],"records");_close(aggregate,saved["aggregate"],"aggregate");_close(overall,saved["overall"],"overall");_close(decision,saved["decision_components"],"decision")
    return {"status":"verified","worlds":len(seen),"strata":len(records),"science_not_certified":True}



