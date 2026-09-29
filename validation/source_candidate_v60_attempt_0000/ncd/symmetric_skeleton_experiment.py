"""Teacher-only symmetric decoded-skeleton graph-program experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .relational_graph import load_relational_graph
from .graph_model import pair_features,graph_probabilities,decode_graph,GRAPH_FEATURES
from .graph_experiment import symbolic_probabilities
from .factorized_graph_experiment import MODES,ENVIRONMENTS,FactorizedGraphConfig,_train_programs as train_factorized,_metrics,_pooled_metrics,_source_features,_source_integrity,_close
from .symmetric_skeleton_program import symmetric_pair_features,SymmetricSkeletonGraphProgram
from .rules import Rule,fit_rule

METHODS=("local","baseline","symmetric")
SYMMETRIC_FEATURES=tuple("min_"+x for x in GRAPH_FEATURES)+tuple("max_"+x for x in GRAPH_FEATURES)

@dataclass
class SymmetricSkeletonConfig:
    seed:int=2793
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=32
    samples:int=96
    max_splits:int=6
    beam_width:int=3
    penalty:float=.001
    @classmethod
    def quick(cls):return cls(seed=2792,nodes=(3,),environments=("test_id",),worlds_per_cell=2,samples=32,max_splits=2)
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid symmetric nodes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid symmetric environments")
        if self.worlds_per_cell<1 or self.samples<16 or self.max_splits<1 or self.beam_width<1 or self.penalty<0:raise ValueError("Invalid symmetric budget")

def _train(source,models,c):
    fc=FactorizedGraphConfig(seed=c.seed,nodes=c.nodes,environments=c.environments,worlds_per_cell=c.worlds_per_cell,samples=c.samples,max_splits=c.max_splits,beam_width=c.beam_width,penalty=c.penalty)
    baselines,baseline_traces=train_factorized(source,models,fc);programs={};traces={}
    for mode in MODES:
        xs=[];ys=[]
        for n in c.nodes:
            _,features=_source_features(source,n,"extraction");teacher=np.stack([decode_graph(z)[0] for z in graph_probabilities(models[mode],features)]);mask=np.triu(np.ones((n,n),bool),1)
            xs.append(np.concatenate([symmetric_pair_features(x) for x in features]));ys.append(np.concatenate([(g|g.T)[mask].astype(int) for g in teacher]))
        x=np.concatenate(xs);y=np.concatenate(ys);skeleton,trace=fit_rule(x,y,SYMMETRIC_FEATURES,max_splits=c.max_splits,beam_width=c.beam_width,penalty=c.penalty,arithmetic=True)
        programs[mode]=SymmetricSkeletonGraphProgram(skeleton,baselines[mode].orientation)
        traces[mode]={"skeleton":trace,"skeleton_training_rows":len(y),"positive_fraction":float(np.mean(y)),"target":"frozen_teacher_decoded_skeleton","baseline_synthesis":baseline_traces[mode],"supervision":"frozen_teacher_only"}
    return baselines,programs,traces

def _evaluate(root,c,models,local,baselines,programs,save):
    records=[];pooled={m:{k:[] for k in ("teacher",*METHODS,"truth")} for m in MODES};seen=set()
    for n in c.nodes:
      for environment in c.environments:
        worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples);ids={w.identity for w in worlds}
        if len(ids)!=len(worlds) or seen&ids:raise ValueError("Symmetric world leakage")
        seen|=ids;path=root/"datasets"/f"n{n}_{environment}"
        if save:save_graph_dataset(path,worlds)
        elif worlds!=load_graph_worlds(path):raise ValueError("Symmetric world mismatch")
        features=np.stack([pair_features(w.sample()) for w in worlds]);truth=np.stack([w.target_graph for w in worlds])
        if save:np.savez_compressed(path/"features.npz",features=features)
        else:
            with np.load(path/"features.npz") as z:np.testing.assert_allclose(z["features"],features,atol=1e-9)
        for mode in MODES:
          tp=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in tp]);pred={"local":np.stack([decode_graph(x)[0] for x in symbolic_probabilities(local[mode],features)]),"baseline":np.stack([baselines[mode].predict(x) for x in features]),"symmetric":np.stack([programs[mode].predict(x) for x in features])}
          if save:np.savez_compressed(path/f"{mode}.npz",teacher_probabilities=tp,teacher=teacher,**pred)
          else:
            with np.load(path/f"{mode}.npz") as z:
              np.testing.assert_allclose(z["teacher_probabilities"],tp,rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(z["teacher"],teacher)
              for method in METHODS:np.testing.assert_array_equal(z[method],pred[method])
          pooled[mode]["teacher"].append(teacher);pooled[mode]["truth"].append(truth)
          for method in METHODS:pooled[mode][method].append(pred[method])
          family={}
          for f in sorted({w.family for w in worlds}):
            choose=np.array([w.family==f for w in worlds]);family[f]={method:_metrics(teacher[choose],pred[method][choose],truth[choose]) for method in METHODS}
          records.append({"nodes":n,"environment":environment,"mode":mode,**{method:_metrics(teacher,pred[method],truth) for method in METHODS},"family":family})
    aggregate={m:{method:_pooled_metrics(pooled[m]["teacher"],pooled[m][method],pooled[m]["truth"]) for method in METHODS} for m in MODES};overall={method:_pooled_metrics([x for m in MODES for x in pooled[m]["teacher"]],[x for m in MODES for x in pooled[m][method]],[x for m in MODES for x in pooled[m]["truth"]]) for method in METHODS}
    decision={"per_mode_non_decline":all(aggregate[m]["symmetric"]["exact_graph_fidelity"]>=aggregate[m]["baseline"]["exact_graph_fidelity"] for m in MODES),"pooled_exact_gain":overall["symmetric"]["exact_graph_fidelity"]-overall["baseline"]["exact_graph_fidelity"],"pooled_active_non_decline":overall["symmetric"]["active_pair_fidelity"]>=overall["baseline"]["active_pair_fidelity"]}
    return records,aggregate,overall,decision,len(seen)

def run_symmetric_skeleton(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty symmetric-skeleton directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    try:
      hashes=_source_integrity(source,config);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};local={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};baselines,programs,traces=_train(source,models,config)
      for mode in MODES:
        path=root/"programs"/mode;path.mkdir(parents=True);save_json(path/"baseline.json",baselines[mode].to_dict());save_json(path/"symmetric.json",programs[mode].to_dict());save_json(path/"synthesis.json",traces[mode])
      records,aggregate,overall,decision,count=_evaluate(root,config,models,local,baselines,programs,True);summary={"config":{**asdict(config),"source":str(source)},"source_hashes":hashes,"programs":{m:{"baseline":baselines[m].to_dict(),"symmetric":programs[m].to_dict()} for m in MODES},"synthesis":traces,"world_count":count,"records":records,"aggregate":aggregate,"overall":overall,"decision_components":decision,"teacher_fidelity_is_primary":True,"truth_is_diagnostic":True};save_json(root/"summary.json",summary)
      snap=root/"source";snap.mkdir();[shutil.copy2(p,snap/p.name) for p in Path(__file__).parent.glob("*.py")];save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def verify_symmetric_skeleton(directory):
    root=Path(directory).resolve();saved=read_json(root/"summary.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete symmetric run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
      p=(root/name).resolve()
      if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Symmetric artifact mismatch")
    raw=dict(saved["config"]);source=Path(raw.pop("source"));c=SymmetricSkeletonConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"])});c.validate();_source_integrity(source,c);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};local={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};baselines,programs,traces=_train(source,models,c);_close(traces,saved["synthesis"],"synthesis")
    for mode in MODES:_close({"baseline":baselines[mode].to_dict(),"symmetric":programs[mode].to_dict()},saved["programs"][mode],"programs."+mode)
    records,aggregate,overall,decision,count=_evaluate(root,c,models,local,baselines,programs,False);_close(records,saved["records"],"records");_close(aggregate,saved["aggregate"],"aggregate");_close(overall,saved["overall"],"overall");_close(decision,saved["decision_components"],"decision");return {"status":"verified","worlds":count,"strata":len(records),"science_not_certified":True}
