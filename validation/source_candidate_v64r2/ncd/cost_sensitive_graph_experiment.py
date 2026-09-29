"""Cost-sensitive teacher-only factorized graph-program experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .relational_graph import load_relational_graph
from .graph_model import pair_features,graph_probabilities,decode_graph,GRAPH_FEATURES
from .graph_experiment import symbolic_probabilities
from .factorized_graph_program import FactorizedGraphProgram
from .factorized_graph_experiment import MODES,ENVIRONMENTS,_metrics,_pooled_metrics,_source_features,_source_integrity,_close
from .rules import Rule,fit_rule

METHODS=("local","baseline","cost_sensitive")

@dataclass
class CostSensitiveGraphConfig:
    seed:int=2593
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=32
    samples:int=96
    max_splits:int=6
    beam_width:int=3
    penalty:float=.001
    positive_weights:tuple=(1.,1.5,2.,3.)
    @classmethod
    def quick(cls):return cls(seed=2592,nodes=(3,),environments=("test_id",),worlds_per_cell=2,samples=32,max_splits=2,positive_weights=(1.,2.))
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid cost-sensitive nodes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid cost-sensitive environments")
        if self.worlds_per_cell<1 or self.samples<16 or self.max_splits<1 or self.beam_width<1 or self.penalty<0:raise ValueError("Invalid cost-sensitive budget")
        if not self.positive_weights or tuple(sorted(set(self.positive_weights)))!=self.positive_weights or self.positive_weights[0]!=1. or any(w<1 or not np.isfinite(w) for w in self.positive_weights):raise ValueError("Invalid positive weights")

def _train_programs(source,models,c):
    extraction=[_source_features(source,n,"extraction")[1] for n in c.nodes]
    refinement=[_source_features(source,n,"refinement")[1] for n in c.nodes]
    selected={};baselines={};traces={}
    kwargs=dict(max_splits=c.max_splits,beam_width=c.beam_width,penalty=c.penalty,arithmetic=True)
    for mode in MODES:
        xs=[];ys=[]
        for features in extraction:
            labels=graph_probabilities(models[mode],features).argmax(-1);n=features.shape[1];mask=~np.eye(n,dtype=bool)
            xs.append(features[:,mask].reshape(-1,len(GRAPH_FEATURES)));ys.append(labels[:,mask].reshape(-1))
        x=np.concatenate(xs);y=np.concatenate(ys);present=y!=0
        orientation,orientation_trace=fit_rule(x[present],y[present],GRAPH_FEATURES,**kwargs)
        teacher=[np.stack([decode_graph(z)[0] for z in graph_probabilities(models[mode],features)]) for features in refinement]
        skeletons={};training={};rows=[]
        for weight in c.positive_weights:
            sample_weight=np.where(present,weight,1.)
            skeleton,trace=fit_rule(x,present.astype(int),GRAPH_FEATURES,sample_weight=sample_weight,**kwargs)
            key=str(float(weight));skeletons[key]=skeleton;training[key]=trace
            for aggregation in ("and","or"):
                program=FactorizedGraphProgram(skeleton,orientation,aggregation)
                pred=[np.stack([program.predict(z) for z in features]) for features in refinement]
                metric=_pooled_metrics(teacher,pred,teacher)
                rows.append({"positive_weight":float(weight),"aggregation":aggregation,"exact_graph_fidelity":metric["exact_graph_fidelity"],"all_pair_fidelity":metric["all_pair_fidelity"],"active_pair_fidelity":metric["active_pair_fidelity"]})
        baseline_row=max([r for r in rows if r["positive_weight"]==1.],key=lambda r:(r["exact_graph_fidelity"],r["all_pair_fidelity"],r["aggregation"]=="and"))
        selected_row=max(rows,key=lambda r:(r["exact_graph_fidelity"],r["all_pair_fidelity"],-r["positive_weight"],r["aggregation"]=="and"))
        def make(row):return FactorizedGraphProgram(skeletons[str(row["positive_weight"])],orientation,row["aggregation"])
        baselines[mode]=make(baseline_row);selected[mode]=make(selected_row)
        traces[mode]={"skeleton_training":training,"skeleton_programs":{k:v.to_dict() for k,v in skeletons.items()},"orientation":orientation_trace,"orientation_program":orientation.to_dict(),"orientation_training_rows":int(present.sum()),"candidates":rows,"baseline":baseline_row,"selected":selected_row,"supervision":"frozen_teacher_only"}
    return baselines,selected,traces

def _evaluate(root,source,c,models,local,baselines,selected,save):
    records=[];pooled={m:{k:[] for k in ("teacher",*METHODS,"truth")} for m in MODES};seen=set()
    for n in c.nodes:
      for environment in c.environments:
        worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples);ids={w.identity for w in worlds}
        if len(ids)!=len(worlds) or seen&ids:raise ValueError("Cost-sensitive world leakage")
        seen|=ids;path=root/"datasets"/f"n{n}_{environment}"
        if save:save_graph_dataset(path,worlds)
        elif worlds!=load_graph_worlds(path):raise ValueError("Cost-sensitive world mismatch")
        features=np.stack([pair_features(w.sample()) for w in worlds]);truth=np.stack([w.target_graph for w in worlds])
        if save:np.savez_compressed(path/"features.npz",features=features)
        else:
            with np.load(path/"features.npz") as z:np.testing.assert_allclose(z["features"],features,atol=1e-9)
        for mode in MODES:
          tp=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in tp]);pred={"local":np.stack([decode_graph(x)[0] for x in symbolic_probabilities(local[mode],features)]),"baseline":np.stack([baselines[mode].predict(x) for x in features]),"cost_sensitive":np.stack([selected[mode].predict(x) for x in features])}
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
    aggregate={m:{method:_pooled_metrics(pooled[m]["teacher"],pooled[m][method],pooled[m]["truth"]) for method in METHODS} for m in MODES}
    overall={method:_pooled_metrics([x for m in MODES for x in pooled[m]["teacher"]],[x for m in MODES for x in pooled[m][method]],[x for m in MODES for x in pooled[m]["truth"]]) for method in METHODS}
    decision={"per_mode_non_decline":all(aggregate[m]["cost_sensitive"]["exact_graph_fidelity"]>=aggregate[m]["baseline"]["exact_graph_fidelity"] for m in MODES),"pooled_exact_gain":overall["cost_sensitive"]["exact_graph_fidelity"]-overall["baseline"]["exact_graph_fidelity"],"pooled_active_non_decline":overall["cost_sensitive"]["active_pair_fidelity"]>=overall["baseline"]["active_pair_fidelity"]}
    return records,aggregate,overall,decision,len(seen)

def run_cost_sensitive_graph(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty cost-sensitive directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    try:
      hashes=_source_integrity(source,config);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};local={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};baselines,selected,traces=_train_programs(source,models,config)
      for mode in MODES:
        path=root/"programs"/mode;path.mkdir(parents=True);save_json(path/"baseline.json",baselines[mode].to_dict());save_json(path/"cost_sensitive.json",selected[mode].to_dict());save_json(path/"synthesis.json",traces[mode])
      records,aggregate,overall,decision,count=_evaluate(root,source,config,models,local,baselines,selected,True)
      summary={"config":{**asdict(config),"source":str(source)},"source_hashes":hashes,"programs":{m:{"baseline":baselines[m].to_dict(),"cost_sensitive":selected[m].to_dict()} for m in MODES},"synthesis":traces,"world_count":count,"records":records,"aggregate":aggregate,"overall":overall,"decision_components":decision,"teacher_fidelity_is_primary":True,"truth_is_diagnostic":True};save_json(root/"summary.json",summary)
      snap=root/"source";snap.mkdir();[shutil.copy2(p,snap/p.name) for p in Path(__file__).parent.glob("*.py")]
      save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def verify_cost_sensitive_graph(directory):
    root=Path(directory).resolve();saved=read_json(root/"summary.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete cost-sensitive run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
      p=(root/name).resolve()
      if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Cost-sensitive artifact mismatch")
    raw=dict(saved["config"]);source=Path(raw.pop("source"));c=CostSensitiveGraphConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"]),"positive_weights":tuple(raw["positive_weights"])});c.validate();_source_integrity(source,c)
    models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};local={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};baselines,selected,traces=_train_programs(source,models,c)
    _close(traces,saved["synthesis"],"synthesis")
    for mode in MODES:_close({"baseline":baselines[mode].to_dict(),"cost_sensitive":selected[mode].to_dict()},saved["programs"][mode],"programs."+mode)
    records,aggregate,overall,decision,count=_evaluate(root,source,c,models,local,baselines,selected,False)
    _close(records,saved["records"],"records");_close(aggregate,saved["aggregate"],"aggregate");_close(overall,saved["overall"],"overall");_close(decision,saved["decision_components"],"decision")
    return {"status":"verified","worlds":count,"strata":len(records),"science_not_certified":True}
