"""Frozen-teacher graph-global sparse ranking experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .relational_graph import load_relational_graph
from .graph_model import pair_features,graph_probabilities,decode_graph,GRAPH_FEATURES
from .graph_experiment import symbolic_probabilities
from .factorized_graph_experiment import MODES,ENVIRONMENTS,FactorizedGraphConfig,_train_programs as train_factorized,_metrics,_pooled_metrics,_source_features,_source_integrity,_close
from .symmetric_skeleton_program import symmetric_pair_features,SymmetricSkeletonGraphProgram
from .symmetric_skeleton_experiment import SYMMETRIC_FEATURES
from .graph_global_program import GraphGlobalRankingProgram,count_features
from .rules import Rule,fit_rule

METHODS=("local","baseline","symmetric","global")

@dataclass
class GraphGlobalConfig:
    seed:int=2993
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=32
    samples:int=96
    max_splits:int=6
    beam_width:int=3
    penalty:float=.001
    edge_cs:tuple=(.01,.03,.1)
    count_alphas:tuple=(.1,1.,10.)
    @classmethod
    def quick(cls):return cls(seed=2992,nodes=(3,),environments=("test_id",),worlds_per_cell=2,samples=32,max_splits=2,edge_cs=(.03,),count_alphas=(1.,))
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid global nodes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid global environments")
        if self.worlds_per_cell<1 or self.samples<16 or self.max_splits<1 or self.beam_width<1 or self.penalty<0:raise ValueError("Invalid global budget")
        if not self.edge_cs or not self.count_alphas or any(x<=0 or not np.isfinite(x) for x in (*self.edge_cs,*self.count_alphas)):raise ValueError("Invalid global candidates")

def _teacher_parts(source,model,n,split):
    _,features=_source_features(source,n,split);teacher=np.stack([decode_graph(z)[0] for z in graph_probabilities(model,features)]);mask=np.triu(np.ones((n,n),bool),1)
    rows=[symmetric_pair_features(x) for x in features];labels=[(g|g.T)[mask].astype(int) for g in teacher];counts=np.array([y.sum() for y in labels],float)
    return features,teacher,rows,labels,counts

def _train(source,models,c):
    fc=FactorizedGraphConfig(seed=c.seed,nodes=c.nodes,environments=c.environments,worlds_per_cell=c.worlds_per_cell,samples=c.samples,max_splits=c.max_splits,beam_width=c.beam_width,penalty=c.penalty);baselines,baseline_traces=train_factorized(source,models,fc)
    globals_={};symmetric={};traces={}
    for mode in MODES:
        extraction=[_teacher_parts(source,models[mode],n,"extraction") for n in c.nodes];refinement=[_teacher_parts(source,models[mode],n,"refinement") for n in c.nodes]
        x=np.concatenate([np.concatenate(p[2]) for p in extraction]);y=np.concatenate([np.concatenate(p[3]) for p in extraction]);mean=x.mean(0);scale=x.std(0).clip(.03);z=(x-mean)/scale
        symmetric_rule,sym_trace=fit_rule(x,y,SYMMETRIC_FEATURES,max_splits=c.max_splits,beam_width=c.beam_width,penalty=c.penalty,arithmetic=True);symmetric[mode]=SymmetricSkeletonGraphProgram(symmetric_rule,baselines[mode].orientation)
        rows=[];programs=[]
        for edge_c in c.edge_cs:
            edge=LogisticRegression(l1_ratio=1.,C=edge_c,solver="liblinear",max_iter=2000,tol=1e-8,random_state=0).fit(z,y);coef=edge.coef_[0];intercept=float(edge.intercept_[0])
            targets=[]
            for _,_,chunks,_,counts in extraction:
                targets.extend(counts)
            count_x=np.stack([count_features(s,len(score)*0+next(n for n,p in zip(c.nodes,extraction) if False)) for s in []]) if False else []
            count_rows=[]
            for n,part in zip(c.nodes,extraction):
                for q in part[2]:
                    score=((q-mean)/scale)@coef+intercept;count_rows.append(count_features(score,n))
            count_x=np.stack(count_rows);targets=np.asarray(targets);cm=count_x.mean(0);cs=count_x.std(0).clip(.03);cz=(count_x-cm)/cs
            for alpha in c.count_alphas:
                reg=Ridge(alpha=alpha,solver="cholesky").fit(cz,targets)
                p=GraphGlobalRankingProgram(tuple(SYMMETRIC_FEATURES),tuple(mean),tuple(scale),tuple(coef),intercept,tuple(cm),tuple(cs),tuple(reg.coef_),float(reg.intercept_),baselines[mode].orientation,float(edge_c),float(alpha));teacher_chunks=[];pred=[]
                for features,t,_,_,_ in refinement:teacher_chunks.append(t);pred.append(np.stack([p.predict(q) for q in features]))
                metric=_pooled_metrics(teacher_chunks,pred,teacher_chunks);row={"edge_c":float(edge_c),"count_alpha":float(alpha),"edge_nonzero":p.edge_nonzero,"count_nonzero":p.count_nonzero,"exact_graph_fidelity":metric["exact_graph_fidelity"],"active_pair_fidelity":metric["active_pair_fidelity"],"all_pair_fidelity":metric["all_pair_fidelity"]};rows.append(row);programs.append(p)
        index=max(range(len(rows)),key=lambda i:(rows[i]["exact_graph_fidelity"],rows[i]["active_pair_fidelity"],-rows[i]["edge_nonzero"],-rows[i]["edge_c"],rows[i]["count_alpha"]));globals_[mode]=programs[index]
        traces[mode]={"candidates":rows,"selected":rows[index],"symmetric_trace":sym_trace,"baseline_synthesis":baseline_traces[mode],"extraction_pair_rows":len(y),"extraction_worlds":sum(len(p[0]) for p in extraction),"supervision":"frozen_teacher_decoded_graph_only"}
    return baselines,symmetric,globals_,traces

def _evaluate(root,c,models,local,baselines,symmetric,globals_,save):
    records=[];pooled={m:{k:[] for k in ("teacher",*METHODS,"truth")} for m in MODES};seen=set()
    for n in c.nodes:
      for environment in c.environments:
        worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples);ids={w.identity for w in worlds}
        if len(ids)!=len(worlds) or seen&ids:raise ValueError("Global world leakage")
        seen|=ids;path=root/"datasets"/f"n{n}_{environment}"
        if save:save_graph_dataset(path,worlds)
        elif worlds!=load_graph_worlds(path):raise ValueError("Global world mismatch")
        features=np.stack([pair_features(w.sample()) for w in worlds]);truth=np.stack([w.target_graph for w in worlds])
        if save:np.savez_compressed(path/"features.npz",features=features)
        else:
            with np.load(path/"features.npz") as z:np.testing.assert_allclose(z["features"],features,atol=1e-9)
        for mode in MODES:
          tp=graph_probabilities(models[mode],features);teacher=np.stack([decode_graph(x)[0] for x in tp]);pred={"local":np.stack([decode_graph(x)[0] for x in symbolic_probabilities(local[mode],features)]),"baseline":np.stack([baselines[mode].predict(x) for x in features]),"symmetric":np.stack([symmetric[mode].predict(x) for x in features]),"global":np.stack([globals_[mode].predict(x) for x in features])}
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
    aggregate={m:{method:_pooled_metrics(pooled[m]["teacher"],pooled[m][method],pooled[m]["truth"]) for method in METHODS} for m in MODES};overall={method:_pooled_metrics([x for m in MODES for x in pooled[m]["teacher"]],[x for m in MODES for x in pooled[m][method]],[x for m in MODES for x in pooled[m]["truth"]]) for method in METHODS};decision={"per_mode_non_decline":all(aggregate[m]["global"]["exact_graph_fidelity"]>=aggregate[m]["baseline"]["exact_graph_fidelity"] for m in MODES),"pooled_exact_gain":overall["global"]["exact_graph_fidelity"]-overall["baseline"]["exact_graph_fidelity"],"pooled_active_non_decline":overall["global"]["active_pair_fidelity"]>=overall["baseline"]["active_pair_fidelity"],"complexity_gate":all(globals_[m].edge_nonzero<=20 and globals_[m].count_nonzero<=8 for m in MODES)}
    return records,aggregate,overall,decision,len(seen)

def run_graph_global(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty graph-global directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"})
    try:
      hashes=_source_integrity(source,config);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};local={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};baselines,symmetric,globals_,traces=_train(source,models,config)
      for mode in MODES:
        path=root/"programs"/mode;path.mkdir(parents=True);save_json(path/"baseline.json",baselines[mode].to_dict());save_json(path/"symmetric.json",symmetric[mode].to_dict());save_json(path/"global.json",globals_[mode].to_dict());save_json(path/"synthesis.json",traces[mode])
      records,aggregate,overall,decision,count=_evaluate(root,config,models,local,baselines,symmetric,globals_,True);summary={"config":{**asdict(config),"source":str(source)},"source_hashes":hashes,"programs":{m:{"baseline":baselines[m].to_dict(),"symmetric":symmetric[m].to_dict(),"global":globals_[m].to_dict()} for m in MODES},"synthesis":traces,"world_count":count,"records":records,"aggregate":aggregate,"overall":overall,"decision_components":decision,"teacher_fidelity_is_primary":True,"truth_is_diagnostic":True};save_json(root/"summary.json",summary);snap=root/"source";snap.mkdir();[shutil.copy2(p,snap/p.name) for p in Path(__file__).parent.glob("*.py")];save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def verify_graph_global(directory):
    root=Path(directory).resolve();saved=read_json(root/"summary.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete graph-global run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
      p=(root/name).resolve()
      if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Graph-global artifact mismatch")
    raw=dict(saved["config"]);source=Path(raw.pop("source"));c=GraphGlobalConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"]),"edge_cs":tuple(raw["edge_cs"]),"count_alphas":tuple(raw["count_alphas"])});c.validate();_source_integrity(source,c);models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};local={m:Rule.from_dict(read_json(source/"programs"/m/"final.json")) for m in MODES};baselines,symmetric,globals_,traces=_train(source,models,c);_close(traces,saved["synthesis"],"synthesis")
    for mode in MODES:_close({"baseline":baselines[mode].to_dict(),"symmetric":symmetric[mode].to_dict(),"global":globals_[mode].to_dict()},saved["programs"][mode],"programs."+mode)
    records,aggregate,overall,decision,count=_evaluate(root,c,models,local,baselines,symmetric,globals_,False);_close(records,saved["records"],"records");_close(aggregate,saved["aggregate"],"aggregate");_close(overall,saved["overall"],"overall");_close(decision,saved["decision_components"],"decision");return {"status":"verified","worlds":count,"strata":len(records),"science_not_certified":True}


