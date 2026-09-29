"""Observational, active-intervention, and oracle graph SCM recovery."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,shutil,tempfile
import numpy as np
from .io import save_json,read_json,digest
from .model import set_seed
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .graph_model import pair_features,graph_probabilities,decode_graph,dag_completion
from .active_intervention_graph import padded_observational_features,intervention_response_features,load_active_factorized_graph
from .mechanisms import recover_mechanisms,structured_symbolic_fit,ExplicitSCM,evaluate_recovery
from .mechanism_search_experiment import ENVIRONMENTS,METRICS,_compact,_assert_nested_close


@dataclass
class ActiveEndToEndConfig:
    seed:int=4993
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=1
    samples:int=96
    observation_rows:int=512
    mechanism_epochs:int=120
    query_count:int=768
    evaluation_rows:int=512
    max_terms:int=8
    beam_width:int=32
    @classmethod
    def quick(cls):
        return cls(seed=4992,nodes=(3,),environments=("test_id",),samples=32,
                   observation_rows=128,mechanism_epochs=12,query_count=192,
                   evaluation_rows=128,max_terms=6,beam_width=8)
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes) or len(set(self.nodes))!=len(self.nodes):
            raise ValueError("Invalid active end-to-end node sizes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments) or len(set(self.environments))!=len(self.environments):
            raise ValueError("Invalid active end-to-end environments")
        for name in ("seed","worlds_per_cell","samples","observation_rows","mechanism_epochs","query_count","evaluation_rows","max_terms","beam_width"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError("Invalid active end-to-end budget")
        if self.samples<16 or min(self.observation_rows,self.query_count,self.evaluation_rows)<64:
            raise ValueError("Insufficient active end-to-end data")


def _aggregate(records,mode,method):
    selected=[r[mode][method] for r in records]
    return {key:float(np.mean([x[key] for x in selected])) for key in METRICS}


def _graph_stats(truth,inferred):
    truth=np.asarray(truth,bool);inferred=np.asarray(inferred,bool)
    return {"exact":bool(np.array_equal(truth,inferred)),"shd":int(np.sum(truth!=inferred)),
            "true_edges":int(truth.sum()),"inferred_edges":int(inferred.sum())}


def _fit_mode(world,graph,path,c,save):
    observations=world.sample(seed=world.seed^1234567,samples=c.observation_rows)
    baseline,models,base_records=recover_mechanisms(observations,graph,path/"baseline",
        seed=c.seed,epochs=c.mechanism_epochs,query_count=c.query_count)
    expressions=[];structured_records=[]
    for j,model in enumerate(models):
        with np.load(path/"baseline"/f"distillation_{j}.npz") as a:
            expression,atoms,history=structured_symbolic_fit(a["queries"],a["teacher"],model.parents,
                max_terms=c.max_terms,beam_width=c.beam_width)
        expressions.append(expression);structured_records.append({**base_records[j],"atoms":atoms,
            "search_history":history,"search":"hierarchy_constrained_four_fold_beam"})
    structured=ExplicitSCM(expressions,baseline.noise_samples,baseline.source_graph)
    baseline_eval=evaluate_recovery(world,baseline,models,base_records,seed=c.seed+1234,samples=c.evaluation_rows)
    structured_eval=evaluate_recovery(world,structured,models,structured_records,seed=c.seed+1234,samples=c.evaluation_rows)
    if save:
        target=path/"structured";target.mkdir(parents=True)
        save_json(target/"explicit_scm.json",structured.to_dict());save_json(target/"recovery.json",structured_records)
        save_json(path/"baseline_evaluation.json",baseline_eval);save_json(path/"structured_evaluation.json",structured_eval)
    return {"baseline":_compact(baseline_eval,base_records),"structured":_compact(structured_eval,structured_records),
            "baseline_scm":baseline.to_dict(),"structured_scm":structured.to_dict(),
            "baseline_records":base_records,"structured_records":structured_records}


def _world_record(root,world,n,environment,index,graphs,decodes,config,save,temp_root=None):
    cell=root/"worlds"/f"n{n}_{environment}_{index}"
    work=cell if save else Path(temp_root)/f"n{n}_{environment}_{index}"
    if save:
        cell.mkdir(parents=True);save_json(cell/"world.json",world.metadata())
        save_json(cell/"graph_inference.json",{
            mode:{"inferred_graph":graphs[mode].astype(int).tolist(),"decode":decodes[mode],
                  "metrics":_graph_stats(world.graph,graphs[mode]),"truth_used_for_inference":False}
            for mode in ("observational_graph","active_graph")})
    result={"path":cell.relative_to(root).as_posix(),"nodes":n,"environment":environment,
            "world_index":index,"world_id":world.identity,
            "graph":{mode:_graph_stats(world.graph,graphs[mode]) for mode in ("observational_graph","active_graph")}}
    modes=(("observational_graph",graphs["observational_graph"]),("active_graph",graphs["active_graph"]),
           ("oracle_graph_diagnostic",np.asarray(world.graph,bool)))
    for mode,graph in modes:
        fitted=_fit_mode(world,graph,work/mode,config,save)
        result[mode]={"baseline":fitted["baseline"],"structured":fitted["structured"]}
        if not save:
            stored=read_json(cell/mode/"baseline"/"explicit_scm.json")
            _assert_nested_close(fitted["baseline_scm"],stored,f"{cell.name}.{mode}.baseline_scm")
            stored=read_json(cell/mode/"structured"/"explicit_scm.json")
            _assert_nested_close(fitted["structured_scm"],stored,f"{cell.name}.{mode}.structured_scm")
            _assert_nested_close(json.loads(json.dumps(fitted["baseline_records"])),read_json(cell/mode/"baseline"/"recovery.json"),f"{cell.name}.{mode}.baseline_records")
            _assert_nested_close(json.loads(json.dumps(fitted["structured_records"])),read_json(cell/mode/"structured"/"recovery.json"),f"{cell.name}.{mode}.structured_records")
            _assert_nested_close(fitted["baseline"],_compact(read_json(cell/mode/"baseline_evaluation.json"),fitted["baseline_records"]),f"{cell.name}.{mode}.baseline")
            _assert_nested_close(fitted["structured"],_compact(read_json(cell/mode/"structured_evaluation.json"),fitted["structured_records"]),f"{cell.name}.{mode}.structured")
    return result

def run_active_end_to_end(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty active end-to-end directory")
    teacher_paths={mode:source/"models"/source_mode/"graph_teacher.pt" for mode,source_mode in
                   (("observational_graph","observational_padded"),("active_graph","active_intervention"))}
    if any(not p.exists() for p in teacher_paths.values()):raise FileNotFoundError("Frozen paired graph teachers missing")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"});set_seed(config.seed)
    try:
        for mode,path in teacher_paths.items():shutil.copy2(path,root/f"{mode}_teacher.pt")
        cfg={**asdict(config),"source":str(source),
             "source_teacher_sha256":{m:digest(p) for m,p in teacher_paths.items()},
             "graph_source":"paired frozen observational-padded and active-intervention teachers"}
        save_json(root/"config.json",cfg)
        teachers={m:load_active_factorized_graph(root/f"{m}_teacher.pt") for m in teacher_paths}
        records=[];seen=set()
        for n in config.nodes:
            for environment in config.environments:
                worlds=generate_graph_worlds(environment,config.worlds_per_cell,n,config.seed,config.samples)
                ids={w.identity for w in worlds}
                if len(ids)!=len(worlds) or seen&ids:raise ValueError("Active end-to-end world leakage")
                seen|=ids;save_graph_dataset(root/"datasets"/f"n{n}_{environment}",worlds)
                for index,world in enumerate(worlds):
                    print(f"active end-to-end n={n} {environment} world={index}",flush=True)
                    data=world.sample();base=pair_features(data)
                    features={"observational_graph":padded_observational_features(data,base),
                              "active_graph":intervention_response_features(world,data,base)}
                    graphs={};decodes={}
                    for mode,teacher in teachers.items():
                        probabilities=graph_probabilities(teacher,features[mode][None])[0]
                        partial,decode=decode_graph(probabilities);inferred,choices=dag_completion(partial)
                        graphs[mode]=inferred;decodes[mode]={**decode,"completion_choices":choices,"probabilities":probabilities.tolist()}
                    records.append(_world_record(root,world,n,environment,index,graphs,decodes,config,True))
        modes=("observational_graph","active_graph","oracle_graph_diagnostic")
        aggregate={mode:{method:_aggregate(records,mode,method) for method in ("baseline","structured")} for mode in modes}
        graph={mode:{"exact_accuracy":float(np.mean([r["graph"][mode]["exact"] for r in records])),
                     "mean_shd":float(np.mean([r["graph"][mode]["shd"] for r in records]))}
               for mode in ("observational_graph","active_graph")}
        summary={"config":cfg,"world_count":len(records),"records":records,"aggregate":aggregate,"graph":graph,
            "primary_target":"active-intervention inferred-graph explicit SCM; observational and oracle graph branches are paired diagnostics"}
        save_json(root/"summary.json",summary);snap=root/"source";snap.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snap/p.name)
        save_json(root/"status.json",{"state":"completed"})
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def verify_active_end_to_end(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json");cfg=read_json(root/"config.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete active end-to-end run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Active end-to-end artifact mismatch")
    raw={k:v for k,v in cfg.items() if k in ActiveEndToEndConfig.__dataclass_fields__}
    c=ActiveEndToEndConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"])});c.validate();set_seed(c.seed)
    for mode,expected in cfg["source_teacher_sha256"].items():
        if digest(root/f"{mode}_teacher.pt")!=expected:raise ValueError("Active end-to-end teacher mismatch")
    teachers={m:load_active_factorized_graph(root/f"{m}_teacher.pt") for m in ("observational_graph","active_graph")}
    records=[];seen=set()
    with tempfile.TemporaryDirectory(prefix="verify_active_end_to_end_",dir=root) as temp:
        for n in c.nodes:
            for environment in c.environments:
                path=root/"datasets"/f"n{n}_{environment}";worlds=load_graph_worlds(path)
                if worlds!=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples):raise ValueError("Active end-to-end world mismatch")
                with np.load(path/"samples.npz") as a:np.testing.assert_array_equal(a["data"],np.stack([w.sample() for w in worlds]))
                for index,world in enumerate(worlds):
                    if world.identity in seen:raise ValueError("Active end-to-end replay leakage")
                    seen.add(world.identity);data=world.sample();base=pair_features(data)
                    features={"observational_graph":padded_observational_features(data,base),
                              "active_graph":intervention_response_features(world,data,base)}
                    saved=read_json(root/"worlds"/f"n{n}_{environment}_{index}"/"graph_inference.json")
                    graphs={};decodes={}
                    for mode,teacher in teachers.items():
                        probabilities=graph_probabilities(teacher,features[mode][None])[0]
                        partial,decode=decode_graph(probabilities);inferred,choices=dag_completion(partial)
                        graphs[mode]=inferred;decodes[mode]={**decode,"completion_choices":choices,"probabilities":probabilities.tolist()}
                        np.testing.assert_array_equal(inferred,np.array(saved[mode]["inferred_graph"],bool))
                        _assert_nested_close(decodes[mode],saved[mode]["decode"],f"n{n}_{environment}_{index}.{mode}.decode")
                    records.append(_world_record(root,world,n,environment,index,graphs,decodes,c,False,temp))
    _assert_nested_close(records,summary["records"],"records")
    modes=("observational_graph","active_graph","oracle_graph_diagnostic")
    aggregate={mode:{method:_aggregate(records,mode,method) for method in ("baseline","structured")} for mode in modes}
    _assert_nested_close(aggregate,summary["aggregate"],"aggregate")
    graph={mode:{"exact_accuracy":float(np.mean([r["graph"][mode]["exact"] for r in records])),
                 "mean_shd":float(np.mean([r["graph"][mode]["shd"] for r in records]))}
           for mode in ("observational_graph","active_graph")}
    _assert_nested_close(graph,summary["graph"],"graph")
    return {"status":"verified","worlds":len(records),"node_sizes":list(c.nodes),
            "environments":list(c.environments),"graph":graph,"science_not_certified":True}
