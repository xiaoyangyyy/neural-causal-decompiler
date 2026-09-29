"""Paired inferred-graph and oracle-graph structured SCM recovery."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,shutil,tempfile
import numpy as np
from .io import save_json,read_json,digest
from .model import set_seed
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .graph_model import pair_features,graph_probabilities,decode_graph,dag_completion,load_graph_model
from .mechanisms import recover_mechanisms,structured_symbolic_fit,ExplicitSCM,evaluate_recovery
from .mechanism_search_experiment import ENVIRONMENTS,METRICS,_compact,_assert_nested_close


@dataclass
class EndToEndMechanismConfig:
    seed:int=3793
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
        return cls(seed=3792,nodes=(3,),environments=("test_id",),samples=32,
                   observation_rows=128,mechanism_epochs=12,query_count=192,
                   evaluation_rows=128,max_terms=6,beam_width=8)
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes) or len(set(self.nodes))!=len(self.nodes):
            raise ValueError("Invalid end-to-end node sizes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments) or len(set(self.environments))!=len(self.environments):
            raise ValueError("Invalid end-to-end environments")
        for name in ("seed","worlds_per_cell","samples","observation_rows","mechanism_epochs","query_count","evaluation_rows","max_terms","beam_width"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError("Invalid end-to-end budget")
        if self.samples<16 or min(self.observation_rows,self.query_count,self.evaluation_rows)<64:
            raise ValueError("Insufficient end-to-end data")


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


def _world_record(root,world,n,environment,index,inferred,decode,config,save,temp_root=None):
    cell=root/"worlds"/f"n{n}_{environment}_{index}"
    work=cell if save else Path(temp_root)/f"n{n}_{environment}_{index}"
    if save:
        cell.mkdir(parents=True);save_json(cell/"world.json",world.metadata())
        save_json(cell/"graph_inference.json",{"inferred_graph":inferred.astype(int).tolist(),"decode":decode,
            "metrics":_graph_stats(world.graph,inferred),"truth_used_for_inference":False})
    result={"path":cell.relative_to(root).as_posix(),"nodes":n,"environment":environment,
            "world_index":index,"world_id":world.identity,"graph":_graph_stats(world.graph,inferred)}
    for mode,graph in (("inferred_graph",inferred),("oracle_graph_diagnostic",np.asarray(world.graph,bool))):
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


def run_end_to_end_mechanism(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty end-to-end mechanism directory")
    teacher_path=source/"model"/"graph_teacher.pt"
    if not teacher_path.exists():raise FileNotFoundError("Frozen graph teacher missing")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"});set_seed(config.seed)
    try:
        shutil.copy2(teacher_path,root/"graph_teacher.pt")
        cfg={**asdict(config),"source":str(source),"source_teacher_sha256":digest(teacher_path),
             "graph_source":"frozen neural teacher with deterministic decoder/completion"}
        save_json(root/"config.json",cfg);teacher=load_graph_model(root/"graph_teacher.pt");records=[];seen=set()
        for n in config.nodes:
            for environment in config.environments:
                worlds=generate_graph_worlds(environment,config.worlds_per_cell,n,config.seed,config.samples)
                ids={w.identity for w in worlds}
                if len(ids)!=len(worlds) or seen&ids:raise ValueError("End-to-end world leakage")
                seen|=ids;save_graph_dataset(root/"datasets"/f"n{n}_{environment}",worlds)
                for index,world in enumerate(worlds):
                    print(f"end-to-end mechanism n={n} {environment} world={index}",flush=True)
                    features=np.array([pair_features(world.sample())]);probabilities=graph_probabilities(teacher,features)[0]
                    partial,decode=decode_graph(probabilities);inferred,choices=dag_completion(partial)
                    decode={**decode,"completion_choices":choices,"probabilities":probabilities.tolist()}
                    records.append(_world_record(root,world,n,environment,index,inferred,decode,config,True))
        aggregate={mode:{method:_aggregate(records,mode,method) for method in ("baseline","structured")}
                   for mode in ("inferred_graph","oracle_graph_diagnostic")}
        summary={"config":cfg,"world_count":len(records),"records":records,"aggregate":aggregate,
            "graph":{"exact_accuracy":float(np.mean([r["graph"]["exact"] for r in records])),
                     "mean_shd":float(np.mean([r["graph"]["shd"] for r in records]))},
            "primary_target":"end-to-end inferred-graph explicit SCM; teacher fidelity and truth reported separately"}
        save_json(root/"summary.json",summary);snap=root/"source";snap.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snap/p.name)
        save_json(root/"status.json",{"state":"completed"})
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise


def verify_end_to_end_mechanism(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json");cfg=read_json(root/"config.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete end-to-end run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("End-to-end artifact mismatch")
    raw={k:v for k,v in cfg.items() if k in EndToEndMechanismConfig.__dataclass_fields__}
    c=EndToEndMechanismConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"])});c.validate();set_seed(c.seed)
    if digest(root/"graph_teacher.pt")!=cfg["source_teacher_sha256"]:raise ValueError("End-to-end teacher mismatch")
    teacher=load_graph_model(root/"graph_teacher.pt");records=[];seen=set()
    with tempfile.TemporaryDirectory(prefix="verify_end_to_end_",dir=root) as temp:
        for n in c.nodes:
            for environment in c.environments:
                path=root/"datasets"/f"n{n}_{environment}";worlds=load_graph_worlds(path)
                if worlds!=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples):raise ValueError("End-to-end world mismatch")
                with np.load(path/"samples.npz") as a:np.testing.assert_array_equal(a["data"],np.stack([w.sample() for w in worlds]))
                for index,world in enumerate(worlds):
                    if world.identity in seen:raise ValueError("End-to-end replay leakage")
                    seen.add(world.identity);features=np.array([pair_features(world.sample())]);p=graph_probabilities(teacher,features)[0]
                    partial,decode=decode_graph(p);inferred,choices=dag_completion(partial)
                    saved=read_json(root/"worlds"/f"n{n}_{environment}_{index}"/"graph_inference.json")
                    np.testing.assert_array_equal(inferred,np.array(saved["inferred_graph"],bool))
                    full_decode={**decode,"completion_choices":choices,"probabilities":p.tolist()}
                    _assert_nested_close(full_decode,saved["decode"],f"n{n}_{environment}_{index}.decode")
                    records.append(_world_record(root,world,n,environment,index,inferred,full_decode,c,False,temp))
    _assert_nested_close(records,summary["records"],"records")
    aggregate={mode:{method:_aggregate(records,mode,method) for method in ("baseline","structured")}
               for mode in ("inferred_graph","oracle_graph_diagnostic")}
    _assert_nested_close(aggregate,summary["aggregate"],"aggregate")
    return {"status":"verified","worlds":len(records),"node_sizes":list(c.nodes),
            "environments":list(c.environments),"graph_exact_accuracy":summary["graph"]["exact_accuracy"],
            "science_not_certified":True}
