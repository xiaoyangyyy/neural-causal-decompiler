"""Paired evaluation of greedy and hierarchy-constrained mechanism extraction."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,shutil
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .mechanisms import (recover_mechanisms,structured_symbolic_fit,sparse_symbolic_fit,
    ExplicitSCM,evaluate_recovery,load_mechanism,neural_values)

ENVIRONMENTS=("test_id","test_function","test_noise","test_scale","test_intervention")
METRICS=("parent_exact_fraction","operator_exact_fraction","interaction_exact_fraction",
         "mean_symbolic_neural_nmse","mean_symbolic_truth_nmse","mean_intervention_effect_mae",
         "mean_nonconstant_atoms")

@dataclass
class MechanismSearchConfig:
    seed:int=1793
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=2
    samples:int=96
    observation_rows:int=512
    mechanism_epochs:int=120
    query_count:int=768
    evaluation_rows:int=512
    max_terms:int=8
    beam_width:int=32

    @classmethod
    def quick(cls):
        return cls(seed=1792,nodes=(3,),environments=("test_id",),worlds_per_cell=1,
                   samples=32,observation_rows=128,mechanism_epochs=12,query_count=192,
                   evaluation_rows=128,max_terms=6,beam_width=8)

    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid node sizes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid environments")
        if len(set(self.nodes))!=len(self.nodes) or len(set(self.environments))!=len(self.environments):raise ValueError("Duplicate cells")
        for name in ("seed","worlds_per_cell","samples","observation_rows","mechanism_epochs","query_count","evaluation_rows","max_terms","beam_width"):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:raise ValueError(f"Invalid {name}")
        if self.observation_rows<64 or self.query_count<64 or self.evaluation_rows<64:raise ValueError("Insufficient mechanism data")

def _compact(result,records):
    nodes=result["nodes"]
    return {
        "parent_exact_fraction":float(np.mean([x["parent_exact"] for x in nodes])),
        "operator_exact_fraction":float(np.mean([x["operator_exact"] for x in nodes])),
        "interaction_exact_fraction":float(np.mean([x["interaction_exact"] for x in nodes])),
        "mean_symbolic_neural_nmse":result["mean_symbolic_neural_nmse"],
        "mean_symbolic_truth_nmse":result["mean_symbolic_truth_nmse"],
        "mean_intervention_effect_mae":result["mean_intervention_effect_mae"],
        "mean_nonconstant_atoms":float(np.mean([sum(a["atom"]!="constant" for a in r["atoms"]) for r in records]))
    }

def _aggregate(records,method):
    return {key:float(np.mean([r[method][key] for r in records])) for key in METRICS}

def _assert_nested_close(actual,expected,path="root"):
    if isinstance(actual,dict) and isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise ValueError(f"Replay keys differ at {path}")
        for key in actual:_assert_nested_close(actual[key],expected[key],f"{path}.{key}")
    elif isinstance(actual,list) and isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError(f"Replay length differs at {path}")
        for index,(left,right) in enumerate(zip(actual,expected)):_assert_nested_close(left,right,f"{path}[{index}]")
    elif isinstance(actual,float) or isinstance(expected,float):
        if not np.isclose(actual,expected,rtol=1e-7,atol=1e-9):raise ValueError(f"Replay float differs at {path}")
    elif actual!=expected:raise ValueError(f"Replay value differs at {path}")


def run_mechanism_search(directory,config):
    config.validate();root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty mechanism-search directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"config.json",asdict(config));save_json(root/"status.json",{"state":"running"})
    try:return _execute(root,config)
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def _execute(root,c):
    records=[];seen=set()
    for n in c.nodes:
        for environment in c.environments:
            worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples)
            if seen&{w.identity for w in worlds}:raise ValueError("World leakage")
            seen|={w.identity for w in worlds}
            dataset=root/"datasets"/f"n{n}_{environment}";save_graph_dataset(dataset,worlds)
            for index,world in enumerate(worlds):
                print(f"mechanism search n={n} {environment} world={index}",flush=True)
                cell=root/"worlds"/f"n{n}_{environment}_{index}";baseline_path=cell/"baseline"
                observations=world.sample(seed=world.seed^1234567,samples=c.observation_rows)
                baseline,models,base_records=recover_mechanisms(observations,np.array(world.graph,bool),baseline_path,
                    seed=c.seed,epochs=c.mechanism_epochs,query_count=c.query_count)
                expressions=[];structured_records=[]
                for j,model in enumerate(models):
                    with np.load(baseline_path/f"distillation_{j}.npz") as a:
                        expression,atoms,history=structured_symbolic_fit(a["queries"],a["teacher"],model.parents,
                            max_terms=c.max_terms,beam_width=c.beam_width)
                    expressions.append(expression);structured_records.append({**base_records[j],"atoms":atoms,"search_history":history,
                        "search":"hierarchy_constrained_four_fold_beam"})
                structured=ExplicitSCM(expressions,baseline.noise_samples,baseline.source_graph)
                base_eval=evaluate_recovery(world,baseline,models,base_records,seed=c.seed+1234,samples=c.evaluation_rows)
                structured_eval=evaluate_recovery(world,structured,models,structured_records,seed=c.seed+1234,samples=c.evaluation_rows)
                structured_path=cell/"structured";structured_path.mkdir(parents=True)
                save_json(structured_path/"explicit_scm.json",structured.to_dict());save_json(structured_path/"recovery.json",structured_records)
                save_json(cell/"world.json",world.metadata());save_json(cell/"baseline_evaluation.json",base_eval)
                save_json(cell/"structured_evaluation.json",structured_eval)
                records.append({"path":cell.relative_to(root).as_posix(),"nodes":n,"environment":environment,"world_index":index,
                    "world_id":world.identity,"baseline":_compact(base_eval,base_records),"structured":_compact(structured_eval,structured_records)})
    summary={"config":asdict(c),"world_count":len(records),"records":records,
             "aggregate":{"baseline":_aggregate(records,"baseline"),"structured":_aggregate(records,"structured")},
             "primary_target":"frozen neural mechanism predictions","truth_is_diagnostic":True,
             "graph_scope":"oracle DAG supplied to isolate equation extraction"}
    save_json(root/"summary.json",summary)
    source=root/"source";source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,source/p.name)
    save_json(root/"status.json",{"state":"completed"})
    save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
    print(f"mechanism search complete: {root}",flush=True);return summary

def verify_mechanism_search(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Mechanism-search artifact mismatch")
    raw=summary["config"];c=MechanismSearchConfig(**{**raw,"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"])});c.validate()
    expected={(n,e) for n in c.nodes for e in c.environments};seen=set();records=[]
    for n,e in [(n,e) for n in c.nodes for e in c.environments]:
        dataset=root/"datasets"/f"n{n}_{e}";worlds=load_graph_worlds(dataset)
        if worlds!=generate_graph_worlds(e,c.worlds_per_cell,n,c.seed,c.samples):raise ValueError("Mechanism world mismatch")
        with np.load(dataset/"samples.npz") as a:
            np.testing.assert_array_equal(a["data"],np.stack([w.sample() for w in worlds]))
        for index,world in enumerate(worlds):
            if world.identity in seen:raise ValueError("World leakage")
            seen.add(world.identity);cell=root/"worlds"/f"n{n}_{e}_{index}"
            if read_json(cell/"world.json")["world_id"]!=world.identity:raise ValueError("World provenance mismatch")
            baseline_path=cell/"baseline";baseline=ExplicitSCM.from_dict(read_json(baseline_path/"explicit_scm.json"))
            structured=ExplicitSCM.from_dict(read_json(cell/"structured"/"explicit_scm.json"))
            base_records=read_json(baseline_path/"recovery.json");structured_records=read_json(cell/"structured"/"recovery.json")
            models=[load_mechanism(baseline_path/f"mechanism_{j}.pt") for j in range(n)]
            observations=world.sample(seed=world.seed^1234567,samples=c.observation_rows)
            with np.load(baseline_path/"observations.npz") as a:np.testing.assert_array_equal(a["data"],observations)
            for j,model in enumerate(models):
                with np.load(baseline_path/f"distillation_{j}.npz") as a:
                    np.testing.assert_allclose(neural_values(model,a["queries"]),a["teacher"],rtol=1e-6,atol=1e-6)
                    old,_,_=sparse_symbolic_fit(a["queries"],a["teacher"],model.parents)
                    new,_,_=structured_symbolic_fit(a["queries"],a["teacher"],model.parents,max_terms=c.max_terms,beam_width=c.beam_width)
                    np.testing.assert_allclose(old.evaluate(a["queries"]),baseline.equations[j].evaluate(a["queries"]),atol=1e-7)
                    np.testing.assert_allclose(new.evaluate(a["queries"]),structured.equations[j].evaluate(a["queries"]),atol=1e-7)
            base_eval=evaluate_recovery(world,baseline,models,base_records,seed=c.seed+1234,samples=c.evaluation_rows)
            new_eval=evaluate_recovery(world,structured,models,structured_records,seed=c.seed+1234,samples=c.evaluation_rows)
            _assert_nested_close(base_eval,read_json(cell/"baseline_evaluation.json"),f"{cell.name}.baseline")
            _assert_nested_close(new_eval,read_json(cell/"structured_evaluation.json"),f"{cell.name}.structured")
            records.append({"path":cell.relative_to(root).as_posix(),"nodes":n,"environment":e,"world_index":index,
                "world_id":world.identity,"baseline":_compact(base_eval,base_records),"structured":_compact(new_eval,structured_records)})
    rebuilt={"baseline":_aggregate(records,"baseline"),"structured":_aggregate(records,"structured")}
    _assert_nested_close(records,summary["records"],"summary.records")
    _assert_nested_close(rebuilt,summary["aggregate"],"summary.aggregate")
    return {"status":"verified","worlds":len(seen),"node_sizes":list(c.nodes),"environments":list(c.environments),
            "primary_nmse":rebuilt["structured"]["mean_symbolic_neural_nmse"],"science_not_certified":True}