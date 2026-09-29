"""Independent replay of composed-IR research and all ablations, without Julia."""
from pathlib import Path
import numpy as np
from .io import read_json,digest
from .worlds import World,generate_worlds,load_worlds
from .cdir import Node
from .model import load_model,predict,set_seed
from .rules import Rule
from .dsl import Program
from .pysr_baseline import SymbolicScores
from .metrics import evaluate
from .distributed_alignment import measure_mapping
from .alignment import hidden
from .research_suite import ResearchConfig,training_worlds,research_report

def _features(data,nodes):
    out=[]
    for d in data:
        cache={};out.append([n.evaluate(d,cache) for n in nodes])
    return np.asarray(out)

def _read_program(d):
    classes={"Rule":Rule,"Program":Program,"SymbolicScores":SymbolicScores}
    if d["type"] not in classes:raise ValueError("Unknown program type")
    return classes[d["type"]].from_dict(d["program"])

def verify_research(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json");c=ResearchConfig(**summary["config"])
    manifest=read_json(root/"manifest.json")["artifacts"]
    for name,expected in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=expected:raise ValueError("Research artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete research")
    set_seed(c.seed);models={name:load_model(root/"models"/name/"discoverer.pt") for name in summary["teacher_hashes"]}
    for name,value in summary["teacher_hashes"].items():
        if digest(root/"models"/name/"discoverer.pt")!=value:raise ValueError("Teacher hash mismatch")
    programs={p.stem:_read_program(read_json(p)) for p in (root/"programs").glob("*.json")}
    if set(programs)!=set(summary["program_complexity"]):raise ValueError("Program catalog mismatch")
    programs={name:programs[name] for name in summary["program_complexity"]}
    catalog=read_json(root/"cdir.json");nodes=[Node.from_dict(n) for n in catalog["nodes"]]
    datasets={};training_ids=set();evaluation_ids=set();total=0
    for path in sorted((root/"datasets").iterdir()):
        ws=load_worlds(path);total+=len(ws)
        if read_json(path/"worlds.json")!=[w.metadata() for w in ws]:raise ValueError("World metadata mismatch")
        if path.name.startswith("train_"):
            expected=training_worlds(c,path.name=="train_full");training_ids|={w.identity for w in ws}
        else:
            count={"dev":c.dev_worlds,"extraction":c.extraction_worlds,"alignment_fit":c.alignment_worlds}.get(path.name,c.test_worlds)
            expected=generate_worlds(path.name,count,c.seed,c.samples)
            ids={w.identity for w in ws}
            if ids&evaluation_ids:raise ValueError("Dataset split collision")
            evaluation_ids|=ids
        if ws!=expected:raise ValueError("Worlds differ from protocol")
        with np.load(path/"samples.npz") as a:
            data=a["data"];truth=a["labels"]
            np.testing.assert_array_equal(data,np.stack([w.sample() for w in ws]))
            np.testing.assert_array_equal(a["observational"],np.stack([w.sample(False) for w in ws]))
            np.testing.assert_array_equal(a["interventions"],np.stack([w.sample(True) for w in ws]))
            np.testing.assert_array_equal(truth,[w.label for w in ws])
        if (path/"features.npz").exists():
            x=_features(data,nodes)
            with np.load(path/"features.npz") as a:np.testing.assert_allclose(x,a["features"],atol=1e-9)
            datasets[path.name]=(data,x,truth)
    if training_ids&evaluation_ids:raise ValueError("Training/evaluation leakage")
    for split,(data,x,truth) in datasets.items():
        if not split.startswith("test_"):continue
        saved=read_json(root/"evaluations"/f"{split}.json")
        for name,p in programs.items():
            teacher=models["without_environment_augmentation"] if name=="without_environment_augmentation" else models["full"]
            probs=predict(teacher,data);pred=p.predict(x)
            if pred.tolist()!=saved["predictions"][name]:raise ValueError("Rule replay failed")
            from .evidence import fidelity_bound
            if fidelity_bound((pred==probs.argmax(1)).astype(float))!=summary["fidelity_bounds"][split][name]:raise ValueError("Fidelity bound mismatch")
            metrics=evaluate(truth,probs,pred)
            if metrics!=saved["metrics"][name] or metrics!=summary["evaluation"][split][name]:
                raise ValueError("Metric replay mismatch")
    fit,fx,_=datasets["alignment_fit"];test,tx,_=datasets["alignment_test"]
    for filename,model_name,initial_name in (("guidance.json","full","full"),("guidance_without_augmentation.json","without_environment_augmentation","without_environment_augmentation")):
        g=read_json(root/filename)
        # Guide validation targets are stored; refit is not required for replay.
        cut=g["fit_rows"];hh=hidden(models[model_name],fit[cut:])
        for r in g["records"]:
            basis=np.asarray(r["basis"])
            np.testing.assert_allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-6)
            v=r["validation"];base=np.array(v["base_indices"]);source=np.array(v["source_indices"])
            from .alignment import logits_from_hidden
            from .distributed_alignment import patch_subspace
            actual=logits_from_hidden(models[model_name],patch_subspace(hh[base],hh[source],basis)).argmax(1)
            if actual.tolist()!=v["neural_after"]:raise ValueError("Circuit guidance replay failed")
    alignment=summary["alignment"]
    if alignment["status"]=="measured":
        with np.load(root/"alignment"/"mapping.npz") as a:
            feature=int(a["feature"]);hh=hidden(models["full"],test)
            result=measure_mapping(models["full"],hh,tx,programs["full"],feature,a["basis"],seed=c.seed+100)
            if result!=alignment["test"]:raise ValueError("Distributed mapping replay failed")
            if measure_mapping(models["full"],hh,tx,programs["full"],feature,a["shuffled_basis"],seed=c.seed+100)!=alignment["shuffled_target_control"]:
                raise ValueError("Shuffled-target replay failed")
            disjoint=measure_mapping(models["full"],hh,tx,programs["full"],feature,a["basis"],seed=c.seed+100,disjoint=True)
            if disjoint!=alignment["disjoint_test"]:raise ValueError("Disjoint intervention replay failed")
            from .evidence import fidelity_bound
            bound=fidelity_bound((np.array(disjoint["neural_after"])==np.array(disjoint["symbolic_target"])).astype(float))
            if bound!=alignment["disjoint_fidelity_bound"]:raise ValueError("Intervention confidence bound mismatch")
            for basis,saved in zip(a["controls"],alignment["random_controls"]):
                if measure_mapping(models["full"],hh,tx,programs["full"],feature,basis,seed=c.seed+100)!=saved:
                    raise ValueError("Random mapping replay failed")
            for split,saved in summary["environment_alignment"].items():
                d,x,_=datasets[split]
                if measure_mapping(models["full"],hidden(models["full"],d),x,programs["full"],feature,a["basis"],seed=c.seed+100)!=saved:
                    raise ValueError("OOD alignment mismatch")
    checked_queries=0
    def check_search(result,program,model):
        nonlocal checked_queries
        records=result["records"];ws=[World(**{k:v for k,v in r["world"].items() if k in World.__dataclass_fields__}) for r in records]
        data=np.stack([w.sample() for w in ws]);probs=predict(model,data);labels=program.predict(_features(data,nodes))
        if len(records)!=result["queries"] or result["queries"]!=result["query_budget"]:raise ValueError("Query budget mismatch")
        neural=probs.argmax(1)
        for w,r,pr,p,n in zip(ws,records,probs,labels,neural):
            if r["world"]!=w.metadata() or (int(p),int(n),w.label)!=(r["program"],r["neural"],r["truth"]):
                raise ValueError("Active counterexample replay failed")
            np.testing.assert_allclose(pr,r["probabilities"],atol=1e-6)
        if sum(labels!=neural)!=result["fidelity_errors"]:raise ValueError("Mismatch count")
        if sum((labels==neural)&(neural!=np.array([w.label for w in ws])))!=result["joint_errors"]:
            raise ValueError("Joint error count")
        checked_queries+=len(records)
    for path in (root/"variants").glob("*/active_cegis.json"):
        variant=path.parent.name;model=models["without_environment_augmentation"] if variant=="without_environment_augmentation" else models["full"]
        for result in read_json(path)["rounds"]:
            if result["causal_truth_used_for_optimization"]:raise ValueError("Truth leaked into CEGIS")
            check_search(result,Rule.from_dict(result["program_before"]),model)
    for path in (root/"attacks").glob("*.json"):
        result=read_json(path);check_search(result,programs["full"],models["full"])
        for replicate in result["replicates"]:
            ws=[World(**{k:v for k,v in r.items() if k in World.__dataclass_fields__}) for r in replicate["replicate_worlds"]]
            data=np.stack([w.sample() for w in ws]);p=programs["full"].predict(_features(data,nodes));n=predict(models["full"],data).argmax(1)
            if float(np.mean(p!=n))!=replicate["fidelity_error_rate"]:raise ValueError("Replication mismatch")
    with np.load(root/"pysr"/"training.npz") as a:
        np.testing.assert_allclose(programs["pysr"].scores(a["features"]),a["exported_scores"],atol=1e-7)
    if (root/"report.html").read_text(encoding="utf-8")!=research_report(summary,programs):raise ValueError("Research report mismatch")
    return {"status":"verified","worlds":total,"active_teacher_queries_replayed":checked_queries,
            "methods":list(programs),"artifacts":len(manifest),"julia_required_for_replay":False}
