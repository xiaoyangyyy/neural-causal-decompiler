"""Independent artifact replay, beyond trusting a saved green status."""
from pathlib import Path
import numpy as np
from .io import read_json,digest
from .worlds import load_worlds,generate_worlds
from .model import load_model,predict,set_seed
from .statistics import extract
from .dsl import Program
from .metrics import evaluate
from .baselines import anm_predict
from .alignment import hidden,interchange,logits_from_hidden,fit_and_test
from .counterexamples import compare_search
from .report import render

def verify(directory):
    root = Path(directory).resolve()
    manifest = read_json(root/"manifest.json")
    for name,expected in manifest["artifacts"].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root): raise ValueError("Artifact path escapes run")
        if not p.is_file() or digest(p)!=expected: raise ValueError(f"Artifact integrity mismatch: {name}")
    if read_json(root/"status.json")["state"] != "completed": raise ValueError("Run not complete")
    summary = read_json(root/"summary.json")
    set_seed(summary["config"]["seed"])
    model = load_model(root/"model"/"discoverer.pt")
    if digest(root/"model"/"discoverer.pt")!=summary["teacher_sha256"]: raise ValueError("Teacher identity mismatch")
    programs = {}
    for name in summary["program_complexity"]:
        a = read_json(root/"programs"/f"{name}.json")
        p = Program.from_dict(a["program"])
        if p.complexity!=a["complexity"] or p.complexity!=summary["program_complexity"][name]: raise ValueError("Complexity mismatch")
        if name!="truth_supervised_same_dsl" and a["teacher_sha256"]!=summary["teacher_sha256"]: raise ValueError("Program teacher mismatch")
        programs[name]=p
    seen,counts,loaded = set(),{},{}
    for directory in sorted((root/"datasets").iterdir()):
        ws = load_worlds(directory)
        metadata = read_json(directory/"worlds.json")
        if metadata != [w.metadata() for w in ws]: raise ValueError("World metadata / AST mismatch")
        cfg = summary["config"]
        key = directory.name
        expected_count = cfg[{"train":"train_worlds","dev":"dev_worlds","extraction":"extraction_worlds",
                              "refinement":"refinement_worlds","attack":"refinement_worlds",
                              "alignment_fit":"alignment_worlds"}.get(key,"test_worlds")]
        expected_worlds = generate_worlds("refinement" if key=="attack" else key, expected_count,
                                         cfg["seed"]+(1000003 if key=="attack" else 0),cfg["samples"])
        if ws!=expected_worlds: raise ValueError("Dataset differs from frozen configuration")
        ids = [w.identity for w in ws]
        if len(set(ids))!=len(ids) or seen.intersection(ids): raise ValueError("World split overlap")
        seen.update(ids)
        counts[directory.name]=len(ws)
        with np.load(directory/"samples.npz") as a:
            data,labels,obs,interventions=a["data"],a["labels"],a["observational"],a["interventions"]
        if not np.array_equal(labels,[w.label for w in ws]): raise ValueError("Ground-truth labels mismatch")
        for i,w in enumerate(ws):
            if not np.array_equal(data[i],w.sample()): raise ValueError("SCM replay failed")
            if not np.array_equal(obs[i],w.sample(False)) or not np.array_equal(interventions[i],w.sample(True)):
                raise ValueError("Intervention replay failed")
        loaded[directory.name]=(data,labels)
        derived = directory/"derived.npz"
        if derived.exists():
            with np.load(derived) as a:
                features=a["features"]
                if not np.allclose(extract(data),features,rtol=1e-8,atol=1e-9): raise ValueError("Primitive replay failed")
                if "probabilities" in a and not np.allclose(predict(model,data),a["probabilities"],atol=1e-6):
                    raise ValueError("Neural replay failed")
        if directory.name.startswith("test_"):
            saved=read_json(root/"evaluations"/f"{directory.name}.json")
            if saved["world_ids"]!=ids or saved["truth"]!=labels.tolist(): raise ValueError("Evaluation identity mismatch")
            probs=predict(model,data)
            if saved["neural"]!=probs.argmax(1).tolist(): raise ValueError("Neural evaluation mismatch")
            anm=read_json(root/"programs"/"anm.json")
            for name,p in programs.items():
                pred=p.predict(features)
                if pred.tolist()!=saved["programs"][name]: raise ValueError("Program replay failed")
                m=evaluate(labels,probs,pred)
                if m!=saved["metrics"][name] or m!=summary["evaluation"][directory.name][name]:
                    raise ValueError("Evaluation metrics mismatch")
            m=evaluate(labels,probs,anm_predict(features,anm["independence_threshold"],anm["ambiguity_threshold"]))
            if m!=summary["evaluation"][directory.name]["anm"]: raise ValueError("ANM metrics mismatch")
    d=loaded["test_id"][0][:8]
    original=predict(model,d)
    if not np.allclose(original,predict(model,d[:,::-1].copy()),atol=1e-6): raise ValueError("Sample permutation violated")
    if not np.allclose(original[:,[1,0,2,3]],predict(model,d[:,:,::-1].copy()),atol=1e-6): raise ValueError("Swap equivariance violated")
    expected_splits={"train","dev","extraction","refinement","alignment_fit","alignment_test",
                     "test_id","test_function","test_noise","test_scale","test_intervention","attack"}
    if set(counts)!=expected_splits: raise ValueError("Missing or unexpected dataset split")
    attack=read_json(root/"counterexamples.json")
    ad,ay=loaded["attack"]
    with np.load(root/"datasets"/"attack"/"derived.npz") as a: ax=a["features"]
    for result in attack["strategies"].values():
        ids=np.array(result["indices"])
        neural=predict(model,ad[ids]).argmax(1)
        p=programs["full"].predict(ax[ids])
        if neural.tolist()!=result["teacher_predictions"]: raise ValueError("Attack neural mismatch")
        if int(np.sum((p==neural)&(neural!=ay[ids])))!=result["joint_errors"]: raise ValueError("Attack count mismatch")
    replayed_attack=compare_search(programs["full"],ax,ay,lambda ids:predict(model,ad[ids]),
                                   budget=summary["config"]["query_budget"],seed=summary["config"]["seed"])
    if replayed_attack!=attack or attack!=summary["counterexamples"]: raise ValueError("Search benchmark mismatch")
    alignment=read_json(root/"alignment.json")
    with np.load(root/"datasets"/"alignment_fit"/"derived.npz") as a: fit_x=a["features"]
    with np.load(root/"datasets"/"alignment_test"/"derived.npz") as a: test_x=a["features"]
    replayed_alignment,_=fit_and_test(model,programs["full"],loaded["alignment_fit"][0],fit_x,
                                     loaded["alignment_test"][0],test_x,summary["config"]["seed"])
    if replayed_alignment!=alignment or alignment!=summary["alignment"]: raise ValueError("Alignment metrics mismatch")
    if alignment["status"]=="measured":
        td=loaded["alignment_test"][0]
        with np.load(root/"alignment_state.npz") as a:
            h=hidden(model,td); z=(h-a["hidden_mean"])/a["hidden_std"]
            with np.load(root/"datasets"/"alignment_test"/"derived.npz") as f: tx=f["features"]
            idx=int(a["feature_index"]); source=a["source_indices"]
            target=(tx[source,idx]-a["target_mean"])/a["target_std"]
            symbolic=tx.copy();symbolic[:,idx]=tx[source,idx]
            if programs["full"].predict(symbolic).tolist()!=alignment["symbolic_after"]: raise ValueError("Symbolic intervention mismatch")
            for control in ("learned","random_direction","permuted_target"):
                patched=interchange(z,a[control],target)
                probs=logits_from_hidden(model,patched*a["hidden_std"]+a["hidden_mean"])
                if probs.argmax(1).tolist()!=alignment["controls"][control]["predictions"]:
                    raise ValueError("Neural intervention replay failed")
    if (root/"report.html").read_text(encoding="utf-8")!=render(summary,programs): raise ValueError("Report mismatch")
    return {"status":"verified","artifacts_checked":len(manifest["artifacts"]),"worlds_replayed":sum(counts.values()),
            "split_counts":counts,"checks":["hashes","SCM samples","root interventions","split isolation",
            "statistical primitives","frozen teacher","programs","metrics","swap equivariance","sample permutation",
            "counterexample outcomes","interchange interventions","HTML report"]}
