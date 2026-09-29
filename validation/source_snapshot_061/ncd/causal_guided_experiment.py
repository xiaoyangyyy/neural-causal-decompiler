"""End-to-end causal-feature-guided symbolic program synthesis experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil,tempfile
import numpy as np

from .io import save_json,read_json,digest
from .model import load_model,predict,set_seed
from .worlds import generate_worlds,save_dataset
from .cdir import composed_features,Node
from .statistics import FEATURES
from .rules import fit_rule,Rule
from .program_trace import ProgramExecutor
from .metrics import evaluate
from .causal_feature_guidance import FeatureGuidanceConfig,fit_causal_feature_guidance

TESTS=("test_id","test_function","test_noise","test_scale","test_intervention")


@dataclass
class CausalGuidedConfig:
    seed:int=1393
    samples:int=96
    extraction_worlds:int=1024
    fit_worlds:int=1024
    validation_worlds:int=3072
    test_worlds:int=1024
    guard_worlds:int=0
    steps:int=120
    splits:int=6
    train_pairs:int=384
    validation_pairs:int=768
    penalties:tuple=(0.,.001,.005)
    alignment_weights:tuple=(.002,.01,.05)
    complexity_weight:float=.001
    support_weight:float=.02
    selection_policy:str="compensatory"
    @classmethod
    def quick(cls):
        return cls(seed=1392,samples=32,extraction_worlds=128,fit_worlds=192,validation_worlds=384,
            test_worlds=64,steps=8,splits=3,train_pairs=48,validation_pairs=105,
            penalties=(0.,.005),alignment_weights=(.01,))
    def validate(self):
        ints=("seed","samples","extraction_worlds","fit_worlds","validation_worlds","test_worlds","steps","splits","train_pairs","validation_pairs")
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in ints) or self.samples<16:
            raise ValueError("Invalid causal-guided budget")
        if not self.penalties or not self.alignment_weights or any(not np.isfinite(x) or x<0 for x in (*self.penalties,*self.alignment_weights,self.complexity_weight,self.support_weight)):
            raise ValueError("Invalid causal-guided weights")
        if type(self.guard_worlds) is not int or self.guard_worlds<0:raise ValueError("Invalid guard budget")
        if self.selection_policy not in ("compensatory","conservative","ood_conservative"):raise ValueError("Invalid causal-guided selection policy")
        if self.selection_policy=="ood_conservative" and self.guard_worlds<1:raise ValueError("OOD policy requires guard worlds")


def _config_dict(c):
    value=asdict(c)
    # Preserve byte/semantic replay of 0.7 compensatory artifacts.
    if value["selection_policy"]=="compensatory":value.pop("selection_policy")
    if value["guard_worlds"]==0:value.pop("guard_worlds")
    return value


def _selection_worlds(c):
    return {"extraction":generate_worlds("extraction",c.extraction_worlds,c.seed,c.samples),
        "fit":generate_worlds("alignment_fit",c.fit_worlds,c.seed,c.samples),
        "validation":generate_worlds("alignment_fit",c.validation_worlds,c.seed+10000,c.samples)}


def _used_features(rule):
    used=set()
    def expr(node):
        if node.op=="var":used.add(node.index)
        for child in node.args:expr(child)
    def tree(t):
        if "label" in t:return
        expr(Node.from_dict(t["expr"]));tree(t["left"]);tree(t["right"])
    tree(rule.tree);return sorted(used)


def _select(candidates,support,c):
    records=[]
    for item in candidates.values():
        used=_used_features(item["rule"]);causal=float(np.mean(support[used])) if used else 0.
        base=item["validation_fidelity"]-c.complexity_weight*item["rule"].complexity
        item["used_features"]=used;item["causal_support"]=causal;item["base_score"]=base;item["guided_score"]=base+c.support_weight*causal
        records.append(item)
    historical=[x for x in records if any(o["alignment_weight"]==0 for o in x["origins"])]
    if not historical:raise ValueError("Missing historical candidates")
    plain=max(historical,key=lambda x:(x["base_score"],-x["rule"].complexity,x["id"]))
    eligible=records if c.selection_policy=="compensatory" else [x for x in records if x["base_score"]>=plain["base_score"]-1e-12]
    if c.selection_policy=="ood_conservative":
        eligible=[x for x in eligible if all(x["guard_fidelity"][split]>=plain["guard_fidelity"][split]-1e-12 for split in TESTS)]
    guided=max(eligible,key=lambda x:(x["guided_score"],-x["rule"].complexity,x["id"]))
    if c.selection_policy in ("conservative","ood_conservative"):
        for item in records:
            item["validation_noninferior"]=bool(item["base_score"]>=plain["base_score"]-1e-12)
            if c.selection_policy=="ood_conservative":
                item["guard_noninferior"]={split:bool(item["guard_fidelity"][split]>=plain["guard_fidelity"][split]-1e-12) for split in TESTS}
    return plain,guided,records


def run_causal_guided(directory,source,c):
    c.validate();root=Path(directory).resolve();source=Path(source).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use a new causal-guided directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"});save_json(root/"config.json",_config_dict(c));set_seed(c.seed)
    try:
        teacher=source/"model"/"discoverer.pt";shutil.copy2(teacher,root/"source_teacher.pt");model=load_model(root/"source_teacher.pt")
        worlds=_selection_worlds(c);seen=set();data={};features={}
        for role,ws in worlds.items():
            ids={w.identity for w in ws}
            if len(ids)!=len(ws) or ids&seen:raise ValueError("Selection world leakage")
            seen|=ids;save_dataset(root/"selection_datasets"/role,ws);data[role]=np.stack([w.sample() for w in ws]);features[role]=composed_features(data[role])
            np.savez_compressed(root/"selection_datasets"/role/"features.npz",features=features[role])
        fg=FeatureGuidanceConfig(c.seed,c.steps,c.train_pairs,c.validation_pairs)
        support,guidance=fit_causal_feature_guidance(model,data["fit"],data["validation"],root/"feature_guidance",fg)
        extraction_labels=predict(model,data["extraction"]).argmax(1);validation_labels=predict(model,data["validation"]).argmax(1)
        candidates={}
        origins=[(p,0.) for p in c.penalties]+[(p,a) for p in c.penalties for a in c.alignment_weights]
        for penalty,alignment_weight in origins:
            rule,trace=fit_rule(features["extraction"],extraction_labels,FEATURES,penalty=penalty,
                beam_width=3,max_splits=c.splits,feature_alignment=support if alignment_weight else None,
                alignment_weight=alignment_weight)
            identifier=ProgramExecutor(rule).program_id
            if identifier not in candidates:
                candidates[identifier]={"id":identifier,"rule":rule,"origins":[],"trace_by_origin":[],
                    "validation_fidelity":float(np.mean(rule.predict(features["validation"])==validation_labels))}
            candidates[identifier]["origins"].append({"penalty":penalty,"alignment_weight":alignment_weight})
            candidates[identifier]["trace_by_origin"].append({"penalty":penalty,"alignment_weight":alignment_weight,"trace":trace})
        if c.selection_policy=="ood_conservative":
            for item in candidates.values():item["guard_fidelity"]={}
            for split in TESTS:
                ws=generate_worlds(split,c.guard_worlds,c.seed+20000,c.samples);ids={w.identity for w in ws}
                if len(ids)!=len(ws) or ids&seen:raise ValueError("Guard-world identity leakage")
                seen|=ids;save_dataset(root/"guard_datasets"/split,ws);raw=np.stack([w.sample() for w in ws])
                f=composed_features(raw);labels=predict(model,raw).argmax(1)
                np.savez_compressed(root/"guard_datasets"/split/"features.npz",features=f,teacher=labels)
                for item in candidates.values():item["guard_fidelity"][split]=float(np.mean(item["rule"].predict(f)==labels))
        plain,guided,records=_select(candidates,support,c)
        folder=root/"candidates";folder.mkdir()
        serial=[]
        for item in records:
            save_json(folder/(item["id"]+".json"),item["rule"].to_dict())
            serial.append({k:v for k,v in item.items() if k!="rule"})
        selection={"without_internal":plain["id"],"with_internal":guided["id"],"selection_changed":plain["id"]!=guided["id"],
            "candidate_count":len(records),"historical_candidate_count":sum(any(o["alignment_weight"]==0 for o in x["origins"]) for x in records),
            "candidates":serial,"feature_support":support.tolist(),"feature_names":list(FEATURES),
            "selected_guided_positive_support":bool(guided["causal_support"]>0),"selected_site":guidance["selected_site"],
            "test_worlds_accessed":False,"score_definition":"historical base over unaligned pool; guided base + .02 mean feature support over union pool"}
        if c.selection_policy=="conservative":
            selection.update({"selection_policy":"conservative","selected_validation_noninferior":guided["validation_noninferior"],
                "score_definition":"historical base over unaligned pool; guided score maximized only among candidates with base >= historical base"})
        elif c.selection_policy=="ood_conservative":
            selection.update({"selection_policy":"ood_conservative","selected_validation_noninferior":guided["validation_noninferior"],
                "selected_guard_noninferior":guided["guard_noninferior"],"guard_worlds_accessed_before_selection":True,
                "score_definition":"guided score maximized only among candidates noninferior on ordinary validation and all five OOD guards"})
        save_json(root/"selection.json",selection)
        programs={"without_internal":plain["rule"],"with_internal":guided["rule"]};evaluation={}
        for split in TESTS:
            ws=generate_worlds(split,c.test_worlds,c.seed,c.samples);ids={w.identity for w in ws}
            if len(ids)!=len(ws) or ids&seen:raise ValueError("Final-test identity leakage")
            seen|=ids;save_dataset(root/"tests"/split,ws);raw=np.stack([w.sample() for w in ws]);truth=np.array([w.label for w in ws])
            f=composed_features(raw);prob=predict(model,raw);pred={name:rule.predict(f) for name,rule in programs.items()}
            metrics={name:evaluate(truth,prob,value) for name,value in pred.items()}
            np.savez_compressed(root/"tests"/split/"predictions.npz",features=f,neural=prob,**pred)
            save_json(root/"tests"/split/"metrics.json",metrics);evaluation[split]=metrics
        mean={name:float(np.mean([evaluation[t][name]["fidelity"] for t in TESTS])) for name in programs}
        summary={"config":_config_dict(c),"source_teacher_sha256":digest(root/"source_teacher.pt"),"selection":{k:selection[k] for k in ("without_internal","with_internal","selection_changed")},
            "candidate_count":len(records),"selected_site":guidance["selected_site"],"positive_features":guidance["positive_features"],
            "evaluation":evaluation,"mean_fidelity":mean,"mean_fidelity_delta":mean["with_internal"]-mean["without_internal"],
            "teacher_fidelity_is_primary":True,"truth_accuracy_is_diagnostic":True}
        save_json(root/"summary.json",summary);save_json(root/"status.json",{"state":"completed"})
        snapshot=root/"source";snapshot.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snapshot/p.name)
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise


def _compare_artifacts(a,b):
    af={p.relative_to(a).as_posix():p for p in a.rglob("*") if p.is_file() and p.name!="manifest.json" and p.relative_to(a).parts[0]!="source"}
    bf={p.relative_to(b).as_posix():p for p in b.rglob("*") if p.is_file() and p.name!="manifest.json" and p.relative_to(b).parts[0]!="source"}
    if set(af)!=set(bf):raise ValueError("Causal-guided replay file coverage mismatch")
    for name,p in af.items():
        q=bf[name]
        if p.suffix==".json":
            if read_json(p)!=read_json(q):raise ValueError("Causal-guided JSON replay mismatch: "+name)
        elif p.suffix==".npz":
            with np.load(p) as x,np.load(q) as y:
                if set(x.files)!=set(y.files):raise ValueError("Causal-guided NPZ keys mismatch: "+name)
                for key in x.files:
                    if x[key].dtype.kind in "fc":np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-7)
                    else:np.testing.assert_array_equal(x[key],y[key])
        elif digest(p)!=digest(q):raise ValueError("Causal-guided byte replay mismatch: "+name)


def verify_causal_guided(directory):
    root=Path(directory).resolve()
    for name,expected in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=expected:raise ValueError("Causal-guided artifact mismatch")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete causal-guided run")
    c=CausalGuidedConfig(**read_json(root/"config.json"));c.validate()
    with tempfile.TemporaryDirectory(prefix="ncd_causal_guided_") as temp:
        base=Path(temp);source=base/"source";(source/"model").mkdir(parents=True);shutil.copy2(root/"source_teacher.pt",source/"model"/"discoverer.pt")
        replay=base/"replay";run_causal_guided(replay,source,c);_compare_artifacts(root,replay)
    s=read_json(root/"summary.json");sel=read_json(root/"selection.json")
    return {"status":"verified","worlds":c.extraction_worlds+c.fit_worlds+c.validation_worlds+len(TESTS)*(c.test_worlds+c.guard_worlds),
        "candidates":s["candidate_count"],"selection_changed":s["selection"]["selection_changed"],
        "positive_features":s["positive_features"],"selected_site":s["selected_site"],
        "mean_fidelity_delta":s["mean_fidelity_delta"],"science_not_certified":True}

