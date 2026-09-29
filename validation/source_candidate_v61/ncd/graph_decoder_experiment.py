"""Development-calibrated decoding of frozen multivariate graph networks."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,shutil
import numpy as np
from .io import save_json,read_json,digest
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .relational_graph import load_relational_graph
from .graph_model import graph_probabilities,pair_features,decode_graph,decode_graph_threshold,graph_metrics,graph_labels

MODES=("without_relations","with_relations")
ENVIRONMENTS=("test_id","test_function","test_noise","test_scale","test_intervention")
GRID=tuple(float(x) for x in np.linspace(.1,.9,33))

@dataclass
class GraphDecoderConfig:
    seed:int=1993
    nodes:tuple=(3,5,8)
    environments:tuple=ENVIRONMENTS
    worlds_per_cell:int=32
    samples:int=96
    thresholds:tuple=GRID
    @classmethod
    def quick(cls):return cls(seed=1992,nodes=(3,),environments=("test_id",),worlds_per_cell=2,samples=32,thresholds=(.3,.5,.7))
    def validate(self):
        if not self.nodes or any(n not in (3,5,8) for n in self.nodes):raise ValueError("Invalid nodes")
        if not self.environments or any(e not in ENVIRONMENTS for e in self.environments):raise ValueError("Invalid environments")
        if self.worlds_per_cell<1 or self.samples<16 or not self.thresholds:raise ValueError("Invalid decoder budget")
        if any(not 0<=t<=1 for t in self.thresholds):raise ValueError("Invalid thresholds")

def _metric(truth,pred):
    truth=np.asarray(truth,bool);pred=np.asarray(pred,bool);n=truth.shape[1];mask=np.triu(np.ones((n,n),bool),1)
    a=np.stack([graph_labels(x)[mask] for x in truth]);b=np.stack([graph_labels(x)[mask] for x in pred])
    m=graph_metrics(truth,pred,pred)
    return {"exact_graph_accuracy":float(np.mean(np.all(a==b,axis=1))),"pair_accuracy":float(np.mean(a==b)),
            "mean_pair_shd":float(np.mean(np.sum(a!=b,axis=1))),"skeleton_f1":m["skeleton_f1"],
            "directed_target_accuracy":m["directed_target_accuracy"]}

def _calibrate(model,source,n,thresholds):
    with np.load(source/"datasets"/f"n{n}_dev"/"features.npz") as z:features=z["features"];labels=z["labels"]
    truth=np.zeros(labels.shape,bool);truth[(labels==1)|(labels==3)]=True
    probabilities=graph_probabilities(model,features);rows=[]
    for threshold in thresholds:
        graphs=np.stack([decode_graph_threshold(p,threshold)[0] for p in probabilities]);metric=_metric(truth,graphs)
        rows.append({"threshold":threshold,**metric})
    chosen=max(rows,key=lambda r:(r["exact_graph_accuracy"],-r["mean_pair_shd"],-abs(r["threshold"]-.5),-r["threshold"]))
    return chosen,rows

def run_graph_decoder(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty decoder directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"config.json",{**asdict(config),"source":str(source)});save_json(root/"status.json",{"state":"running"})
    try:return _execute(root,source,config)
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def _execute(root,source,c):
    manifest=read_json(source/"manifest.json")["artifacts"];source_hashes={};models={};calibration={}
    for mode in MODES:
        model_path=source/"models"/mode/"graph_teacher.pt";key=model_path.relative_to(source).as_posix()
        if digest(model_path)!=manifest[key]:raise ValueError("Source model integrity mismatch")
        source_hashes[key]=manifest[key];models[mode]=load_relational_graph(model_path);calibration[mode]={}
        for n in c.nodes:
            feature_path=source/"datasets"/f"n{n}_dev"/"features.npz";key=feature_path.relative_to(source).as_posix()
            if digest(feature_path)!=manifest[key]:raise ValueError("Source development integrity mismatch")
            source_hashes[key]=manifest[key];chosen,trace=_calibrate(models[mode],source,n,c.thresholds);calibration[mode][str(n)]={"chosen":chosen,"trace":trace}
    records=[];seen=set()
    for n in c.nodes:
        for environment in c.environments:
            worlds=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples);ids={w.identity for w in worlds}
            if len(ids)!=len(worlds) or seen&ids:raise ValueError("World leakage")
            seen|=ids;path=root/"datasets"/f"n{n}_{environment}";save_graph_dataset(path,worlds)
            features=np.stack([pair_features(w.sample()) for w in worlds]);truth=np.stack([w.target_graph for w in worlds]);np.savez_compressed(path/"features.npz",features=features)
            for mode in MODES:
                p=graph_probabilities(models[mode],features);base=np.stack([decode_graph(x)[0] for x in p]);threshold=calibration[mode][str(n)]["chosen"]["threshold"];new=np.stack([decode_graph_threshold(x,threshold)[0] for x in p])
                np.savez_compressed(path/f"{mode}.npz",probabilities=p,argmax=base,calibrated=new)
                records.append({"mode":mode,"nodes":n,"environment":environment,"threshold":threshold,"argmax":_metric(truth,base),"calibrated":_metric(truth,new)})
    aggregate={method:{key:float(np.mean([r[method][key] for r in records if r[method][key] is not None])) for key in ("exact_graph_accuracy","pair_accuracy","mean_pair_shd","skeleton_f1","directed_target_accuracy")} for method in ("argmax","calibrated")}
    summary={"config":{**asdict(c),"source":str(source)},"source_hashes":source_hashes,"calibration":calibration,"world_count":len(seen),"records":records,"aggregate":aggregate,"scope":"frozen neural graph decoding; symbolic programs unchanged"};save_json(root/"summary.json",summary)
    src=root/"source";src.mkdir();[shutil.copy2(p,src/p.name) for p in Path(__file__).parent.glob("*.py")]
    save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary

def verify_graph_decoder(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        if digest(root/name)!=value:raise ValueError("Decoder artifact mismatch")
    raw=summary["config"];c=GraphDecoderConfig(**{**{k:v for k,v in raw.items() if k!="source"},"nodes":tuple(raw["nodes"]),"environments":tuple(raw["environments"]),"thresholds":tuple(raw["thresholds"])});source=Path(raw["source"])
    for name,value in summary["source_hashes"].items():
        if digest(source/name)!=value:raise ValueError("Decoder source changed")
    models={m:load_relational_graph(source/"models"/m/"graph_teacher.pt") for m in MODES};records=[];seen=set()
    for mode in MODES:
        for n in c.nodes:
            chosen,trace=_calibrate(models[mode],source,n,c.thresholds)
            if {"chosen":chosen,"trace":trace}!=summary["calibration"][mode][str(n)]:raise ValueError("Calibration mismatch")
    for n in c.nodes:
        for environment in c.environments:
            path=root/"datasets"/f"n{n}_{environment}";worlds=load_graph_worlds(path)
            if worlds!=generate_graph_worlds(environment,c.worlds_per_cell,n,c.seed,c.samples):raise ValueError("Decoder world mismatch")
            seen|={w.identity for w in worlds};features=np.stack([pair_features(w.sample()) for w in worlds]);truth=np.stack([w.target_graph for w in worlds])
            with np.load(path/"features.npz") as z:np.testing.assert_allclose(z["features"],features,atol=1e-9)
            for mode in MODES:
                p=graph_probabilities(models[mode],features);threshold=summary["calibration"][mode][str(n)]["chosen"]["threshold"];base=np.stack([decode_graph(x)[0] for x in p]);new=np.stack([decode_graph_threshold(x,threshold)[0] for x in p])
                with np.load(path/f"{mode}.npz") as z:np.testing.assert_allclose(z["probabilities"],p,rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(z["argmax"],base);np.testing.assert_array_equal(z["calibrated"],new)
                records.append({"mode":mode,"nodes":n,"environment":environment,"threshold":threshold,"argmax":_metric(truth,base),"calibrated":_metric(truth,new)})
    if records!=summary["records"]:raise ValueError("Decoder metrics mismatch")
    return {"status":"verified","worlds":len(seen),"strata":len(records),"science_not_certified":True}