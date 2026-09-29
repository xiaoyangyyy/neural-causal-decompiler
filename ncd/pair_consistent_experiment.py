"""Paired raw versus pair-consistent factorized training experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy,shutil,tempfile
import numpy as np
import torch
from torch import nn
from .io import save_json,read_json,digest
from .model import set_seed
from .multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds
from .graph_model import GraphDiscoverer,GRAPH_FEATURES,pair_features,graph_labels,graph_probabilities,decode_graph,graph_metrics
from .node_context_graph import NodeContextGraphDiscoverer
from .factorized_node_context_graph import FactorizedNodeContextGraphDiscoverer
from .pair_consistent_graph import symmetrize_factorized_components

MODES=("raw_factorized","pair_consistent")
TESTS=("test_id","test_function","test_noise","test_scale","test_intervention")


@dataclass
class PairConsistentConfig:
    seed:int=4593
    nodes:tuple=(3,5,8)
    samples:int=96
    train_worlds:int=256
    dev_worlds:int=64
    test_worlds:int=96
    epochs:int=40
    width:int=48
    @classmethod
    def quick(cls):return cls(seed=4592,samples=32,train_worlds=8,dev_worlds=4,test_worlds=4,epochs=2,width=16)
    def validate(self):
        if tuple(self.nodes)!=(3,5,8) or self.width%4:raise ValueError("Invalid pair-consistent training shape")
        for k,v in asdict(self).items():
            if k!="nodes" and (type(v) is not int or v<1):raise ValueError("Invalid pair-consistent training budget")
        if self.samples<16:raise ValueError("Insufficient pair-consistent training samples")


def _counts(c):return {"train":c.train_worlds,"dev":c.dev_worlds,**{x:c.test_worlds for x in TESTS}}


def _prepare_model(model,groups):
    flat=np.concatenate([x[:,~np.eye(x.shape[1],dtype=bool)].reshape(-1,len(GRAPH_FEATURES)) for x,y in groups])
    model.mean.copy_(torch.tensor(flat.mean(0),dtype=torch.float32))
    model.std.copy_(torch.tensor(flat.std(0).clip(.03),dtype=torch.float32))
    counts=np.zeros(4)
    for x,y in groups:
        counts+=np.bincount(y[:,~np.eye(y.shape[1],dtype=bool)].ravel(),minlength=4)
    return counts


def _train_one(model,groups,dev,orders,directory,c,mode):
    counts=_prepare_model(model,groups)
    skeleton_loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts[0]/max(counts[1:].sum(),1),dtype=torch.float32))
    orientation_loss=nn.CrossEntropyLoss(weight=torch.tensor(counts[1:].sum()/(3*np.maximum(counts[1:],1)),dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    history=[];best=float("inf");state=None;best_epoch=0
    for epoch in range(c.epochs):
        model.train()
        for group_index,(x,y) in enumerate(groups):
            mask=~torch.eye(x.shape[1],dtype=torch.bool)
            for ids in np.array_split(orders[epoch][group_index],max(1,int(np.ceil(len(x)/24)))):
                batch=torch.as_tensor(x[ids],dtype=torch.float32)
                truth=torch.as_tensor(y[ids],dtype=torch.long)
                optimizer.zero_grad()
                hidden=model.representation(batch);skel,orient=model.raw_components(hidden)
                if mode=="pair_consistent":skel,orient=symmetrize_factorized_components(skel,orient)
                labels=truth[:,mask].reshape(-1);edge=labels>0
                loss=skeleton_loss(skel[:,mask].reshape(-1),edge.to(torch.float32))
                if edge.any():
                    loss=loss+orientation_loss(orient[:,mask].reshape(-1,3)[edge],labels[edge]-1)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        losses=[]
        for x,y in dev:
            p=graph_probabilities(model,x);mask=~np.eye(x.shape[1],dtype=bool)
            losses.append(float(-np.log(np.take_along_axis(p[:,mask],y[:,mask,None],axis=-1).clip(1e-9)).mean()))
        value=float(np.mean(losses));history.append({"epoch":epoch+1,"dev_cross_entropy":value})
        if value<best:best=value;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch==0 or (epoch+1)%10==0:print(f"{mode} epoch {epoch+1}/{c.epochs}: dev loss {value:.4f}",flush=True)
    model.load_state_dict(state);model.eval();directory.mkdir(parents=True)
    architecture="factorized_raw_loss_v1" if mode=="raw_factorized" else "factorized_pair_consistent_loss_v1"
    torch.save({"width":c.width,"state_dict":state,"architecture":architecture},directory/"graph_teacher.pt")
    record={"seed":c.seed,"history":history,"best_epoch":best_epoch,"architecture":architecture,
            "parameter_count":sum(p.numel() for p in model.parameters()),"batch_orders":orders,
            "selection":"minimum unweighted dev cross-entropy; test labels not used"}
    save_json(directory/"training.json",record);return model,record


def _train_paired(groups,dev,root,c):
    set_seed(c.seed);baseline=FactorizedNodeContextGraphDiscoverer(c.width)
    candidate=FactorizedNodeContextGraphDiscoverer(c.width);candidate.load_state_dict(baseline.state_dict())
    rng=np.random.default_rng(c.seed+777)
    orders=[[rng.permutation(len(x)).tolist() for x,y in groups] for _ in range(c.epochs)]
    result={};records={}
    for mode,model in (("raw_factorized",baseline),("pair_consistent",candidate)):
        result[mode],records[mode]=_train_one(model,groups,dev,orders,root/mode,c,mode)
    return result,records
def _evaluate(models,groups,root,save):
    evaluation={mode:{} for mode in MODES}
    for mode,model in models.items():
        for split in TESTS:
            for g in groups[split]:
                key=f"n{g['nodes']}_{split}";p=graph_probabilities(model,g["features"])
                graphs=np.stack([decode_graph(x)[0] for x in p]);metric=graph_metrics(g["target"],graphs,graphs)
                evaluation[mode][key]=metric
                if save:
                    path=root/"evaluations"/mode/key;path.mkdir(parents=True)
                    np.savez_compressed(path/"predictions.npz",probabilities=p,graphs=graphs)
    aggregate={}
    for mode in MODES:
        values=list(evaluation[mode].values())
        aggregate[mode]={k:float(np.mean([x[k] for x in values])) for k in
            ("macro_f1","skeleton_f1","mean_pair_shd","exact_graph_accuracy","directed_target_accuracy")}
        aggregate[mode]["parameter_count"]=sum(p.numel() for p in models[mode].parameters())
        aggregate[mode]["by_environment_exact"]={split:float(np.mean([evaluation[mode][f"n{n}_{split}"]["exact_graph_accuracy"] for n in (3,5,8)])) for split in TESTS}
    return evaluation,aggregate


def _datasets(root,c,save):
    groups={x:[] for x in _counts(c)};seen=set()
    for n in c.nodes:
        for split,count in _counts(c).items():
            path=root/"datasets"/f"n{n}_{split}"
            worlds=generate_graph_worlds(split,count,n,c.seed,c.samples) if save else load_graph_worlds(path)
            if not save and worlds!=generate_graph_worlds(split,count,n,c.seed,c.samples):raise ValueError("Pair-consistent training world mismatch")
            ids={w.identity for w in worlds}
            if len(ids)!=len(worlds) or seen&ids:raise ValueError("Pair-consistent training world leakage")
            seen|=ids
            if save:save_graph_dataset(path,worlds)
            data=np.stack([w.sample() for w in worlds]);features=np.stack([pair_features(x) for x in data])
            target=np.stack([w.target_graph for w in worlds]);labels=np.stack([graph_labels(x) for x in target])
            if save:np.savez_compressed(path/"features.npz",features=features,labels=labels)
            else:
                with np.load(path/"samples.npz") as a:np.testing.assert_array_equal(a["data"],data)
                with np.load(path/"features.npz") as a:
                    np.testing.assert_allclose(a["features"],features,atol=1e-9);np.testing.assert_array_equal(a["labels"],labels)
            groups[split].append({"nodes":n,"features":features,"labels":labels,"target":target})
    return groups,len(seen)


def run_pair_consistent(directory,config):
    config.validate();root=Path(directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty pair-consistent training directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"});save_json(root/"config.json",asdict(config))
    try:
        groups,worlds=_datasets(root,config,True)
        train=[(g["features"],g["labels"]) for g in groups["train"]];dev=[(g["features"],g["labels"]) for g in groups["dev"]]
        models,training=_train_paired(train,dev,root/"models",config)
        evaluation,aggregate=_evaluate(models,groups,root,True)
        summary={"config":asdict(config),"worlds":worlds,"evaluation":evaluation,"aggregate":aggregate,
            "training":{m:{"best_epoch":x["best_epoch"],"parameter_count":x["parameter_count"]} for m,x in training.items()},
            "pooling":"macro average over the 15 equal-budget node-size/environment cells",
            "claim":"pair-consistent loss comparison on a new teacher architecture; not historical-teacher decompilation"}
        save_json(root/"summary.json",summary);snap=root/"source";snap.mkdir()
        for p in Path(__file__).parent.glob("*.py"):shutil.copy2(p,snap/p.name)
        save_json(root/"status.json",{"state":"completed"})
        save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}})
        return summary
    except BaseException as exc:
        save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise


def verify_pair_consistent(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete pair-consistent training run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Pair-consistent training artifact mismatch")
    raw=read_json(root/"config.json");c=PairConsistentConfig(**{**raw,"nodes":tuple(raw["nodes"])});c.validate()
    groups,count=_datasets(root,c,False);train=[(g["features"],g["labels"]) for g in groups["train"]];dev=[(g["features"],g["labels"]) for g in groups["dev"]]
    with tempfile.TemporaryDirectory(prefix="verify_pair_consistent_",dir=root) as temp:
        models,training=_train_paired(train,dev,Path(temp),c)
        evaluation,aggregate=_evaluate(models,groups,Path(temp),False)
    if evaluation!=summary["evaluation"] or aggregate!=summary["aggregate"]:raise ValueError("Pair-consistent training metric replay mismatch")
    for mode in MODES:
        saved=torch.load(root/"models"/mode/"graph_teacher.pt",map_location="cpu",weights_only=True)
        for key,value in models[mode].state_dict().items():torch.testing.assert_close(value,saved["state_dict"][key],rtol=0,atol=0)
        if training[mode]!=read_json(root/"models"/mode/"training.json"):raise ValueError("Pair-consistent training training replay mismatch")
    return {"status":"verified","worlds":count,"modes":list(MODES),"node_sizes":list(c.nodes),
            "test_environments":list(TESTS),"science_not_certified":True}

