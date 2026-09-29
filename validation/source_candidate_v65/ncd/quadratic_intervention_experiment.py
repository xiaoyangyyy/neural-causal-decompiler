"""Linear versus diagonal-quadratic readout intervention experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .model import load_model,set_seed
from .rules import Rule
from .worlds import generate_worlds,save_dataset,load_worlds
from .raw_program_trace import RawDiscoveryExecutor,raw_scalar_groups,dependence_scalar_groups,regression_scalar_groups
from .numeric_audit import fit_readout_values,intervention_numeric_targets,audit_numeric_values,audit_oblique_numeric_values,NumericReadout
from .mechanism_pairs import compatible_combinations,execution_conditioned_pairs
from .fixed_numeric_mapping import train_fixed_numeric_mapping
from .oblique_mapping import train_fixed_oblique_mapping,biorthogonal_bases,measure_oblique_mapping
from .joint_alignment import JointPairs,measure_joint_mapping
from .neural_sites import SITES,SiteDecoder
from .quadratic_readout import fit_quadratic_readout_values,QuadraticNumericReadout,train_quadratic_orthogonal,train_quadratic_oblique
from .raw_numeric_experiment import _save_pairs,_load_pairs,_trace_values,_schema

@dataclass
class QuadraticInterventionConfig:
    seed:int=3193;samples:int=96;fit_worlds:int=1024;test_worlds:int=8192;train_pairs:int=384;test_pairs:int=2048;steps:int=120;rank:int=1;numeric_weight:float=.5
    @classmethod
    def quick(cls):return cls(seed=3192,samples=32,fit_worlds=128,test_worlds=384,train_pairs=48,test_pairs=96,steps=8)
    def validate(self):
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in ("seed","samples","fit_worlds","test_worlds","train_pairs","test_pairs","steps","rank")):raise ValueError("Invalid quadratic intervention budget")
        if self.samples<16 or self.numeric_weight<0 or not np.isfinite(self.numeric_weight):raise ValueError("Invalid quadratic intervention config")

def _addresses(executor):return raw_scalar_groups(executor)+dependence_scalar_groups(executor)+regression_scalar_groups(executor)
def _method_summary(numeric,behavioral):return {"numeric_nodes":numeric["per_node"],"behavioral":behavioral["overall"]}

def _execute_site(root,site,model,hf,ht,addresses,natural,train,train_targets,train_visited,test,test_targets,test_visited,c,save):
    folder=root/"sites"/site
    if save:folder.mkdir(parents=True)
    linear=fit_readout_values(hf,addresses,*natural["alignment_fit"]);quadratic=fit_quadratic_readout_values(hf,addresses,*natural["alignment_fit"])
    if save:save_json(folder/"linear_probe.json",linear.to_dict());save_json(folder/"quadratic_probe.json",quadratic.to_dict())
    else:
        if linear.to_dict()!=read_json(folder/"linear_probe.json") or quadratic.to_dict()!=read_json(folder/"quadratic_probe.json"):raise ValueError("Readout replay mismatch")
    decoder=SiteDecoder(model,site)
    lb,lbr=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,linear,rank=c.rank,steps=c.steps,seed=c.seed+3000,numeric_weight=c.numeric_weight)
    lr,lw,lrec=train_fixed_oblique_mapping(decoder,hf,train,train_targets,train_visited,linear,rank=c.rank,steps=c.steps,seed=c.seed,numeric_weight=c.numeric_weight,initial_write=lb.reshape(hf.shape[1],-1))
    qb,qbr=train_quadratic_orthogonal(decoder,hf,train,train_targets,train_visited,quadratic,rank=c.rank,steps=c.steps,seed=c.seed+6000,numeric_weight=c.numeric_weight)
    qr,qw,qrec=train_quadratic_oblique(decoder,hf,train,train_targets,train_visited,quadratic,rank=c.rank,steps=c.steps,seed=c.seed+7000,numeric_weight=c.numeric_weight,initial_write=qb.reshape(hf.shape[1],-1))
    br,bw,brec=train_quadratic_oblique(decoder,hf,train,train_targets,train_visited,quadratic,rank=c.rank,steps=c.steps,seed=c.seed+8000,numeric_weight=0.)
    sr,sw,srec=train_quadratic_oblique(decoder,hf,train,train_targets,train_visited,quadratic,rank=c.rank,steps=c.steps,seed=c.seed+9000,numeric_weight=c.numeric_weight,shuffle_targets=True)
    rng=np.random.default_rng(c.seed+10000);rr,rw=biorthogonal_bases(rng.normal(size=(hf.shape[1],len(addresses)*c.rank)),len(addresses),c.rank)
    records={"linear_orthogonal":lbr,"linear_oblique":lrec,"quadratic_orthogonal":qbr,"quadratic_oblique":qrec,"quadratic_behavior_only":brec,"quadratic_shuffled":srec}
    mappings={"linear_orthogonal":("orthogonal",linear,lb),"linear_oblique":("oblique",linear,lr,lw),"quadratic_orthogonal":("orthogonal",quadratic,qb),"quadratic_oblique":("oblique",quadratic,qr,qw),"quadratic_behavior_only":("oblique",quadratic,br,bw),"quadratic_shuffled":("oblique",quadratic,sr,sw),"quadratic_random":("oblique",quadratic,rr,rw)}
    if save:
        for name,rec in records.items():save_json(folder/(name+"_training.json"),rec)
        packed={}
        for name,m in mappings.items():
            if m[0]=="orthogonal":packed[name+"_basis"]=m[2]
            else:packed[name+"_read"]=m[2];packed[name+"_write"]=m[3]
        np.savez_compressed(folder/"mappings.npz",**packed)
    else:
        for name,rec in records.items():
            if rec!=read_json(folder/(name+"_training.json")):raise ValueError("Quadratic training replay mismatch")
        with np.load(folder/"mappings.npz") as z:
            expected={}
            for name,m in mappings.items():
                if m[0]=="orthogonal":expected[name+"_basis"]=m[2]
                else:expected[name+"_read"]=m[2];expected[name+"_write"]=m[3]
            if set(z.files)!=set(expected):raise ValueError("Quadratic mapping coverage mismatch")
            for k,v in expected.items():np.testing.assert_allclose(z[k],v,rtol=0,atol=1e-7)
    result={}
    for name,m in mappings.items():
        if m[0]=="orthogonal":numeric=audit_numeric_values(m[1],ht,m[2],test,*natural["alignment_test"],test_targets,test_visited);behavioral=measure_joint_mapping(decoder,ht,m[2],test)
        else:numeric=audit_oblique_numeric_values(m[1],ht,m[2],m[3],test,*natural["alignment_test"],test_targets,test_visited);behavioral=measure_oblique_mapping(decoder,ht,m[2],m[3],test)
        artifact={"numeric":numeric,"behavioral":behavioral}
        if save:save_json(folder/(name+".json"),artifact)
        elif artifact!=read_json(folder/(name+".json")):raise ValueError("Quadratic audit replay mismatch")
        result[name]=_method_summary(numeric,behavioral)
    return result

def run_quadratic_intervention(directory,source_directory,config):
    config.validate();root=Path(directory).resolve();source=Path(source_directory).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use an empty quadratic intervention directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/"status.json",{"state":"running"});set_seed(config.seed)
    try:
      shutil.copy2(source/"model"/"discoverer.pt",root/"teacher.pt");shutil.copy2(source/"program.json",root/"program.json");cfg={**asdict(config),"source":str(source),"source_model_sha256":digest(source/"model"/"discoverer.pt"),"source_program_sha256":digest(source/"program.json"),"sites":list(SITES),"scope":"all 54 scalar groups; paired linear/quadratic readouts"};save_json(root/"config.json",cfg);model=load_model(root/"teacher.pt");executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/"program.json")),trace_dependence=True,trace_regression=True);addresses=_addresses(executor);data={};natural={};seen=set()
      for split,count in (("alignment_fit",config.fit_worlds),("alignment_test",config.test_worlds)):
        worlds=generate_worlds(split,count,config.seed,config.samples);ids={w.identity for w in worlds}
        if len(ids)!=count or ids&seen:raise ValueError("Quadratic world leakage")
        seen|=ids;save_dataset(root/"datasets"/split,worlds);data[split]=np.stack([w.sample() for w in worlds]);trace=executor.execute(data[split]);natural[split]=_trace_values(trace,addresses);np.savez_compressed(root/"datasets"/split/"raw_trace.npz",features=trace.features,values=natural[split][0],visited=natural[split][1],output=trace.output)
      single=np.eye(len(addresses),dtype=bool);masks=compatible_combinations(executor,addresses,max_order=2);train,tr=execution_conditioned_pairs(executor,data["alignment_fit"],addresses,single,count=config.train_pairs,seed=config.seed+4000);train_targets,train_visited=intervention_numeric_targets(executor,data["alignment_fit"],addresses,train);test,te=execution_conditioned_pairs(executor,data["alignment_test"],addresses,masks,count=config.test_pairs,seed=config.seed+5000)
      if config.test_pairs>=len(masks) and (len(test.base)!=config.test_pairs or len(np.unique(test.masks,axis=0))!=len(masks)):raise ValueError("Quadratic full mask coverage exhausted")
      test_targets,test_visited=intervention_numeric_targets(executor,data["alignment_test"],addresses,test);_save_pairs(root/"train_pairs.npz",train,train_targets,train_visited);_save_pairs(root/"test_pairs.npz",test,test_targets,test_visited);save_json(root/"pairing.json",{"train":tr,"test":te});summary={"config":cfg,"groups":_schema(executor,addresses),"test_combinations":masks.tolist(),"sites":{}}
      for site in SITES:
        decoder=SiteDecoder(model,site);summary["sites"][site]=_execute_site(root,site,model,decoder.extract(data["alignment_fit"]),decoder.extract(data["alignment_test"]),addresses,natural,train,train_targets,train_visited,test,test_targets,test_visited,config,True)
      save_json(root/"summary.json",summary);snap=root/"source";snap.mkdir();[shutil.copy2(p,snap/p.name) for p in Path(__file__).parent.glob("*.py")];save_json(root/"status.json",{"state":"completed"});save_json(root/"manifest.json",{"artifacts":{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*")) if p.is_file()}});return summary
    except BaseException as exc:save_json(root/"status.json",{"state":"failed","type":type(exc).__name__,"error":str(exc)});raise

def verify_quadratic_intervention(directory):
    root=Path(directory).resolve();summary=read_json(root/"summary.json");cfg=read_json(root/"config.json")
    if read_json(root/"status.json")["state"]!="completed":raise ValueError("Incomplete quadratic run")
    for name,value in read_json(root/"manifest.json")["artifacts"].items():
      p=(root/name).resolve()
      if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Quadratic artifact mismatch")
    c=QuadraticInterventionConfig(**{k:v for k,v in cfg.items() if k in QuadraticInterventionConfig.__dataclass_fields__});c.validate();set_seed(c.seed)
    if cfg!=summary["config"] or digest(root/"teacher.pt")!=cfg["source_model_sha256"] or digest(root/"program.json")!=cfg["source_program_sha256"]:raise ValueError("Quadratic source mismatch")
    model=load_model(root/"teacher.pt");executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/"program.json")),trace_dependence=True,trace_regression=True);addresses=_addresses(executor);data={};natural={};seen=set()
    for split,count in (("alignment_fit",c.fit_worlds),("alignment_test",c.test_worlds)):
      worlds=load_worlds(root/"datasets"/split)
      if worlds!=generate_worlds(split,count,c.seed,c.samples):raise ValueError("Quadratic world mismatch")
      ids={w.identity for w in worlds}
      if len(ids)!=count or ids&seen:raise ValueError("Quadratic replay leakage")
      seen|=ids;data[split]=np.stack([w.sample() for w in worlds]);trace=executor.execute(data[split]);natural[split]=_trace_values(trace,addresses)
      with np.load(root/"datasets"/split/"raw_trace.npz") as z:np.testing.assert_allclose(z["features"],trace.features,atol=1e-10);np.testing.assert_allclose(z["values"],natural[split][0],atol=1e-10);np.testing.assert_array_equal(z["visited"],natural[split][1]);np.testing.assert_array_equal(z["output"],trace.output)
    if _schema(executor,addresses)!=summary["groups"]:raise ValueError("Quadratic schema mismatch")
    single=np.eye(len(addresses),dtype=bool);masks=compatible_combinations(executor,addresses,max_order=2);train,tr=execution_conditioned_pairs(executor,data["alignment_fit"],addresses,single,count=c.train_pairs,seed=c.seed+4000);train_targets,train_visited=intervention_numeric_targets(executor,data["alignment_fit"],addresses,train);test,te=execution_conditioned_pairs(executor,data["alignment_test"],addresses,masks,count=c.test_pairs,seed=c.seed+5000);test_targets,test_visited=intervention_numeric_targets(executor,data["alignment_test"],addresses,test)
    if {"train":tr,"test":te}!=read_json(root/"pairing.json"):raise ValueError("Quadratic pairing mismatch")
    for path,expected in ((root/"train_pairs.npz",(train,train_targets,train_visited)),(root/"test_pairs.npz",(test,test_targets,test_visited))):
      actual=_load_pairs(path)
      for a,b in zip(actual[0].__dict__.values(),expected[0].__dict__.values()):np.testing.assert_array_equal(a,b)
      np.testing.assert_allclose(actual[1],expected[1]);np.testing.assert_array_equal(actual[2],expected[2])
    for site in SITES:
      decoder=SiteDecoder(model,site);actual=_execute_site(root,site,model,decoder.extract(data["alignment_fit"]),decoder.extract(data["alignment_test"]),addresses,natural,train,train_targets,train_visited,test,test_targets,test_visited,c,False)
      if actual!=summary["sites"][site]:raise ValueError("Quadratic summary mismatch")
    return {"status":"verified","worlds":len(seen),"groups":len(addresses),"sites":list(SITES),"train_pairs":len(train.base),"test_pairs":len(test.base),"science_not_certified":True}

