"""Raw-sample statistical-operation interchange experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
import shutil
import numpy as np
from .io import save_json,read_json,digest
from .model import load_model,set_seed
from .rules import Rule
from .worlds import generate_worlds,save_dataset,load_worlds
from .raw_program_trace import RawDiscoveryExecutor,raw_scalar_groups,dependence_scalar_groups,regression_scalar_groups
from .numeric_audit import (numeric_values,fit_readout_values,intervention_numeric_targets,
    audit_numeric_values,audit_oblique_numeric_values,NumericReadout)
from .mechanism_pairs import compatible_combinations,execution_conditioned_pairs
from .fixed_numeric_mapping import train_fixed_numeric_mapping
from .oblique_mapping import (train_fixed_oblique_mapping,biorthogonal_bases,
    measure_oblique_mapping)
from .joint_alignment import JointPairs,measure_joint_mapping
from .neural_sites import SITES,SiteDecoder

@dataclass
class RawNumericConfig:
    seed:int=793;samples:int=96;fit_worlds:int=1024;test_worlds:int=4096
    train_pairs:int=384;test_pairs:int=1024;steps:int=120;rank:int=1;numeric_weight:float=.5;target_scope:str="base"
    mapping_kind:str="orthogonal"
    @classmethod
    def quick(cls):return cls(seed=791,samples=32,fit_worlds=128,test_worlds=384,train_pairs=48,test_pairs=96,steps=8)
    def validate(self):
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in ('seed','samples','fit_worlds','test_worlds','train_pairs','test_pairs','steps','rank')):raise ValueError("Invalid raw numeric budget")
        if self.samples<16 or not np.isfinite(self.numeric_weight) or self.numeric_weight<0 or self.target_scope not in ("base","dependence","regression","all") or self.mapping_kind not in ("orthogonal","oblique"):raise ValueError("Invalid raw numeric configuration")

def _save_pairs(path,pairs,targets,visited):
    np.savez_compressed(path,base=pairs.base,sources=pairs.sources,masks=pairs.masks,target=pairs.target,initial=pairs.initial,
        source_visited=pairs.source_visited,target_visited=pairs.target_visited,numeric_targets=targets,numeric_visited=visited)
def _load_pairs(path):
    with np.load(path) as z:
        pairs=JointPairs(*(z[k] for k in ('base','sources','masks','target','initial','source_visited','target_visited')))
        return pairs,z['numeric_targets'],z['numeric_visited']
def _trace_values(trace,addresses):
    values=[];visited=[]
    for address in addresses:
        record=trace.records[address];values.append(np.asarray(record['value']));visited.append(np.asarray(record['visited'],dtype=bool))
    return np.column_stack(values),np.column_stack(visited)

def _schema(executor,addresses):
    rows=[]
    for address in addresses:
        path=address.split(':',1)[1];members=executor.catalog[path]['members'];member=members[0]
        if member in executor._nodes:
            node=executor._nodes[member];ast=node.to_dict();operation=node.op
        else:
            meta=executor.catalog[member];parent_path=meta.get('dep_parent',meta.get('reg_parent'));parent=executor._nodes[parent_path]
            ast={'op':meta['op'],'role':meta.get('dep_role',meta.get('reg_role')),'parent':parent.to_dict(),'fold':meta.get('reg_fold')};operation=meta['op']
        rows.append({'address':address,'path':path,'ast':ast,'operation':operation,'occurrences':list(members)})
    return rows

def run_raw_numeric(directory,source,c):
    c.validate();root=Path(directory).resolve();source=Path(source).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError("Use a new raw numeric directory")
    root.mkdir(parents=True,exist_ok=True);save_json(root/'status.json',{'state':'running'});set_seed(c.seed)
    config={**asdict(c),'source_model_sha256':digest(source/'model'/'discoverer.pt'),'source_program_sha256':digest(source/'program.json'),
        'sites':list(SITES),'selection':'all sites and controls reported; no test selection'};save_json(root/'config.json',config)
    try:
        shutil.copy2(source/'model'/'discoverer.pt',root/'teacher.pt');shutil.copy2(source/'program.json',root/'program.json')
        model=load_model(root/'teacher.pt');executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/'program.json')),trace_dependence=c.target_scope in ('dependence','all'),trace_regression=c.target_scope in ('regression','all'));addresses=(raw_scalar_groups(executor)+dependence_scalar_groups(executor)+regression_scalar_groups(executor) if c.target_scope=='all' else dependence_scalar_groups(executor) if c.target_scope=='dependence' else regression_scalar_groups(executor) if c.target_scope=='regression' else raw_scalar_groups(executor))
        data={};features={};natural={};seen=set()
        for split,count in (('alignment_fit',c.fit_worlds),('alignment_test',c.test_worlds)):
            print('raw worlds',split,count,flush=True);worlds=generate_worlds(split,count,c.seed,c.samples);ids={w.identity for w in worlds}
            if len(ids)!=count or ids&seen:raise ValueError("Raw world leakage")
            seen|=ids;save_dataset(root/'datasets'/split,worlds);data[split]=np.stack([w.sample() for w in worlds])
            trace=executor.execute(data[split]);features[split]=trace.features;natural[split]=_trace_values(trace,addresses)
            np.savez_compressed(root/'datasets'/split/'raw_trace.npz',features=trace.features,values=natural[split][0],visited=natural[split][1],output=trace.output)
        single=np.eye(len(addresses),dtype=bool);all_masks=compatible_combinations(executor,addresses,max_order=2)
        train,train_record=execution_conditioned_pairs(executor,data['alignment_fit'],addresses,single,count=c.train_pairs,seed=c.seed+4000)
        train_targets,train_visited=intervention_numeric_targets(executor,data['alignment_fit'],addresses,train)
        test,test_record=execution_conditioned_pairs(executor,data['alignment_test'],addresses,all_masks,count=c.test_pairs,seed=c.seed+5000)
        if c.mapping_kind=='oblique' and c.target_scope=='all' and c.test_pairs>=len(all_masks):
            if len(test.base)!=c.test_pairs or len(np.unique(test.masks,axis=0))!=len(all_masks):
                raise ValueError("Full oblique test-mask coverage budget exhausted")
        test_targets,test_visited=intervention_numeric_targets(executor,data['alignment_test'],addresses,test)
        _save_pairs(root/'train_pairs.npz',train,train_targets,train_visited);_save_pairs(root/'test_pairs.npz',test,test_targets,test_visited)
        save_json(root/'pairing.json',{'train':train_record,'test':test_record})
        summary={'config':config,'groups':_schema(executor,addresses),'train_combinations':single.tolist(),'test_combinations':all_masks.tolist(),
            'held_out_combinations':[m.tolist() for m in all_masks if m.sum()>1],'sites':{},
            'scope':('all raw feature-statistic, dependence-kernel, and cross-fit regression scalar groups' if c.target_scope=='all' else 'dependence kernel bandwidth/energy/numerator/denominator groups; regression internals remain opaque' if c.target_scope=='dependence' else 'cross-fit regression fold normalization and basis-coefficient groups' if c.target_scope=='regression' else 'raw-sample mean/variance/std/correlation/variance-ratio operation groups; dependence and regression internals remain opaque')}
        for site in SITES:
            print('raw numeric site',site,flush=True);folder=root/'sites'/site;folder.mkdir(parents=True)
            decoder=SiteDecoder(model,site);hf=decoder.extract(data['alignment_fit']);ht=decoder.extract(data['alignment_test'])
            probe=fit_readout_values(hf,addresses,*natural['alignment_fit']);save_json(folder/'probe.json',probe.to_dict())
            summary['sites'][site]={}
            if c.mapping_kind=='orthogonal':
                methods={}
                for name,weight,shuffle,offset in (('numeric',c.numeric_weight,False,0),('behavior_only',0.,False,0),('shuffled',c.numeric_weight,True,1000)):
                    basis,record=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                        seed=c.seed+offset,numeric_weight=weight,shuffle_targets=shuffle);methods[name]=basis;save_json(folder/(name+'_training.json'),record)
                rng=np.random.default_rng(c.seed+2000);methods['random']=np.linalg.qr(rng.normal(size=(hf.shape[1],len(addresses)*c.rank)))[0].reshape(hf.shape[1],len(addresses),c.rank)
                np.savez_compressed(folder/'bases.npz',**methods)
                for name,basis in methods.items():
                    numeric=audit_numeric_values(probe,ht,basis,test,*natural['alignment_test'],test_targets,test_visited)
                    behavioral=measure_joint_mapping(decoder,ht,basis,test);save_json(folder/(name+'.json'),{'numeric':numeric,'behavioral':behavioral})
                    summary['sites'][site][name]={'numeric_nodes':numeric['per_node'],'behavioral':behavioral['overall']}
            else:
                methods={}
                basis,record=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                    seed=c.seed+3000,numeric_weight=c.numeric_weight);methods['orthogonal_numeric']=('orthogonal',basis);save_json(folder/'orthogonal_numeric_training.json',record)
                for name,weight,shuffle,offset in (('oblique_numeric',c.numeric_weight,False,0),('oblique_behavior_only',0.,False,0),('oblique_shuffled',c.numeric_weight,True,1000)):
                    warm=basis.reshape(hf.shape[1],-1) if name=='oblique_numeric' else None
                    read,write,record=train_fixed_oblique_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                        seed=c.seed+offset,numeric_weight=weight,shuffle_targets=shuffle,initial_write=warm);methods[name]=('oblique',read,write);save_json(folder/(name+'_training.json'),record)
                rng=np.random.default_rng(c.seed+2000);read,write=biorthogonal_bases(rng.normal(size=(hf.shape[1],len(addresses)*c.rank)),len(addresses),c.rank)
                methods['oblique_random']=('oblique',read,write)
                packed={}
                for name,method in methods.items():
                    if method[0]=='orthogonal':packed[name+'_basis']=method[1]
                    else:packed[name+'_read']=method[1];packed[name+'_write']=method[2]
                np.savez_compressed(folder/'mappings.npz',**packed)
                for name,method in methods.items():
                    if method[0]=='orthogonal':
                        numeric=audit_numeric_values(probe,ht,method[1],test,*natural['alignment_test'],test_targets,test_visited)
                        behavioral=measure_joint_mapping(decoder,ht,method[1],test)
                    else:
                        numeric=audit_oblique_numeric_values(probe,ht,method[1],method[2],test,*natural['alignment_test'],test_targets,test_visited)
                        behavioral=measure_oblique_mapping(decoder,ht,method[1],method[2],test)
                    save_json(folder/(name+'.json'),{'numeric':numeric,'behavioral':behavioral})
                    summary['sites'][site][name]={'numeric_nodes':numeric['per_node'],'behavioral':behavioral['overall']}
        save_json(root/'summary.json',summary);save_json(root/'status.json',{'state':'completed'});snapshot=root/'source';snapshot.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
        save_json(root/'manifest.json',{'artifacts':{p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file()}});return summary
    except BaseException as exc:save_json(root/'status.json',{'state':'failed','type':type(exc).__name__,'error':str(exc)});raise

def verify_raw_numeric(directory):
    root=Path(directory).resolve()
    for name,value in read_json(root/'manifest.json')['artifacts'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=value:raise ValueError("Raw numeric artifact mismatch")
    if read_json(root/'status.json')['state']!='completed':raise ValueError("Incomplete raw numeric run")
    c=RawNumericConfig(**{k:v for k,v in read_json(root/'config.json').items() if k in RawNumericConfig.__dataclass_fields__});c.validate();set_seed(c.seed)
    config=read_json(root/'config.json');summary=read_json(root/'summary.json')
    if config!=summary['config'] or digest(root/'teacher.pt')!=config['source_model_sha256'] or digest(root/'program.json')!=config['source_program_sha256']:raise ValueError("Raw source mismatch")
    model=load_model(root/'teacher.pt');executor=RawDiscoveryExecutor(Rule.from_dict(read_json(root/'program.json')),trace_dependence=c.target_scope in ('dependence','all'),trace_regression=c.target_scope in ('regression','all'));addresses=(raw_scalar_groups(executor)+dependence_scalar_groups(executor)+regression_scalar_groups(executor) if c.target_scope=='all' else dependence_scalar_groups(executor) if c.target_scope=='dependence' else regression_scalar_groups(executor) if c.target_scope=='regression' else raw_scalar_groups(executor))
    if _schema(executor,addresses)!=summary['groups']:raise ValueError("Raw group schema mismatch")
    data={};natural={};seen=set()
    for split,count in (('alignment_fit',c.fit_worlds),('alignment_test',c.test_worlds)):
        worlds=load_worlds(root/'datasets'/split)
        if worlds!=generate_worlds(split,count,c.seed,c.samples):raise ValueError("Raw world mismatch")
        ids={w.identity for w in worlds}
        if len(ids)!=count or ids&seen:raise ValueError("Raw world reuse")
        seen|=ids;data[split]=np.stack([w.sample() for w in worlds]);trace=executor.execute(data[split]);natural[split]=_trace_values(trace,addresses)
        with np.load(root/'datasets'/split/'raw_trace.npz') as z:
            np.testing.assert_allclose(trace.features,z['features'],atol=1e-10);np.testing.assert_array_equal(trace.output,z['output']);np.testing.assert_allclose(natural[split][0],z['values'],atol=1e-10);np.testing.assert_array_equal(natural[split][1],z['visited'])
    single=np.eye(len(addresses),dtype=bool);masks=compatible_combinations(executor,addresses,max_order=2);pair_record=read_json(root/'pairing.json')
    train,record=execution_conditioned_pairs(executor,data['alignment_fit'],addresses,single,count=c.train_pairs,seed=c.seed+4000)
    train_targets,train_visited=intervention_numeric_targets(executor,data['alignment_fit'],addresses,train)
    test,trecord=execution_conditioned_pairs(executor,data['alignment_test'],addresses,masks,count=c.test_pairs,seed=c.seed+5000)
    if c.mapping_kind=='oblique' and c.target_scope=='all' and c.test_pairs>=len(masks):
        if len(test.base)!=c.test_pairs or len(np.unique(test.masks,axis=0))!=len(masks):
            raise ValueError("Full oblique test-mask coverage budget exhausted")
    test_targets,test_visited=intervention_numeric_targets(executor,data['alignment_test'],addresses,test)
    if record!=pair_record['train'] or trecord!=pair_record['test']:raise ValueError("Raw pairing mismatch")
    for path,expected in ((root/'train_pairs.npz',(train,train_targets,train_visited)),(root/'test_pairs.npz',(test,test_targets,test_visited))):
        actual=_load_pairs(path)
        for a,b in zip(actual[0].__dict__.values(),expected[0].__dict__.values()):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(actual[1],expected[1]);np.testing.assert_array_equal(actual[2],expected[2])
    for site in SITES:
        folder=root/'sites'/site;decoder=SiteDecoder(model,site);hf=decoder.extract(data['alignment_fit']);ht=decoder.extract(data['alignment_test'])
        probe=fit_readout_values(hf,addresses,*natural['alignment_fit']);saved=NumericReadout.from_dict(read_json(folder/'probe.json'))
        if probe.to_dict()!=saved.to_dict():raise ValueError("Raw probe mismatch")
        if c.mapping_kind=='orthogonal':
            with np.load(folder/'bases.npz') as z:bases={k:z[k] for k in z.files}
            expected_methods={}
            for name,weight,shuffle,offset in (('numeric',c.numeric_weight,False,0),('behavior_only',0.,False,0),('shuffled',c.numeric_weight,True,1000)):
                basis,record=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                    seed=c.seed+offset,numeric_weight=weight,shuffle_targets=shuffle);expected_methods[name]=basis
                if record!=read_json(folder/(name+'_training.json')):raise ValueError("Raw mapping training replay mismatch")
            rng=np.random.default_rng(c.seed+2000);expected_methods['random']=np.linalg.qr(rng.normal(size=(hf.shape[1],len(addresses)*c.rank)))[0].reshape(hf.shape[1],len(addresses),c.rank)
            if set(bases)!=set(expected_methods):raise ValueError("Raw method coverage mismatch")
            methods={name:('orthogonal',basis) for name,basis in bases.items()}
            for name,basis in bases.items():np.testing.assert_allclose(basis,expected_methods[name],rtol=0,atol=1e-7)
        else:
            expected_methods={}
            basis,record=train_fixed_numeric_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                seed=c.seed+3000,numeric_weight=c.numeric_weight);expected_methods['orthogonal_numeric']=('orthogonal',basis)
            if record!=read_json(folder/'orthogonal_numeric_training.json'):raise ValueError("Orthogonal comparator replay mismatch")
            for name,weight,shuffle,offset in (('oblique_numeric',c.numeric_weight,False,0),('oblique_behavior_only',0.,False,0),('oblique_shuffled',c.numeric_weight,True,1000)):
                warm=basis.reshape(hf.shape[1],-1) if name=='oblique_numeric' else None
                read,write,record=train_fixed_oblique_mapping(decoder,hf,train,train_targets,train_visited,probe,rank=c.rank,steps=c.steps,
                    seed=c.seed+offset,numeric_weight=weight,shuffle_targets=shuffle,initial_write=warm);expected_methods[name]=('oblique',read,write)
                if record!=read_json(folder/(name+'_training.json')):raise ValueError("Oblique mapping training replay mismatch")
            rng=np.random.default_rng(c.seed+2000);read,write=biorthogonal_bases(rng.normal(size=(hf.shape[1],len(addresses)*c.rank)),len(addresses),c.rank)
            expected_methods['oblique_random']=('oblique',read,write)
            with np.load(folder/'mappings.npz') as z:packed={k:z[k] for k in z.files}
            methods={}
            for name,method in expected_methods.items():
                if method[0]=='orthogonal':
                    actual=('orthogonal',packed.pop(name+'_basis'));np.testing.assert_allclose(actual[1],method[1],rtol=0,atol=1e-7)
                else:
                    actual=('oblique',packed.pop(name+'_read'),packed.pop(name+'_write'))
                    np.testing.assert_allclose(actual[1],method[1],rtol=0,atol=1e-7);np.testing.assert_allclose(actual[2],method[2],rtol=0,atol=1e-7)
                methods[name]=actual
            if packed:raise ValueError("Oblique method coverage mismatch")
        for name,method in methods.items():
            if method[0]=='orthogonal':
                numeric=audit_numeric_values(probe,ht,method[1],test,*natural['alignment_test'],test_targets,test_visited);behavioral=measure_joint_mapping(decoder,ht,method[1],test)
            else:
                numeric=audit_oblique_numeric_values(probe,ht,method[1],method[2],test,*natural['alignment_test'],test_targets,test_visited);behavioral=measure_oblique_mapping(decoder,ht,method[1],method[2],test)
            if {'numeric':numeric,'behavioral':behavioral}!=read_json(folder/(name+'.json')):raise ValueError("Raw audit mismatch")
            if {'numeric_nodes':numeric['per_node'],'behavioral':behavioral['overall']}!=summary['sites'][site][name]:raise ValueError("Raw summary mismatch")
    return {'status':'verified','worlds':len(seen),'groups':len(addresses),'sites':list(SITES),'train_pairs':len(train.base),'test_pairs':len(test.base),'science_not_certified':True}
