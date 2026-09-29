from pathlib import Path
Path('ncd/raw_program_trace.py').write_text('''"""Occurrence-level execution from raw datasets through CDIR features to a Rule."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
from .cdir import discovery_feature_nodes
from .program_trace import _apply
from .rules import Rule
from .statistics import FEATURES

def _key(value):return json.dumps(value,sort_keys=True,separators=(",",":"))

@dataclass
class RawExecutionTrace:
    program_id:str
    output:np.ndarray
    features:np.ndarray
    records:dict
    def to_dict(self):
        return {"program_id":self.program_id,"output":self.output.tolist(),"features":self.features.tolist(),
            "records":{k:{q:(np.asarray(v[q]).tolist() if q in ("value","visited") else v[q]) for q in
                ("kind","semantic_kind","op","value","visited","intervened","members") if q in v}
                for k,v in self.records.items()}}

class RawDiscoveryExecutor:
    """Execute raw-data feature ASTs and expose scalar operation occurrences."""
    _DEP_ROLES=("bandwidth_0","bandwidth_1","energy_0","energy_1","numerator","denominator")
    _DEP_ANCESTORS={"bandwidth_0":{"energy_0","numerator","denominator"},
                    "bandwidth_1":{"energy_1","numerator","denominator"},
                    "energy_0":{"denominator"},"energy_1":{"denominator"}}
    def __init__(self,rule):
        if not isinstance(rule,Rule) or tuple(rule.names)!=tuple(FEATURES):raise ValueError("Raw execution requires a Rule over the discovery feature schema")
        self.program=rule;self.nodes=discovery_feature_nodes();self.catalog={};self._nodes={};physical=[];virtual=[];dep_keys={}
        def visit(node,path):
            kind="vector" if node.kind=="scalar" else "matrix" if node.kind=="vector" else node.kind
            self.catalog[path]={"kind":kind,"semantic_kind":node.kind,"op":node.op};self._nodes[path]=node;physical.append(path)
            if node.op=="dep":
                descriptors={
                    "bandwidth_0":{"internal":"dep_bandwidth","input":node.args[0].to_dict()},
                    "bandwidth_1":{"internal":"dep_bandwidth","input":node.args[1].to_dict()},
                    "energy_0":{"internal":"dep_energy","input":node.args[0].to_dict()},
                    "energy_1":{"internal":"dep_energy","input":node.args[1].to_dict()},
                    "numerator":{"internal":"dep_numerator","inputs":[a.to_dict() for a in node.args]},
                    "denominator":{"internal":"dep_denominator","inputs":[a.to_dict() for a in node.args]}}
                for role in self._DEP_ROLES:
                    p=path+"/internal/"+role;meta={"kind":"vector","semantic_kind":"scalar","op":"dep_"+role,"dep_parent":path,"dep_role":role}
                    self.catalog[p]=meta;virtual.append(p);dep_keys[p]=_key(descriptors[role])
            for i,child in enumerate(node.args):visit(child,path+"/"+str(i))
        for i,node in enumerate(self.nodes):visit(node,f"feature/{i}")
        grouped={}
        for path in physical:
            node=self._nodes[path]
            if node.kind=="scalar" and node.op!="constant" and path.count("/")>1:grouped.setdefault(_key(node.to_dict()),[]).append(path)
        dep_grouped={}
        for path in virtual:dep_grouped.setdefault(dep_keys[path],[]).append(path)
        def make_groups(items,prefix):
            result={}
            for key,members in sorted(items.items()):
                path=prefix+hashlib.sha256(key.encode()).hexdigest()[:16];result[path]=tuple(members)
                self.catalog[path]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","members":list(members)}
            return result
        self.groups=make_groups(grouped,"group/");self.dep_groups=make_groups(dep_grouped,"dep_group/");self._all_groups={**self.groups,**self.dep_groups}
        self._physical=tuple(physical);self._virtual=tuple(virtual)
        payload={"features":[n.to_dict() for n in self.nodes],"rule":rule.to_dict(),"groups":self._all_groups}
        self.program_id=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()
    def address(self,path):
        if path not in self.catalog:raise ValueError("Unknown raw-program occurrence")
        return self.program_id+":"+path
    def intervention_paths(self,address_or_path):
        path=address_or_path.split(":",1)[1] if ":" in address_or_path else address_or_path
        if path not in self.catalog:raise ValueError("Unknown raw-program occurrence")
        return self._all_groups.get(path,(path,))
    def interventions_compatible(self,addresses):
        paths=[p for a in addresses for p in self.intervention_paths(a)]
        for i,a in enumerate(paths):
            for b in paths[i+1:]:
                ma,mb=self.catalog[a],self.catalog[b]
                if ma.get("dep_parent")==mb.get("dep_parent") and "dep_role" in ma and "dep_role" in mb:
                    if mb["dep_role"] in self._DEP_ANCESTORS.get(ma["dep_role"],set()) or ma["dep_role"] in self._DEP_ANCESTORS.get(mb["dep_role"],set()):return False
        return True
    def execute(self,data,interventions=None):
        data=np.asarray(data,dtype=float)
        if data.ndim!=3 or data.shape[2]!=2 or data.shape[1]<16 or not np.isfinite(data).all():raise ValueError("Expected finite worlds x samples x 2 raw datasets")
        worlds,samples,_=data.shape;requested={}
        for address,value in (interventions or {}).items():
            prefix,sep,path=address.partition(":")
            if not sep or prefix!=self.program_id or path not in self.catalog:raise ValueError("Foreign or unknown intervention")
            value=np.asarray(value);semantic=self.catalog[path]["semantic_kind"];shape=(worlds,) if semantic=="scalar" else (worlds,samples)
            if value.shape!=shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():raise ValueError("Raw intervention shape/type mismatch")
            requested[path]=value
        if not self.interventions_compatible(requested):raise ValueError("Causally overlapping dependence interventions")
        expanded={}
        for path,value in requested.items():
            for target in self._all_groups.get(path,(path,)):
                if target in expanded:raise ValueError("Overlapping interventions")
                expanded[target]=value
        for a in expanded:
            if any(b.startswith(a+"/") for b in expanded if a!=b):raise ValueError("Overlapping interventions")
        collected={path:[] for path in self._physical+self._virtual}
        def scalar(path,natural,index):
            value=float(expanded[path][index]) if path in expanded else float(natural);collected[path].append(np.asarray(value));return value
        def dep(values,path,index):
            kernels=[];energies=[]
            for side,v in enumerate(values):
                dist=(v[:,None]-v[None,:])**2;positive=dist[dist>1e-12];natural=np.median(positive) if len(positive) else 1.
                bandwidth=scalar(path+f"/internal/bandwidth_{side}",natural,index)
                k=np.exp(-dist/(2*max(abs(bandwidth),1e-8)));k=k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean();kernels.append(k)
                energies.append(scalar(path+f"/internal/energy_{side}",np.sum(k*k),index))
            numerator=scalar(path+"/internal/numerator",np.sum(kernels[0]*kernels[1]),index)
            denominator=scalar(path+"/internal/denominator",max(np.sqrt(max(energies[0],0)*max(energies[1],0)),1e-12),index)
            return numerator/max(abs(denominator),1e-12)
        def evaluate(node,path,dataset,index):
            if path in expanded:value=float(expanded[path][index]) if node.kind=="scalar" else expanded[path][index].astype(float,copy=True)
            elif node.op in ("var","constant"):value=node.evaluate(dataset)
            else:
                values=[evaluate(child,path+"/"+str(i),dataset,index) for i,child in enumerate(node.args)]
                value=dep(values,path,index) if node.op=="dep" else _apply(node,values,samples)
            collected[path].append(np.asarray(value).copy());return value
        features=np.empty((worlds,len(self.nodes)),float)
        for w,dataset in enumerate(data):
            for i,node in enumerate(self.nodes):features[w,i]=evaluate(node,f"feature/{i}",dataset,w)
        if not np.isfinite(features).all():raise ValueError("Non-finite raw feature result")
        records={self.address(path):{"kind":self.catalog[path]["kind"],"semantic_kind":self.catalog[path]["semantic_kind"],"op":self.catalog[path]["op"],
            "value":np.asarray(collected[path]),"visited":np.ones(worlds,bool),"intervened":path in expanded}
            for path in self._physical+self._virtual if len(collected[path])==worlds}
        for path,members in self._all_groups.items():
            if any(len(collected[p])!=worlds for p in members):
                if path in requested:raise RuntimeError("Group intervention made a member unreachable")
                continue
            values=[np.asarray(collected[p]) for p in members];agrees=all(np.allclose(values[0],v,rtol=1e-10,atol=1e-10) for v in values[1:])
            if path in requested and not agrees:raise RuntimeError("Group intervention failed to preserve structural equivalence")
            if agrees:records[self.address(path)]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","value":values[0],
                "visited":np.ones(worlds,bool),"intervened":path in requested,"members":list(members)}
        return RawExecutionTrace(self.program_id,self.program.predict(features),features,records)

def raw_scalar_frontier(executor,*,include_feature_roots=False):
    result=[]
    for path in executor._physical:
        meta=executor.catalog[path]
        if meta["kind"]!="vector" or meta["op"]=="constant":continue
        if not include_feature_roots and path.count("/")==1:continue
        result.append(executor.address(path))
    return result

def raw_scalar_groups(executor):return [executor.address(path) for path in sorted(executor.groups)]
def dependence_scalar_groups(executor):return [executor.address(path) for path in sorted(executor.dep_groups)]
''',encoding='utf-8')
