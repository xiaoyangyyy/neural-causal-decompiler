"""Occurrence-level execution from raw datasets through CDIR features to a Rule."""
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
    def __init__(self,rule,*,trace_dependence=False,trace_regression=False):
        if not isinstance(rule,Rule) or tuple(rule.names)!=tuple(FEATURES):raise ValueError("Raw execution requires a Rule over the discovery feature schema")
        self.program=rule;self.trace_dependence=bool(trace_dependence);self.trace_regression=bool(trace_regression);self.nodes=discovery_feature_nodes();self.catalog={};self._nodes={};physical=[];virtual=[];dep_keys={};reg_keys={}
        def visit(node,path):
            kind="vector" if node.kind=="scalar" else "matrix" if node.kind=="vector" else node.kind
            self.catalog[path]={"kind":kind,"semantic_kind":node.kind,"op":node.op};self._nodes[path]=node;physical.append(path)
            if self.trace_dependence and node.op=="dep":
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
            if self.trace_regression and node.op=="regress":
                predictors=node.args[1];dimensions=len(predictors.args) if predictors.op=="columns" else 1
                coefficients=1+4*dimensions+dimensions*(dimensions-1)//2
                descriptor={"response":node.args[0].to_dict(),"predictors":predictors.to_dict()}
                for fold in (0,1):
                    for role in [*(f"mean_{j}" for j in range(dimensions)),*(f"std_{j}" for j in range(dimensions)),*(f"beta_{j}" for j in range(coefficients))]:
                        p=f"{path}/internal/fold_{fold}/{role}";meta={"kind":"vector","semantic_kind":"scalar","op":"reg_"+role,
                            "reg_parent":path,"reg_fold":fold,"reg_role":role}
                        self.catalog[p]=meta;virtual.append(p);reg_keys[p]=_key({"internal":"regression","fold":fold,"role":role,**descriptor})
            for i,child in enumerate(node.args):visit(child,path+"/"+str(i))
        for i,node in enumerate(self.nodes):visit(node,f"feature/{i}")
        grouped={}
        for path in physical:
            node=self._nodes[path]
            if node.kind=="scalar" and node.op!="constant" and path.count("/")>1:grouped.setdefault(_key(node.to_dict()),[]).append(path)
        dep_grouped={}
        for path in virtual:
            if path in dep_keys:dep_grouped.setdefault(dep_keys[path],[]).append(path)
        def make_groups(items,prefix):
            result={}
            for key,members in sorted(items.items()):
                path=prefix+hashlib.sha256(key.encode()).hexdigest()[:16];result[path]=tuple(members)
                self.catalog[path]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","members":list(members)}
            return result
        reg_grouped={}
        for path in virtual:
            if path in reg_keys:reg_grouped.setdefault(reg_keys[path],[]).append(path)
        self.groups=make_groups(grouped,"group/");self.dep_groups=make_groups(dep_grouped,"dep_group/");self.reg_groups=make_groups(reg_grouped,"reg_group/")
        self._all_groups={**self.groups,**self.dep_groups,**self.reg_groups}
        self._physical=tuple(physical);self._virtual=tuple(virtual)
        payload={"features":[n.to_dict() for n in self.nodes],"rule":rule.to_dict(),"groups":self.groups}
        if self.trace_dependence:payload["dependence_groups"]=self.dep_groups
        if self.trace_regression:payload["regression_groups"]=self.reg_groups
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
                if ma.get("reg_parent")==mb.get("reg_parent") and ma.get("reg_fold")==mb.get("reg_fold") and "reg_role" in ma and "reg_role" in mb:
                    ra,rb=ma["reg_role"],mb["reg_role"]
                    if (ra.startswith(("mean_","std_")) and rb.startswith("beta_")) or (rb.startswith(("mean_","std_")) and ra.startswith("beta_")):return False
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
        def regress(values,path,index):
            y=np.asarray(values[0],dtype=float);x=np.asarray(values[1],dtype=float)
            if x.ndim==1:x=x[:,None]
            order=np.lexsort(tuple(x[:,i] for i in reversed(range(x.shape[1]))));prediction=np.empty(len(y))
            def basis(z):return np.column_stack([np.ones(len(z)),z,z*z,np.sin(z),np.tanh(z),
                *[z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i)]])
            for fold in (0,1):
                hold,fit=order[fold::2],order[1-fold::2];natural_mean=x[fit].mean(0);natural_std=x[fit].std(0).clip(1e-6)
                mean=np.array([scalar(path+f"/internal/fold_{fold}/mean_{j}",natural_mean[j],index) for j in range(x.shape[1])])
                std=np.array([max(abs(scalar(path+f"/internal/fold_{fold}/std_{j}",natural_std[j],index)),1e-6) for j in range(x.shape[1])])
                a=basis(np.clip((x[fit]-mean)/std,-10,10));b=basis(np.clip((x[hold]-mean)/std,-10,10))
                penalty=np.eye(a.shape[1])*.01;penalty[0,0]=1e-8;natural_beta=np.linalg.solve(a.T@a+penalty,a.T@y[fit])
                beta=np.array([scalar(path+f"/internal/fold_{fold}/beta_{j}",natural_beta[j],index) for j in range(len(natural_beta))])
                prediction[hold]=b@beta
            return prediction
        def evaluate(node,path,dataset,index):
            if path in expanded:value=float(expanded[path][index]) if node.kind=="scalar" else expanded[path][index].astype(float,copy=True)
            elif node.op in ("var","constant"):value=node.evaluate(dataset)
            else:
                values=[evaluate(child,path+"/"+str(i),dataset,index) for i,child in enumerate(node.args)]
                if self.trace_dependence and node.op=="dep":value=dep(values,path,index)
                elif self.trace_regression and node.op=="regress":value=regress(values,path,index)
                else:value=_apply(node,values,samples)
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

def feature_root_addresses(executor):
    """Addresses of the 14 fixed scalar discovery-feature outputs."""
    paths=[f"feature/{i}" for i in range(len(executor.nodes))]
    if any(executor.catalog[p]["kind"]!="vector" or executor.catalog[p]["semantic_kind"]!="scalar" for p in paths):
        raise ValueError("Discovery feature roots must be per-world scalars")
    return [executor.address(p) for p in paths]

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
def regression_scalar_groups(executor):return [executor.address(path) for path in sorted(executor.reg_groups)]
