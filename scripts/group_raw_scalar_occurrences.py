from pathlib import Path
Path('ncd/raw_program_trace.py').write_text('''"""Occurrence-level execution from raw datasets through CDIR features to a Rule."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
from .cdir import discovery_feature_nodes
from .program_trace import _apply
from .rules import Rule
from .statistics import FEATURES

def _node_key(node):return json.dumps(node.to_dict(),sort_keys=True,separators=(",",":"))

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
    """Execute the fourteen raw-data ASTs and then an extracted Rule."""
    def __init__(self,rule):
        if not isinstance(rule,Rule) or tuple(rule.names)!=tuple(FEATURES):
            raise ValueError("Raw execution requires a Rule over the discovery feature schema")
        self.program=rule;self.nodes=discovery_feature_nodes();self.catalog={};self._nodes={};physical=[]
        def visit(node,path):
            kind="vector" if node.kind=="scalar" else "matrix" if node.kind=="vector" else node.kind
            self.catalog[path]={"kind":kind,"semantic_kind":node.kind,"op":node.op};self._nodes[path]=node;physical.append(path)
            for i,child in enumerate(node.args):visit(child,path+"/"+str(i))
        for i,node in enumerate(self.nodes):visit(node,f"feature/{i}")
        grouped={}
        for path in physical:
            node=self._nodes[path]
            if node.kind=="scalar" and node.op!="constant" and path.count("/")>1:grouped.setdefault(_node_key(node),[]).append(path)
        self.groups={}
        for key,members in sorted(grouped.items()):
            token=hashlib.sha256(key.encode()).hexdigest()[:16];path="group/"+token;self.groups[path]=tuple(members)
            self.catalog[path]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","members":list(members)}
        self._physical=tuple(physical)
        payload={"features":[n.to_dict() for n in self.nodes],"rule":rule.to_dict(),"groups":self.groups}
        self.program_id=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()
    def address(self,path):
        if path not in self.catalog:raise ValueError("Unknown raw-program occurrence")
        return self.program_id+":"+path
    def execute(self,data,interventions=None):
        data=np.asarray(data,dtype=float)
        if data.ndim!=3 or data.shape[2]!=2 or data.shape[1]<16 or not np.isfinite(data).all():
            raise ValueError("Expected finite worlds x samples x 2 raw datasets")
        worlds,samples,_=data.shape;requested={}
        for address,value in (interventions or {}).items():
            prefix,sep,path=address.partition(":")
            if not sep or prefix!=self.program_id or path not in self.catalog:raise ValueError("Foreign or unknown intervention")
            value=np.asarray(value);semantic=self.catalog[path]["semantic_kind"]
            shape=(worlds,) if semantic=="scalar" else (worlds,samples)
            if value.shape!=shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():raise ValueError("Raw intervention shape/type mismatch")
            requested[path]=value
        expanded={}
        for path,value in requested.items():
            targets=self.groups.get(path,(path,))
            for target in targets:
                if target in expanded:raise ValueError("Overlapping interventions")
                expanded[target]=value
        for a in expanded:
            if any(b.startswith(a+"/") for b in expanded if a!=b):raise ValueError("Overlapping interventions")
        collected={path:[] for path in self._physical}
        def evaluate(node,path,dataset,index):
            if path in expanded:value=float(expanded[path][index]) if node.kind=="scalar" else expanded[path][index].astype(float,copy=True)
            elif node.op in ("var","constant"):value=node.evaluate(dataset)
            else:value=_apply(node,[evaluate(child,path+"/"+str(i),dataset,index) for i,child in enumerate(node.args)],samples)
            collected[path].append(np.asarray(value).copy());return value
        features=np.empty((worlds,len(self.nodes)),float)
        for w,dataset in enumerate(data):
            for i,node in enumerate(self.nodes):features[w,i]=evaluate(node,f"feature/{i}",dataset,w)
        if not np.isfinite(features).all():raise ValueError("Non-finite raw feature result")
        records={self.address(path):{"kind":self.catalog[path]["kind"],"semantic_kind":self.catalog[path]["semantic_kind"],
            "op":self.catalog[path]["op"],"value":np.asarray(collected[path]),"visited":np.ones(worlds,bool),
            "intervened":path in expanded} for path in self._physical}
        for path,members in self.groups.items():
            values=[np.asarray(collected[p]) for p in members]
            if any(not np.allclose(values[0],v,rtol=1e-10,atol=1e-10) for v in values[1:]):raise RuntimeError("Structural equivalence group disagrees")
            records[self.address(path)]={"kind":"vector","semantic_kind":"scalar","op":"equivalence_group","value":values[0],
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

def raw_scalar_groups(executor):
    return [executor.address(path) for path in sorted(executor.groups)]
''',encoding='utf-8')
p=Path('tests/test_raw_program_trace.py');s=p.read_text(encoding='utf-8');s=s.replace('from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier','from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier,raw_scalar_groups')
s=s.replace("    assert raw_scalar_frontier(e) and all(e.catalog[a.split(':',1)[1]]['semantic_kind']=='scalar' for a in raw_scalar_frontier(e))","    assert raw_scalar_frontier(e) and raw_scalar_groups(e)\n    assert all(e.catalog[a.split(':',1)[1]]['semantic_kind']=='scalar' for a in raw_scalar_frontier(e)+raw_scalar_groups(e))")
s += '''\n\ndef test_structural_equivalence_group_patches_all_occurrences():
    data=worlds();e=RawDiscoveryExecutor(rule());groups=raw_scalar_groups(e)
    address=next(a for a in groups if len(e.catalog[a.split(':',1)[1]]['members'])>1)
    base=e.execute(data);source=base.records[address]['value'][::-1]
    patched=e.execute(data,{address:source});record=patched.records[address]
    assert record['intervened'] and len(record['members'])>1
    for path in record['members']:
        np.testing.assert_allclose(patched.records[e.address(path)]['value'],source)
        assert patched.records[e.address(path)]['intervened']
'''
p.write_text(s,encoding='utf-8')
