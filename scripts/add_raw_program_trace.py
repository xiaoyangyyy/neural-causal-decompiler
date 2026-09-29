from pathlib import Path
Path('ncd/raw_program_trace.py').write_text('''"""Occurrence-level execution from raw datasets through CDIR features to a Rule."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
from .cdir import discovery_feature_nodes
from .program_trace import _apply
from .rules import Rule
from .statistics import FEATURES

@dataclass
class RawExecutionTrace:
    program_id:str
    output:np.ndarray
    features:np.ndarray
    records:dict

    def to_dict(self):
        return {"program_id":self.program_id,"output":self.output.tolist(),"features":self.features.tolist(),
                "records":{k:{"kind":v["kind"],"semantic_kind":v["semantic_kind"],"op":v["op"],
                    "value":np.asarray(v["value"]).tolist(),"visited":np.asarray(v["visited"]).tolist(),
                    "intervened":v["intervened"]} for k,v in self.records.items()}}

class RawDiscoveryExecutor:
    """Execute the declared fourteen raw-data ASTs, then an extracted rule.

    Scalar-in-one-dataset nodes are exposed as per-world vectors, which makes them
    compatible with world-level interchange experiments. Sample vectors are traced
    as matrices and may be patched, but are excluded from scalar mapping utilities.
    """
    def __init__(self,rule):
        if not isinstance(rule,Rule) or tuple(rule.names)!=tuple(FEATURES):
            raise ValueError("Raw execution requires a Rule over the discovery feature schema")
        self.program=rule;self.nodes=discovery_feature_nodes()
        payload={"features":[n.to_dict() for n in self.nodes],"rule":rule.to_dict()}
        self.program_id=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.catalog={};self._nodes={}
        def visit(node,path):
            kind="vector" if node.kind=="scalar" else "matrix" if node.kind=="vector" else node.kind
            self.catalog[path]={"kind":kind,"semantic_kind":node.kind,"op":node.op};self._nodes[path]=node
            for i,child in enumerate(node.args):visit(child,path+"/"+str(i))
        for i,node in enumerate(self.nodes):visit(node,f"feature/{i}")
    def address(self,path):
        if path not in self.catalog:raise ValueError("Unknown raw-program occurrence")
        return self.program_id+":"+path
    def execute(self,data,interventions=None):
        data=np.asarray(data,dtype=float)
        if data.ndim!=3 or data.shape[2]!=2 or data.shape[1]<16 or not np.isfinite(data).all():
            raise ValueError("Expected finite worlds x samples x 2 raw datasets")
        worlds,samples,_=data.shape;patches={}
        for address,value in (interventions or {}).items():
            prefix,sep,path=address.partition(":")
            if not sep or prefix!=self.program_id or path not in self.catalog:raise ValueError("Foreign or unknown intervention")
            patches[path]=np.asarray(value)
        for a in patches:
            if any(b.startswith(a+"/") for b in patches if a!=b):raise ValueError("Overlapping interventions")
        for path,value in patches.items():
            shape=(worlds,) if self.catalog[path]["semantic_kind"]=="scalar" else (worlds,samples)
            if value.shape!=shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("Raw intervention shape/type mismatch")
        collected={path:[] for path in self.catalog}
        def evaluate(node,path,dataset,index):
            if path in patches:value=float(patches[path][index]) if node.kind=="scalar" else patches[path][index].astype(float,copy=True)
            elif node.op in ("var","constant"):value=node.evaluate(dataset)
            else:value=_apply(node,[evaluate(child,path+"/"+str(i),dataset,index) for i,child in enumerate(node.args)],samples)
            collected[path].append(np.asarray(value).copy());return value
        features=np.empty((worlds,len(self.nodes)),float)
        for w,dataset in enumerate(data):
            for i,node in enumerate(self.nodes):features[w,i]=evaluate(node,f"feature/{i}",dataset,w)
        if not np.isfinite(features).all():raise ValueError("Non-finite raw feature result")
        records={self.address(path):{"kind":meta["kind"],"semantic_kind":meta["semantic_kind"],"op":meta["op"],
            "value":np.asarray(collected[path]),"visited":np.ones(worlds,bool),"intervened":path in patches}
            for path,meta in self.catalog.items()}
        return RawExecutionTrace(self.program_id,self.program.predict(features),features,records)

def raw_scalar_frontier(executor,*,include_feature_roots=False):
    """Per-world scalar statistics; raw variables and sample vectors stay excluded."""
    result=[]
    for path,meta in executor.catalog.items():
        if meta["kind"]!="vector" or meta["op"]=="constant":continue
        if not include_feature_roots and path.count("/")==1:continue
        result.append(executor.address(path))
    return result
''',encoding='utf-8')
Path('tests/test_raw_program_trace.py').write_text('''import numpy as np
import pytest
from ncd.cdir import composed_features
from ncd.raw_program_trace import RawDiscoveryExecutor,raw_scalar_frontier
from ncd.rules import Rule
from ncd.statistics import FEATURES


def rule():
    return Rule(FEATURES,{"expr":{"op":"var","args":[],"value":0.,"index":2},"threshold":0.,
                          "left":{"label":0},"right":{"label":1}})

def worlds():
    rng=np.random.default_rng(81);a=rng.normal(size=(32,2));b=rng.normal(size=(32,2));
    a[:,0]*=.3;a[:,1]*=2.;b[:,0]*=3.;b[:,1]*=.4
    return np.stack([a,b])

def test_raw_execution_exactly_composes_declared_features_and_rule():
    data=worlds();e=RawDiscoveryExecutor(rule());trace=e.execute(data)
    np.testing.assert_allclose(trace.features,composed_features(data),rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(trace.output,rule().predict(trace.features))
    assert all(r['visited'].all() for r in trace.records.values())
    assert raw_scalar_frontier(e) and all(e.catalog[a.split(':',1)[1]]['semantic_kind']=='scalar' for a in raw_scalar_frontier(e))

def test_inner_variance_interchange_creates_hybrid_not_predicate_alias():
    data=worlds();e=RawDiscoveryExecutor(rule());base=e.execute(data)
    # feature 2 = log(var(x)/var(y)); patch only var(x), retaining base-world var(y).
    address=e.address('feature/2/0/0');source=base.records[address]['value'][::-1]
    patched=e.execute(data,{address:source})
    expected=np.log(source/base.records[e.address('feature/2/0/1')]['value'])
    np.testing.assert_allclose(patched.features[:,2],expected)
    assert not np.array_equal(patched.output,base.output[::-1])

def test_occurrence_identity_and_intervention_validation():
    data=worlds();e=RawDiscoveryExecutor(rule());same=RawDiscoveryExecutor(Rule.from_dict(rule().to_dict()))
    assert e.program_id==same.program_id
    address=e.address('feature/2/0/0')
    with pytest.raises(ValueError,match='shape'):e.execute(data,{address:np.ones((2,1))})
    with pytest.raises(ValueError,match='Foreign'):e.execute(data,{address.replace(e.program_id,'0'*64):np.ones(2)})
    with pytest.raises(ValueError,match='Overlapping'):e.execute(data,{address:np.ones(2),e.address('feature/2/0'):np.ones(2)})

def test_node_permutation_is_not_silently_accepted_as_feature_schema():
    bad=Rule(tuple(reversed(FEATURES)),rule().tree)
    with pytest.raises(ValueError,match='schema'):RawDiscoveryExecutor(bad)
''',encoding='utf-8')
