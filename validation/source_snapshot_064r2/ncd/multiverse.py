"""Serializable nonlinear multivariate SCMs and surgical interventions."""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import hashlib
import json
import numpy as np
from .graphs import random_dag,topological_order,cpdag
from .worlds import noise
from .io import save_json,read_json

@dataclass(frozen=True)
class Term:
    operator: str
    parents: tuple
    coefficient: float

    def __post_init__(self):
        arity={"linear":1,"square":1,"sin":1,"cos":1,"tanh":1,"interaction":2}
        if self.operator not in arity or len(self.parents)!=arity[self.operator]:
            raise ValueError("Invalid mechanism term arity")
        if len(set(self.parents))!=len(self.parents) or any(not isinstance(i,int) or i<0 for i in self.parents):
            raise ValueError("Invalid mechanism parents")
        if not np.isfinite(self.coefficient) or self.coefficient==0:
            raise ValueError("Structural term coefficient must be finite and nonzero")

    def evaluate(self,data):
        if self.operator=="linear":value=data[:,self.parents[0]]
        elif self.operator=="square":value=data[:,self.parents[0]]**2
        elif self.operator=="sin":value=np.sin(data[:,self.parents[0]])
        elif self.operator=="tanh":value=np.tanh(data[:,self.parents[0]])
        elif self.operator=="cos":value=np.cos(data[:,self.parents[0]])
        elif self.operator=="interaction":value=data[:,self.parents[0]]*data[:,self.parents[1]]
        else:raise ValueError("Unsupported mechanism operator")
        return self.coefficient*value

    def ast(self):
        variables=[{"op":"var","index":int(p)} for p in self.parents]
        inner=variables[0] if self.operator=="linear" else {"op":"mul" if self.operator=="interaction" else self.operator,"args":variables}
        return {"op":"mul","args":[{"op":"constant","value":self.coefficient},inner]}

@dataclass(frozen=True)
class GraphWorld:
    seed: int
    split: str
    graph: tuple
    equations: tuple
    noise_family: str="laplace"
    noise_scale: float=.35
    samples: int=128
    scales: tuple=()
    family: str="nonlinear"
    root_shift: bool=False

    def __post_init__(self):
        a=np.asarray(self.graph,dtype=bool);topological_order(a)
        if self.family=="linear_gaussian" and (self.noise_family!="gaussian" or any(t.operator!="linear" for ts in self.equations for t in ts)):
            raise ValueError("Linear-Gaussian label contradicts mechanisms")
        if len(self.equations)!=len(a) or self.samples<16 or self.noise_scale<=0:
            raise ValueError("Invalid SCM dimensions")
        if self.scales and (len(self.scales)!=len(a) or min(self.scales)<=0):raise ValueError("Invalid scales")
        for child,terms in enumerate(self.equations):
            declared=set(np.flatnonzero(a[:,child]))
            used=set(p for t in terms for p in t.parents)
            if declared!=used:raise ValueError("Equation parents differ from graph")

    @property
    def nodes(self):return len(self.graph)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:24]

    @property
    def target_graph(self):
        return cpdag(self.graph) if self.family=="linear_gaussian" else np.asarray(self.graph,dtype=bool)

    def sample(self,*,interventions=None,seed=None,samples=None,exogenous=None,return_exogenous=False):
        n=self.samples if samples is None else samples
        interventions={} if interventions is None else interventions
        if any(not isinstance(k,(int,np.integer)) or not 0<=k<self.nodes for k in interventions):
            raise ValueError("Unknown intervention target")
        rng=np.random.default_rng(self.seed if seed is None else seed)
        if exogenous is None:
            u=np.column_stack([noise(rng,self.noise_family,n)*self.noise_scale for _ in range(self.nodes)])
            if self.root_shift:
                for j in np.flatnonzero(np.asarray(self.graph).sum(0)==0):u[:,j]=rng.normal(.5,.8,n)
        else:
            u=np.asarray(exogenous,dtype=float)
            if u.shape!=(n,self.nodes):raise ValueError("Invalid exogenous array")
        x=np.zeros((n,self.nodes))
        scales=np.array(self.scales or (1.,)*self.nodes)
        for j in topological_order(self.graph):
            if j in interventions:
                # User specifies observed-coordinate do value.
                x[:,j]=np.asarray(interventions[j])/scales[j]
            else:
                x[:,j]=sum((t.evaluate(x) for t in self.equations[j]),start=np.zeros(n))+u[:,j]
        observed=x*scales
        if not np.isfinite(observed).all():raise ValueError("Non-finite SCM output")
        return (observed,u) if return_exogenous else observed

    def intervention_graph(self,targets):
        a=np.array(self.graph,dtype=bool);a[:,list(targets)]=False
        return a

    def metadata(self):
        return {**asdict(self),"world_id":self.identity,"equation_ast":[
            {"op":"add","args":[t.ast() for t in terms]+[{"op":"noise","index":j,"family":self.noise_family,
             "scale":self.noise_scale}]} for j,terms in enumerate(self.equations)],
            "target_graph":self.target_graph.astype(int).tolist(),
            "target_semantics":"CPDAG" if self.family=="linear_gaussian" else "DAG_under_benchmark_assumptions",
            "observed_scaling":list(self.scales or (1.,)*self.nodes)}

    @classmethod
    def from_dict(cls,d):
        fields={k:v for k,v in d.items() if k in cls.__dataclass_fields__}
        fields["graph"]=tuple(tuple(row) for row in fields["graph"])
        fields["equations"]=tuple(tuple(Term(t["operator"],tuple(t["parents"]),t["coefficient"]) for t in ts) for ts in fields["equations"])
        fields["scales"]=tuple(fields.get("scales",()))
        return cls(**fields)

def generate_graph_worlds(split,count,nodes,seed=42,samples=128):
    namespace=int.from_bytes(hashlib.sha256(split.encode()).digest()[:4],"little")
    rng=np.random.default_rng(np.random.SeedSequence([seed,namespace,nodes]))
    worlds=[]
    for i in range(count):
        a=random_dag(nodes,rng,1.5)
        gaussian=i%5==0
        family="linear_gaussian" if gaussian else "nonlinear"
        equations=[]
        for j in range(nodes):
            parents=list(np.flatnonzero(a[:,j]));terms=[]
            for p in parents:
                operators=["linear"] if gaussian else ["linear","sin","tanh","square"]
                if split=="test_function" and not gaussian:operators=["cos","square"]
                op=str(rng.choice(operators))
                coef=float(rng.choice([-1,1])*rng.uniform(.4,1.1))
                if op=="square":coef*=.3
                terms.append(Term(op,(int(p),),coef))
            if not gaussian and len(parents)>=2 and rng.random()<.5:
                terms.append(Term("interaction",tuple(map(int,parents[:2])),float(rng.uniform(.2,.5))))
            equations.append(tuple(terms))
        noise_family="gaussian" if gaussian else ("student" if split=="test_noise" else "laplace")
        scales=tuple(map(float,np.exp(rng.uniform(-1.5,1.5,nodes)))) if split=="test_scale" else (1.,)*nodes
        worlds.append(GraphWorld(int(rng.integers(0,2**63-1)),split,tuple(tuple(int(v) for v in row) for row in a),
            tuple(equations),noise_family,.35,samples,scales,family,split=="test_intervention"))
    return worlds

def save_graph_dataset(directory,worlds):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    data=np.stack([w.sample() for w in worlds])
    intervened=np.stack([w.sample(interventions={0:1.}) for w in worlds])
    np.savez_compressed(directory/"samples.npz",data=data,hard_intervention=intervened,
                        target_graph=np.stack([w.target_graph for w in worlds]))
    save_json(directory/"worlds.json",[w.metadata() for w in worlds])

def load_graph_worlds(directory):
    return [GraphWorld.from_dict(d) for d in read_json(Path(directory)/"worlds.json")]
