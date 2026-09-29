"""Composable typed causal-discovery IR and arithmetic mechanism expressions.

ASTs are executable without eval. Statistics operate on sample vectors; comparisons
on scalars. REGRESS and RESIDUAL are independent operations, not a direction oracle.
"""
from dataclasses import dataclass
import numpy as np
from .statistics import dependence

@dataclass(frozen=True)
class Node:
    op:str
    args:tuple=()
    value:float=0.
    index:int=0

    @property
    def kind(self):
        if self.op=="var":return "vector"
        if self.op=="constant":return "scalar"
        if self.op=="columns":return "matrix"
        if self.op in ("regress","residual"):return "vector"
        if self.op in ("mean","var_stat","skew","kurt","cov","dep","mi","partial"):return "scalar"
        if self.op in ("lt","gt","and","or","not"):return "bool"
        if self.op=="if":return self.args[1].kind
        return "vector" if any(a.kind=="vector" for a in self.args) else "scalar"

    def __post_init__(self):
        arities={"var":0,"constant":0,"add":2,"sub":2,"mul":2,"div":2,"abs":1,"sin":1,"cos":1,
                 "tanh":1,"square":1,"sqrt":1,"log":1,"mean":1,"var_stat":1,"skew":1,"kurt":1,"cov":2,"dep":2,
                 "mi":2,"regress":2,"residual":2,"partial":3,"lt":2,"gt":2,"and":2,"or":2,"not":1,"if":3}
        if self.op=="columns":
            if not self.args or any(a.kind!="vector" for a in self.args):raise ValueError("columns needs vectors")
            return
        if self.op not in arities or len(self.args)!=arities[self.op]:raise ValueError("Invalid IR arity/operator")
        if any(not isinstance(a,Node) for a in self.args):raise ValueError("IR child must be Node")
        if self.op=="constant" and not np.isfinite(self.value):raise ValueError("Non-finite constant")
        if self.op=="var" and (not isinstance(self.index,int) or self.index<0):raise ValueError("Invalid variable index")
        if self.op in ("mean","var_stat","skew","kurt") and self.args[0].kind!="vector":raise ValueError("Statistic needs vector")
        if self.op in ("dep","mi","cov","residual") and any(a.kind!="vector" for a in self.args):raise ValueError("Vector operands required")
        if self.op=="partial" and (self.args[0].kind!="vector" or self.args[1].kind!="vector" or self.args[2].kind not in ("vector","matrix")):
            raise ValueError("partial needs two vectors and conditioning predictors")
        if self.op=="regress" and (self.args[0].kind!="vector" or self.args[1].kind not in ("vector","matrix")):
            raise ValueError("regress(response,predictors)")
        if self.op in ("lt","gt") and any(a.kind!="scalar" for a in self.args):raise ValueError("Scalar comparison required")
        if self.op in ("and","or","not") and any(a.kind!="bool" for a in self.args):raise ValueError("Boolean operands required")
        if self.op=="if" and (self.args[0].kind!="bool" or self.args[1].kind!=self.args[2].kind):raise ValueError("Invalid conditional")
        if self.op in ("add","sub","mul","div","abs","sin","cos","tanh","square","sqrt","log") and any(a.kind not in ("scalar","vector") for a in self.args):
            raise ValueError("Numeric operands required")

    @property
    def complexity(self):return 1+sum(a.complexity for a in self.args)

    def to_dict(self):
        d={"op":self.op}
        if self.args:d["args"]=[a.to_dict() for a in self.args]
        if self.op=="constant":d["value"]=self.value
        if self.op=="var":d["index"]=self.index
        return d

    @classmethod
    def from_dict(cls,d,depth=0):
        if depth>24:raise ValueError("IR too deep")
        return cls(d["op"],tuple(cls.from_dict(a,depth+1) for a in d.get("args",())),float(d.get("value",0)),int(d.get("index",0)))

    def evaluate(self,data,cache=None):
        cache={} if cache is None else cache
        if self in cache:return cache[self]
        if self.op=="var":
            if self.index>=data.shape[1]:raise ValueError("Variable out of bounds")
            out=np.asarray(data[:,self.index],dtype=float)
        elif self.op=="constant":out=self.value
        else:
            a=[x.evaluate(data,cache) for x in self.args];op=self.op
            if op=="columns":out=np.column_stack(a)
            elif op=="add":out=a[0]+a[1]
            elif op=="sub":out=a[0]-a[1]
            elif op=="mul":out=a[0]*a[1]
            elif op=="div":
                denominator=np.asarray(a[1])
                out=a[0]/np.where(np.abs(denominator)<1e-8,np.where(denominator<0,-1e-8,1e-8),denominator)
            elif op=="abs":out=np.abs(a[0])
            elif op=="sin":out=np.sin(a[0])
            elif op=="cos":out=np.cos(a[0])
            elif op=="tanh":out=np.tanh(a[0])
            elif op=="square":out=a[0]**2
            elif op=="sqrt":out=np.sqrt(np.maximum(a[0],0))
            elif op=="log":out=np.log(np.maximum(np.abs(a[0]),1e-12))
            elif op=="mean":out=float(np.mean(a[0]))
            elif op=="var_stat":out=float(np.var(a[0]))
            elif op in ("skew","kurt"):
                z=(a[0]-np.mean(a[0]))/max(float(np.std(a[0])),1e-8)
                out=float(np.mean(z**(3 if op=="skew" else 4))-(3 if op=="kurt" else 0))
            elif op=="cov":out=float(np.mean((a[0]-np.mean(a[0]))*(a[1]-np.mean(a[1]))))
            elif op=="dep":out=dependence(a[0],a[1])
            elif op=="mi":
                h,_,_=np.histogram2d(a[0],a[1],bins=max(3,int(np.sqrt(len(a[0])/4))))
                p=h/max(h.sum(),1);q=p.sum(1)[:,None]*p.sum(0)[None,:];nz=p>0
                out=float(np.sum(p[nz]*np.log(p[nz]/q[nz])))
            elif op=="regress":out=crossfit_prediction(a[0],a[1])
            elif op=="residual":out=a[0]-a[1]
            elif op=="partial":
                out=dependence(a[0]-crossfit_prediction(a[0],a[2]),a[1]-crossfit_prediction(a[1],a[2]))
            elif op=="lt":out=bool(a[0]<a[1])
            elif op=="gt":out=bool(a[0]>a[1])
            elif op=="and":out=a[0] and a[1]
            elif op=="or":out=a[0] or a[1]
            elif op=="not":out=not a[0]
            elif op=="if":out=a[1] if a[0] else a[2]
            else:raise ValueError("Unknown operation")
        if not np.isfinite(out).all():raise ValueError("Non-finite IR result")
        cache[self]=out
        return out

def crossfit_prediction(y,x):
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x[:,None]
    if len(x)<16:raise ValueError("Cross-fit requires >=16 rows")
    order=np.lexsort(tuple(x[:,i] for i in reversed(range(x.shape[1]))))
    prediction=np.empty(len(y))
    def basis(z):
        return np.column_stack([np.ones(len(z)),z,z*z,np.sin(z),np.tanh(z),
            *[z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i)]])
    for side in (0,1):
        hold,fit=order[side::2],order[1-side::2]
        mean,std=x[fit].mean(0),x[fit].std(0).clip(1e-6)
        a=basis(np.clip((x[fit]-mean)/std,-10,10));b=basis(np.clip((x[hold]-mean)/std,-10,10))
        penalty=np.eye(a.shape[1])*.01;penalty[0,0]=1e-8
        prediction[hold]=b@np.linalg.solve(a.T@a+penalty,a.T@y[fit])
    return prediction

def statistical_primitives(x=0,y=1,condition=()):
    a,b=Node("var",index=x),Node("var",index=y)
    result={}
    for name,node in (("x",a),("y",b)):
        for op in ("mean","var_stat","skew","kurt"):result[f"{op}_{name}"]=Node(op,(node,))
    for op in ("cov","dep","mi"):result[op]=Node(op,(a,b))
    # Composed AST, with no hidden direction-test operator.
    for label,cause,effect in (("xy",a,b),("yx",b,a)):
        predicted=Node("regress",(effect,cause))
        residual=Node("residual",(effect,predicted))
        result["resdep_"+label]=Node("dep",(cause,residual))
    if condition:
        z=Node("columns",tuple(Node("var",index=int(i)) for i in condition))
        result["conditional_dep"]=Node("partial",(a,b,z))
    return result

def numeric_equivalence(a,b,data,atol=1e-7,rtol=1e-6):
    x,y=a.evaluate(data),b.evaluate(data)
    return {"equivalent_on_probes":bool(np.allclose(x,y,atol=atol,rtol=rtol)),
            "maximum_error":float(np.max(np.abs(x-y))),"proof":False}

def to_sympy(node):
    import sympy as sp
    if node.op=="var":return sp.Symbol(f"x{node.index}",real=True)
    if node.op=="constant":return sp.Rational(*float(node.value).as_integer_ratio())
    a=[to_sympy(x) for x in node.args]
    operations={"add":lambda:a[0]+a[1],"sub":lambda:a[0]-a[1],"mul":lambda:a[0]*a[1],
                "square":lambda:a[0]**2,"sin":lambda:sp.sin(a[0]),"cos":lambda:sp.cos(a[0]),
                "tanh":lambda:sp.tanh(a[0]),"abs":lambda:sp.Abs(a[0])}
    # Protected division is not algebraically equivalent to ordinary division.
    if node.op not in operations:raise ValueError("Node outside exact algebraic subset")
    return operations[node.op]()

def canonical_expression(node):
    import sympy as sp
    return str(sp.factor(sp.expand(to_sympy(node))))

def algebraic_equivalence(a,b):
    import sympy as sp
    return bool(sp.simplify(to_sympy(a)-to_sympy(b))==0)

def discovery_feature_nodes():
    """Full raw-dataset ASTs matching the declared fourteen-feature schema."""
    x,y=Node("var",index=0),Node("var",index=1)
    vx,vy=Node("var_stat",(x,)),Node("var_stat",(y,))
    zx=Node("div",(Node("sub",(x,Node("mean",(x,)))),Node("sqrt",(vx,))))
    zy=Node("div",(Node("sub",(y,Node("mean",(y,)))),Node("sqrt",(vy,))))
    corr=Node("mean",(Node("mul",(zx,zy)),))
    ry=Node("residual",(zy,Node("regress",(zy,zx))))
    rx=Node("residual",(zx,Node("regress",(zx,zy))))
    return (corr,Node("abs",(corr,)),Node("log",(Node("div",(vx,vy)),)),
        Node("skew",(x,)),Node("skew",(y,)),Node("kurt",(x,)),Node("kurt",(y,)),
        Node("dep",(x,y)),Node("dep",(zx,ry)),Node("dep",(zy,rx)),
        Node("mean",(Node("square",(ry,)),)),Node("mean",(Node("square",(rx,)),)),
        Node("mean",(Node("mul",(Node("square",(zx,)),zy)),)),
        Node("mean",(Node("mul",(Node("square",(zy,)),zx)),)))

def composed_features(data):
    nodes=discovery_feature_nodes();rows=[]
    for dataset in data:
        cache={};rows.append([node.evaluate(dataset,cache) for node in nodes])
    return np.asarray(rows,dtype=float)
