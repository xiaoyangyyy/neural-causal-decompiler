"""Schema-independent executable rules with compositional arithmetic splits."""
from dataclasses import dataclass
import json
import numpy as np
from .cdir import Node

@dataclass(frozen=True)
class Rule:
    names:tuple
    tree:dict

    @property
    def complexity(self):
        def count(t):
            if "label" in t:return 1
            return 2+Node.from_dict(t["expr"]).complexity+count(t["left"])+count(t["right"])
        return count(self.tree)

    def predict(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.names) or not np.isfinite(x).all():raise ValueError("Feature schema mismatch")
        def walk(t):
            if "label" in t:return np.full(len(x),t["label"],dtype=int)
            e=Node.from_dict(t["expr"]).evaluate(x)
            return np.where(e<t["threshold"],walk(t["left"]),walk(t["right"]))
        return walk(self.tree)

    def to_dict(self):return {"names":list(self.names),"tree":self.tree,"complexity":self.complexity}

    @classmethod
    def from_dict(cls,d):
        def check(t,depth=0):
            if depth>20:raise ValueError("Rule too deep")
            if "label" in t:
                if not isinstance(t["label"],int) or t["label"] not in range(4):raise ValueError("Invalid class")
            else:
                e=Node.from_dict(t["expr"])
                if not np.isfinite(t["threshold"]):raise ValueError("Invalid threshold")
                check(t["left"],depth+1);check(t["right"],depth+1)
        check(d["tree"])
        rule=cls(tuple(d["names"]),d["tree"])
        if "complexity" in d and d["complexity"]!=rule.complexity:raise ValueError("Complexity mismatch")
        return rule

    def text(self):
        def expression(e):
            if e.op=="var":return self.names[e.index]
            if e.op=="constant":return str(e.value)
            if e.op in ("abs","sin","cos","tanh","square"):return f"{e.op}({expression(e.args[0])})"
            return f"({expression(e.args[0])} {dict(add='+',sub='-',mul='*',div='/')[e.op]} {expression(e.args[1])})"
        def walk(t,level=0):
            space="    "*level
            if "label" in t:return space+f"return {t['label']}"
            return space+f"if {expression(Node.from_dict(t['expr']))} < {t['threshold']:.7g}:\n"+walk(t["left"],level+1)+"\n"+space+"else:\n"+walk(t["right"],level+1)
        return walk(self.tree)

def candidate_expressions(names,arithmetic=True):
    base=[Node("var",index=i) for i in range(len(names))]
    result=list(base)
    if arithmetic:
        result.extend(Node("abs",(n,)) for n in base)
        # Enumerate primitive compositions without hard-coded causal-feature pairs.
        for i,a in enumerate(base):
            result.append(Node("mul",(a,a)))
            for b in base[i+1:]:
                result.extend(Node(op,(a,b)) for op in ("add","sub","mul","div"))
                result.append(Node("div",(b,a)))
                result.append(Node("abs",(Node("sub",(a,b)),)))
    return result


def fit_rule(x,labels,names,*,penalty=.001,beam_width=3,max_splits=6,min_leaf=8,arithmetic=True,
             feature_alignment=None,alignment_weight=.0,sample_weight=None):
    x=np.asarray(x,dtype=float);labels=np.asarray(labels,dtype=int)
    if x.ndim!=2 or x.shape!=(len(labels),len(names)) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid rule data")
    if not np.isin(labels,range(4)).all():raise ValueError("Invalid class labels")
    # Weights alter empirical fidelity, including expression pruning and leaf votes.
    # Threshold quantiles and min_leaf retain their original sample-count semantics.
    weights=np.ones(len(x),dtype=float) if sample_weight is None else np.asarray(sample_weight,dtype=float)
    if weights.shape!=(len(x),) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("sample_weight must contain one finite positive value per row")
    total=float(weights.sum())
    if not np.isfinite(total):raise ValueError("Non-finite total sample weight")
    if feature_alignment is not None:
        feature_alignment=np.asarray(feature_alignment,dtype=float)
        if feature_alignment.shape!=(len(names),) or not np.isfinite(feature_alignment).all() or np.any((feature_alignment<0)|(feature_alignment>1)):
            raise ValueError("feature_alignment must contain one finite [0,1] support per feature")
    def counts(ids):return np.bincount(labels[ids],weights=weights[ids],minlength=4)
    def fidelity(pred):return float(weights[pred==labels].sum()/total)
    expressions=candidate_expressions(names,arithmetic)
    values=np.column_stack([e.evaluate(x) for e in expressions])
    # Bounded expression beam ranked only against this synthesis teacher.
    if len(expressions)>96:
        ranking=[]
        for i,e in enumerate(expressions):
            best_correct=0
            for t in np.unique(np.r_[0.,np.quantile(values[:,i],np.linspace(.05,.95,9))]):
                left=values[:,i]<t
                lc=counts(left);rc=counts(~left)
                best_correct=max(best_correct,float(lc.max()+rc.max()))
            ranking.append((best_correct/total-penalty*e.complexity,i))
        retained=list(range(len(names)))
        for _,i in sorted(ranking,reverse=True):
            if i not in retained:retained.append(i)
            if len(retained)>=96:break
        expressions=[expressions[i] for i in retained];values=values[:,retained]

    def alignment_cost(e):
        if feature_alignment is None:return 0.
        if e.op=="var":return 1-float(feature_alignment[e.index])
        return sum(alignment_cost(a) for a in e.args)
    predicates=[]
    for i,e in enumerate(expressions):
        for t in np.unique(np.r_[0.,np.quantile(values[:,i],np.linspace(.03,.97,17))]):
            predicates.append((e,float(t),values[:,i]<t,alignment_cost(e)))
    initial={"label":int(counts(slice(None)).argmax())}
    def leaf_paths(t,path=()):
        if "label" in t:return [(path,t)]
        return leaf_paths(t["left"],path+("left",))+leaf_paths(t["right"],path+("right",))
    def replace(t,path,sub):
        if not path:return sub
        return {**t,path[0]:replace(t[path[0]],path[1:],sub)}
    def mask(t,path):
        m=np.ones(len(x),bool)
        for side in path:
            decision=Node.from_dict(t["expr"]).evaluate(x)<t["threshold"]
            m&=decision if side=="left" else ~decision;t=t[side]
        return m
    def score(t):
        p=Rule(tuple(names),t)
        def cost(node):
            if "label" in node:return 0.
            return alignment_cost(Node.from_dict(node["expr"]))+cost(node["left"])+cost(node["right"])
        return fidelity(p.predict(x))-penalty*p.complexity-alignment_weight*cost(t)
    best=initial;beam=[initial];trace=[]
    for depth in range(max_splits):
        proposals=[]
        for tree in beam:
            for path,leaf in leaf_paths(tree):
                ids=np.flatnonzero(mask(tree,path));old=weights[ids][labels[ids]==leaf["label"]].sum()
                local=[]
                for e,t,decision,align_cost in predicates:
                    l=ids[decision[ids]];r=ids[~decision[ids]]
                    if min(len(l),len(r))<min_leaf:continue
                    lc,rc=counts(l),counts(r)
                    ll,rl=int(lc.argmax()),int(rc.argmax())
                    if ll==rl:continue
                    gain=(float(lc[ll])+float(rc[rl])-old)/total-penalty*(3+e.complexity)-alignment_weight*align_cost
                    local.append((gain,e,t,ll,rl))
                for gain,e,t,ll,rl in sorted(local,key=lambda z:-z[0])[:beam_width]:
                    sub={"expr":e.to_dict(),"threshold":t,"left":{"label":ll},"right":{"label":rl}}
                    p=replace(tree,path,sub);proposals.append((score(p),p))
        if not proposals:break
        seen=set();beam=[]
        for s,p in sorted(proposals,key=lambda z:-z[0]):
            key=json.dumps(p,sort_keys=True)
            if key not in seen:beam.append(p);seen.add(key)
            if len(beam)>=beam_width:break
        if score(beam[0])>score(best)+1e-12:best=beam[0]
        trace.append({"splits":depth+1,"score":score(beam[0]),"fidelity":float(np.mean(Rule(tuple(names),beam[0]).predict(x)==labels)),
                      "complexity":Rule(tuple(names),beam[0]).complexity})
        if sample_weight is not None:
            trace[-1]["weighted_fidelity"]=fidelity(Rule(tuple(names),beam[0]).predict(x))
    return Rule(tuple(names),best),trace
