"""Frozen neural conditional mechanisms -> explicit arithmetic SCM.

Symbolic fitting only receives neural predictions, never SCM equation metadata.
Graph errors and equation errors are evaluated separately by the experiment layer.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from .cdir import Node,canonical_expression
from .graphs import topological_order
from .statistics import dependence
from .model import set_seed
from .io import save_json,read_json

class NeuralMechanism(nn.Module):
    def __init__(self,parents,width=48):
        super().__init__();self.parents=tuple(parents);self.width=width
        p=max(len(parents),1)
        self.register_buffer("mean",torch.zeros(p))
        self.register_buffer("std",torch.ones(p))
        self.register_buffer("ymean",torch.tensor(0.))
        self.register_buffer("ystd",torch.tensor(1.))
        self.network=nn.Sequential(nn.Linear(p,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))

    def forward(self,full_data):
        x=full_data[:,list(self.parents)] if self.parents else torch.zeros((len(full_data),1))
        return self.network((x-self.mean)/self.std).squeeze(-1)*self.ystd+self.ymean

def train_mechanism(data,target,parents,*,epochs=180,width=48,seed=42):
    set_seed(seed);model=NeuralMechanism(parents,width)
    cut=int(.8*len(data));fit=torch.tensor(data[:cut],dtype=torch.float32);dev=torch.tensor(data[cut:],dtype=torch.float32)
    yfit=fit[:,target];ydev=dev[:,target]
    if parents:
        x=fit[:,list(parents)]
        model.mean.copy_(x.mean(0));model.std.copy_(x.std(0,unbiased=False).clamp_min(.05))
    model.ymean.copy_(yfit.mean());model.ystd.copy_(yfit.std(unbiased=False).clamp_min(.05))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.0005)
    best=float("inf");state=None;best_epoch=0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss=(((model(fit)-yfit)/model.ystd)**2).mean()
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval()
        with torch.no_grad():value=float(((model(dev)-ydev)**2).mean())
        if value<best:best=value;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(state);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,{"best_epoch":best_epoch,"dev_mse":best,"train_rows":cut,"dev_rows":len(dev)}

def neural_values(model,data):
    with torch.no_grad():return model(torch.as_tensor(data,dtype=torch.float32)).numpy().astype(float)

def mechanism_library(parents):
    terms=[("constant",Node("constant",value=1.))]
    for j in parents:
        x=Node("var",index=int(j));terms.append((f"linear:{j}",x))
        for op in ("square","sin","cos","tanh"):terms.append((f"{op}:{j}",Node(op,(x,))))
    for k,i in enumerate(parents):
        for j in parents[:k]:
            terms.append((f"interaction:{min(i,j)},{max(i,j)}",Node("mul",(Node("var",index=int(i)),Node("var",index=int(j))))))
    return terms

def sparse_symbolic_fit(inputs,teacher_values,parents,max_terms=8):
    library=mechanism_library(parents)
    matrix=np.column_stack([np.broadcast_to(node.evaluate(inputs),(len(inputs),)) for _,node in library])
    cut=int(.75*len(inputs))
    a,b=matrix[:cut],matrix[cut:];y,z=teacher_values[:cut],teacher_values[cut:]
    selected=[0];best=None;best_score=float("inf");history=[]
    scales=np.linalg.norm(a,axis=0).clip(1e-8)
    for step in range(min(max_terms,len(library))):
        coef=np.linalg.lstsq(a[:,selected],y,rcond=None)[0]
        mse=float(np.mean((b[:,selected]@coef-z)**2))
        # Validation neural fidelity with explicit sparsity penalty.
        score=np.log(mse/max(float(np.var(z)),1e-8)+1e-6)+.12*len(selected)
        history.append({"terms":list(selected),"validation_neural_mse":mse,"score":score})
        if score<best_score:best_score=score;best=(selected.copy(),coef.copy())
        residual=y-a[:,selected]@coef
        relevance=np.abs(a.T@residual)/scales
        relevance[selected]=-np.inf
        if not np.isfinite(relevance.max()):break
        selected.append(int(relevance.argmax()))
    selected,_=best
    # Refit only selected terms on all distillation queries, all targets neural.
    coef=np.linalg.lstsq(matrix[:,selected],teacher_values,rcond=None)[0]
    weighted=[];atoms=[]
    for idx,c in zip(selected,coef):
        if abs(c)<.005 and idx!=0:continue
        name,node=library[idx];atoms.append({"atom":name,"coefficient":float(c)})
        weighted.append(Node("mul",(Node("constant",value=float(c)),node)))
    expression=weighted[0] if weighted else Node("constant",value=0.)
    for term in weighted[1:]:expression=Node("add",(expression,term))
    return expression,atoms,history

def structured_symbolic_fit(inputs,teacher_values,parents,max_terms=8,beam_width=32,folds=4):
    """Fit a hierarchy-constrained mechanism to frozen neural predictions.

    Selection sees only query coordinates and teacher values.  Each parent may
    contribute at most one unary operator.  An interaction may enter only after
    both corresponding main effects, which avoids redundant correlated bases
    while retaining a generic, executable arithmetic grammar.
    """
    x=np.asarray(inputs,dtype=float);y=np.asarray(teacher_values,dtype=float)
    parents=tuple(map(int,parents))
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)<max(16,folds*2):raise ValueError("Invalid structured-fit data")
    if folds<2 or beam_width<1 or max_terms<1:raise ValueError("Invalid structured-fit search")
    library=mechanism_library(parents)
    matrix=np.column_stack([np.broadcast_to(node.evaluate(x),(len(x),)) for _,node in library])
    unary_parent={};interactions={}
    for index,(name,_) in enumerate(library):
        if index==0:continue
        kind,raw=name.split(":",1)
        if kind=="interaction":interactions[index]=tuple(map(int,raw.split(",")))
        else:unary_parent[index]=int(raw)
    fold_id=np.arange(len(x))%folds
    cache={}
    def score(support):
        support=tuple(sorted(support))
        if support in cache:return cache[support]
        errors=[]
        for fold in range(folds):
            test=fold_id==fold;train=~test
            coef=np.linalg.lstsq(matrix[train][:,support],y[train],rcond=None)[0]
            scale=max(float(np.var(y[test])),1e-8)
            errors.append(float(np.mean((matrix[test][:,support]@coef-y[test])**2)/scale))
        value=float(np.log(np.mean(errors)+1e-6)+.08*(len(support)-1))
        cache[support]=(value,float(np.mean(errors)))
        return cache[support]
    beam={(0,)};best=(0,);best_score=score(best)[0];history=[]
    for depth in range(1,min(max_terms,len(library))):
        expanded=set()
        for support in beam:
            selected_parents={unary_parent[i] for i in support if i in unary_parent}
            for index in range(1,len(library)):
                if index in support:continue
                if index in unary_parent and unary_parent[index] in selected_parents:continue
                if index in interactions and not set(interactions[index])<=selected_parents:continue
                expanded.add(tuple(sorted((*support,index))))
        if not expanded:break
        ranked=sorted(expanded,key=lambda s:(score(s)[0],s))[:beam_width]
        beam=set(ranked)
        candidate=ranked[0];candidate_score,candidate_nmse=score(candidate)
        history.append({"terms":list(candidate),"cross_validated_neural_nmse":candidate_nmse,"score":candidate_score,"depth":depth})
        if (candidate_score,candidate)<(best_score,best):best,best_score=candidate,candidate_score
    coef=np.linalg.lstsq(matrix[:,best],y,rcond=None)[0]
    weighted=[];atoms=[]
    for index,c in zip(best,coef):
        if abs(c)<.005 and index!=0:continue
        name,node=library[index];atoms.append({"atom":name,"coefficient":float(c)})
        weighted.append(Node("mul",(Node("constant",value=float(c)),node)))
    expression=weighted[0] if weighted else Node("constant",value=0.)
    for term in weighted[1:]:expression=Node("add",(expression,term))
    return expression,atoms,history
def expression_parents(node):
    if node.op=="var":return {node.index}
    return set().union(*(expression_parents(a) for a in node.args)) if node.args else set()

@dataclass
class ExplicitSCM:
    equations:list
    noise_samples:list
    source_graph:np.ndarray

    def __post_init__(self):
        topological_order(self.source_graph)
        if len(self.equations)!=len(self.source_graph) or len(self.noise_samples)!=len(self.equations):
            raise ValueError("Invalid SCM size")
        for j,e in enumerate(self.equations):
            if not expression_parents(e)<=set(np.flatnonzero(self.source_graph[:,j])):
                raise ValueError("Equation uses an undeclared parent")

    @property
    def graph(self):
        a=np.zeros_like(self.source_graph,dtype=bool)
        for j,e in enumerate(self.equations):
            for i in expression_parents(e):a[i,j]=True
        return a

    def sample(self,n=256,seed=42,interventions=None,noise_values=None):
        interventions={} if interventions is None else interventions
        if any(k not in range(len(self.equations)) for k in interventions):raise ValueError("Invalid intervention")
        rng=np.random.default_rng(seed)
        u=np.column_stack([rng.choice(v,n) for v in self.noise_samples]) if noise_values is None else np.asarray(noise_values)
        if u.shape!=(n,len(self.equations)):raise ValueError("Noise shape mismatch")
        data=np.zeros_like(u,dtype=float)
        for j in topological_order(self.source_graph):
            data[:,j]=interventions[j] if j in interventions else self.equations[j].evaluate(data)+u[:,j]
        if not np.isfinite(data).all():raise ValueError("Non-finite explicit SCM simulation")
        return data

    def to_dict(self):
        return {"equations":[e.to_dict() for e in self.equations],
                "canonical_equations":[canonical_expression(e) for e in self.equations],
                "noise_samples":[list(map(float,v)) for v in self.noise_samples],
                "source_graph":self.source_graph.astype(int).tolist(),"effective_graph":self.graph.astype(int).tolist(),
                "noise_assumption":"independent empirical additive residuals",
                "symbolic_supervision":"frozen_neural_mechanism_predictions_only"}

    @classmethod
    def from_dict(cls,d):
        return cls([Node.from_dict(e) for e in d["equations"]],[np.asarray(v) for v in d["noise_samples"]],np.array(d["source_graph"],bool))

def recover_mechanisms(data,graph,directory,*,seed=42,epochs=180,query_count=768):
    graph=np.asarray(graph,bool);topological_order(graph)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    expressions=[];noises=[];models=[];records=[]
    # Query coordinates come only from observed parent marginal ranges.
    query=np.empty((query_count,data.shape[1]))
    for j in range(data.shape[1]):
        low,high=np.quantile(data[:,j],[.02,.98])
        query[:,j]=rng.uniform(low,high,query_count)
    for j in range(data.shape[1]):
        parents=tuple(map(int,np.flatnonzero(graph[:,j])))
        model,training=train_mechanism(data,j,parents,seed=seed+j,epochs=epochs)
        targets=neural_values(model,query)
        equation,atoms,history=sparse_symbolic_fit(query,targets,parents)
        residual=data[:,j]-neural_values(model,data)
        expressions.append(equation);noises.append(residual);models.append(model)
        state={"parents":parents,"width":model.width,"state_dict":model.state_dict()}
        torch.save(state,directory/f"mechanism_{j}.pt")
        np.savez_compressed(directory/f"distillation_{j}.npz",queries=query,teacher=targets)
        records.append({"node":j,"parents":parents,"atoms":atoms,"training":training,"search_history":history,
                        "residual_parent_dependence":[dependence(residual[:128],data[:128,i]) for i in parents]})
    scm=ExplicitSCM(expressions,noises,graph)
    save_json(directory/"explicit_scm.json",scm.to_dict());save_json(directory/"recovery.json",records)
    np.savez_compressed(directory/"observations.npz",data=data)
    return scm,models,records

def load_mechanism(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    model=NeuralMechanism(state["parents"],state["width"]);model.load_state_dict(state["state_dict"]);model.eval()
    return model

def evaluate_recovery(world,scm,models,records,*,seed=1024,samples=512):
    data=world.sample(seed=seed,samples=samples)
    scales=np.array(world.scales or (1.,)*world.nodes);internal=data/scales
    details=[]
    for j,model in enumerate(models):
        truth=sum((t.evaluate(internal) for t in world.equations[j]),start=np.zeros(samples))*scales[j]
        neural=neural_values(model,data);symbolic=np.broadcast_to(scm.equations[j].evaluate(data),(samples,))
        scale=max(float(np.var(truth)),world.noise_scale**2*scales[j]**2,1e-8)
        true_parents=set(np.flatnonzero(np.asarray(world.graph)[:,j]));found=expression_parents(scm.equations[j])
        true_operators={t.operator for t in world.equations[j]}
        found_operators={a["atom"].split(":")[0] for a in records[j]["atoms"] if a["atom"]!="constant"}
        true_interactions={tuple(sorted(t.parents)) for t in world.equations[j] if t.operator=="interaction"}
        found_interactions={tuple(map(int,a["atom"].split(":")[1].split(","))) for a in records[j]["atoms"] if a["atom"].startswith("interaction:")}
        details.append({"node":j,"parent_exact":true_parents==found,"parent_jaccard":len(true_parents&found)/max(len(true_parents|found),1) if true_parents|found else 1.,
            "operator_exact":true_operators==found_operators,"interaction_exact":true_interactions==found_interactions,
            "neural_ground_truth_nmse":float(np.mean((neural-truth)**2)/scale),
            "symbolic_ground_truth_nmse":float(np.mean((symbolic-truth)**2)/scale),
            "symbolic_neural_mse":float(np.mean((symbolic-neural)**2)),
            "symbolic_neural_normalizer":max(float(np.var(neural)),world.noise_scale**2*scales[j]**2,1e-8),
            "symbolic_neural_nmse":float(np.mean((symbolic-neural)**2)/max(float(np.var(neural)),world.noise_scale**2*scales[j]**2,1e-8))})
    # Compare intervention contrasts, sharing exogenous/noise draws within each model.
    true_base,u=world.sample(seed=seed+1,samples=samples,return_exogenous=True)
    rng=np.random.default_rng(seed+1);estimated_u=np.column_stack([rng.choice(v,samples) for v in scm.noise_samples])
    estimated_base=scm.sample(samples,noise_values=estimated_u)
    intervention_metrics=[]
    for target in range(world.nodes):
        for value in (-1.,1.):
            truth=world.sample(interventions={target:value},samples=samples,exogenous=u)
            estimate=scm.sample(samples,interventions={target:value},noise_values=estimated_u)
            ate=(truth-true_base).mean(0);pred=(estimate-estimated_base).mean(0)
            intervention_metrics.append({"target":target,"value":value,"true_mean_effect":ate.tolist(),
                "estimated_mean_effect":pred.tolist(),"mean_effect_mae":float(np.mean(np.abs(ate-pred)))})
    return {"nodes":details,"interventions":intervention_metrics,
            "parent_exact_fraction":float(np.mean([d["parent_exact"] for d in details])),
            "mean_symbolic_neural_nmse":float(np.mean([d["symbolic_neural_nmse"] for d in details])),
            "mean_symbolic_truth_nmse":float(np.mean([d["symbolic_ground_truth_nmse"] for d in details])),
            "mean_intervention_effect_mae":float(np.mean([d["mean_effect_mae"] for d in intervention_metrics])),
            "graph_errors_are_not_hidden":True,
            "nmse_normalization":"neural signal variance floored at known benchmark exogenous noise variance"}
