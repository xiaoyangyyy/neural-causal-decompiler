"""Fit-only linear readouts and numerical intervention audits.

Readouts are probes, not evidence on their own. Post-intervention comparisons
include the no-intervention baseline and report target versus collateral error.
"""
from dataclasses import dataclass
import numpy as np
from .joint_alignment import patch_blocks

def numeric_values(executor,x,addresses):
    trace=executor.execute(x);values=[];visited=[]
    for address in addresses:
        if address not in trace.records:raise ValueError("Missing numeric occurrence")
        r=trace.records[address];v=np.asarray(r["value"])
        if r["kind"]!="vector" or v.shape!=(len(x),) or not np.isfinite(v).all():
            raise ValueError("Finite per-world numeric occurrence required")
        values.append(v);visited.append(np.broadcast_to(r["visited"],(len(x),)))
    return np.column_stack(values),np.column_stack(visited)

@dataclass
class NumericReadout:
    addresses:tuple
    mean:np.ndarray
    scale:np.ndarray
    coefficients:np.ndarray
    target_variance:np.ndarray
    fit_counts:np.ndarray

    def predict(self,h):
        h=np.asarray(h)
        if h.ndim!=2 or h.shape[1]!=len(self.mean) or not np.isfinite(h).all():raise ValueError("Readout input mismatch")
        return np.c_[(h-self.mean)/self.scale,np.ones(len(h))]@self.coefficients
    def to_dict(self):
        return {"addresses":list(self.addresses),**{k:getattr(self,k).tolist() for k in
            ("mean","scale","coefficients","target_variance","fit_counts")}}
    @classmethod
    def from_dict(cls,d):
        return cls(tuple(d["addresses"]),*(np.array(d[k]) for k in
            ("mean","scale","coefficients","target_variance","fit_counts")))

def fit_readout_values(h,addresses,target,visited,ridge=.01):
    h=np.asarray(h,dtype=float);target=np.asarray(target,dtype=float);visited=np.asarray(visited,dtype=bool)
    if h.ndim!=2 or target.shape!=visited.shape or target.shape!=(len(h),len(addresses)) or not np.isfinite(h).all() or ridge<=0:
        raise ValueError("Invalid readout fitting data")
    mean=h.mean(0);scale=h.std(0).clip(1e-6);design=np.c_[(h-mean)/scale,np.ones(len(h))]
    coefficients=[];variances=[];counts=[]
    for j in range(len(addresses)):
        ids=visited[:,j]
        if ids.sum()<4:raise ValueError("Insufficient executed fitting worlds for readout")
        a=design[ids];y=target[ids,j];penalty=np.eye(a.shape[1])*ridge;penalty[-1,-1]=0
        coefficients.append(np.linalg.solve(a.T@a+penalty,a.T@y))
        variances.append(max(float(y.var()),1e-8));counts.append(int(ids.sum()))
    return NumericReadout(tuple(addresses),mean,scale,np.column_stack(coefficients),np.array(variances),np.array(counts))

def fit_readout(h,executor,features,addresses,ridge=.01):
    target,visited=numeric_values(executor,features,addresses)
    return fit_readout_values(h,addresses,target,visited,ridge)

def intervention_numeric_targets(executor,features,addresses,pairs):
    values,_=numeric_values(executor,features,addresses);k=len(addresses)
    targets=np.empty((len(pairs.base),k));visits=np.zeros_like(targets,dtype=bool)
    for mask in np.unique(pairs.masks,axis=0):
        ids=np.flatnonzero(np.all(pairs.masks==mask,axis=1))
        patches={a:values[pairs.sources[ids,j],j] for j,a in enumerate(addresses) if mask[j]}
        trace=executor.execute(features[pairs.base[ids]],patches)
        for j,a in enumerate(addresses):
            if a in trace.records:
                targets[ids,j]=trace.records[a]["value"];visits[ids,j]=trace.records[a]["visited"]
            else:
                targets[ids,j]=0  # Never scored: visited remains false.
    return targets,visits

def audit_numeric_values(readout,h,basis,pairs,original,original_visited,targets,visited):
    h=np.asarray(h);q=np.asarray(basis);original=np.asarray(original);original_visited=np.asarray(original_visited,dtype=bool)
    targets=np.asarray(targets);visited=np.asarray(visited,dtype=bool)
    if not len(pairs.base):return {"status":"no_executed_pairs","pairs":0,"per_node":[]}
    if q.shape[1]!=len(readout.addresses) or original.shape!=original_visited.shape or original.shape!=(len(h),len(readout.addresses)):
        raise ValueError("Readout/mapping mismatch")
    if targets.shape!=visited.shape or targets.shape!=(len(pairs.base),len(readout.addresses)):raise ValueError("Numeric target mismatch")
    flat=q.reshape(q.shape[0],-1)
    if not np.allclose(flat.T@flat,np.eye(flat.shape[1]),atol=1e-5):raise ValueError("Nonorthogonal blocks")
    predicted=readout.predict(patch_blocks(h[pairs.base],h[pairs.sources],q,pairs.masks));before=readout.predict(h[pairs.base]);natural=readout.predict(h)
    def metrics(y,p,b,mask,variance):
        if not mask.any():return {"n":0,"mse":None,"nmse":None,"no_intervention_nmse":None}
        error=float(np.mean((p[mask]-y[mask])**2))
        return {"n":int(mask.sum()),"mse":error,"nmse":error/variance,"no_intervention_nmse":float(np.mean((b[mask]-y[mask])**2)/variance)}
    rows=[]
    for j,address in enumerate(readout.addresses):
        variance=float(readout.target_variance[j]);ids=original_visited[:,j]
        rows.append({"address":address,"natural_nmse":float(np.mean((natural[ids,j]-original[ids,j])**2)/variance) if ids.any() else None,
            "natural_n":int(ids.sum()),"fit_variance":variance,
            "targeted":metrics(targets[:,j],predicted[:,j],before[:,j],visited[:,j]&pairs.masks[:,j],variance),
            "collateral":metrics(targets[:,j],predicted[:,j],before[:,j],visited[:,j]&~pairs.masks[:,j],variance)})
    return {"status":"measured","pairs":len(pairs.base),"per_node":rows,"predicted":predicted.tolist(),
        "symbolic_numeric_targets":targets.tolist(),"visited":visited.tolist(),
        "claim":"held-out numerical probe/intervention audit, not unique mechanism identification"}

def audit_oblique_numeric_values(readout,h,read,write,pairs,original,original_visited,targets,visited):
    """Numerical audit for a validated biorthogonal read/write mapping."""
    from .oblique_mapping import patch_oblique,validate_oblique
    validate_oblique(read,write)
    h=np.asarray(h);original=np.asarray(original);original_visited=np.asarray(original_visited,dtype=bool)
    targets=np.asarray(targets);visited=np.asarray(visited,dtype=bool)
    if not len(pairs.base):return {"status":"no_executed_pairs","pairs":0,"per_node":[]}
    if read.shape[1]!=len(readout.addresses) or original.shape!=original_visited.shape or original.shape!=(len(h),len(readout.addresses)):
        raise ValueError("Readout/oblique mapping mismatch")
    if targets.shape!=visited.shape or targets.shape!=(len(pairs.base),len(readout.addresses)):raise ValueError("Numeric target mismatch")
    predicted=readout.predict(patch_oblique(h[pairs.base],h[pairs.sources],read,write,pairs.masks))
    before=readout.predict(h[pairs.base]);natural=readout.predict(h)
    def metrics(y,p,b,mask,variance):
        if not mask.any():return {"n":0,"mse":None,"nmse":None,"no_intervention_nmse":None}
        error=float(np.mean((p[mask]-y[mask])**2))
        return {"n":int(mask.sum()),"mse":error,"nmse":error/variance,"no_intervention_nmse":float(np.mean((b[mask]-y[mask])**2)/variance)}
    rows=[]
    for j,address in enumerate(readout.addresses):
        variance=float(readout.target_variance[j]);ids=original_visited[:,j]
        rows.append({"address":address,"natural_nmse":float(np.mean((natural[ids,j]-original[ids,j])**2)/variance) if ids.any() else None,
            "natural_n":int(ids.sum()),"fit_variance":variance,
            "targeted":metrics(targets[:,j],predicted[:,j],before[:,j],visited[:,j]&pairs.masks[:,j],variance),
            "collateral":metrics(targets[:,j],predicted[:,j],before[:,j],visited[:,j]&~pairs.masks[:,j],variance)})
    return {"status":"measured","pairs":len(pairs.base),"per_node":rows,"predicted":predicted.tolist(),
        "symbolic_numeric_targets":targets.tolist(),"visited":visited.tolist(),"mapping_diagnostics":validate_oblique(read,write),
        "claim":"held-out biorthogonal numerical intervention audit, not unique mechanism identification"}

def audit_numeric(readout,executor,features,h,basis,pairs):
    if not len(pairs.base):return {"status":"no_executed_pairs","pairs":0,"per_node":[]}
    targets,visited=intervention_numeric_targets(executor,features,readout.addresses,pairs)
    original,original_visited=numeric_values(executor,features,readout.addresses)
    return audit_numeric_values(readout,h,basis,pairs,original,original_visited,targets,visited)
