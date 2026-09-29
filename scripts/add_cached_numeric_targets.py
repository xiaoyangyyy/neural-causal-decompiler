from pathlib import Path
p=Path('ncd/numeric_audit.py');s=p.read_text(encoding='utf-8')
start=s.index('def fit_readout(');end=s.index('\ndef intervention_numeric_targets',start)
replacement='''def fit_readout_values(h,addresses,target,visited,ridge=.01):
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
'''
s=s[:start]+replacement+s[end:]
start=s.index('def audit_numeric(')
old=s[start:]
# retain a cache-based core and compatibility wrapper
replacement='''def audit_numeric_values(readout,h,basis,pairs,original,original_visited,targets,visited):
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

def audit_numeric(readout,executor,features,h,basis,pairs):
    if not len(pairs.base):return {"status":"no_executed_pairs","pairs":0,"per_node":[]}
    targets,visited=intervention_numeric_targets(executor,features,readout.addresses,pairs)
    original,original_visited=numeric_values(executor,features,readout.addresses)
    return audit_numeric_values(readout,h,basis,pairs,original,original_visited,targets,visited)
'''
s=s[:start]+replacement
p.write_text(s,encoding='utf-8')
# Test exact compatibility of cached and direct paths.
p=Path('tests/test_numeric_mechanisms.py');s=p.read_text(encoding='utf-8');s=s.replace('from ncd.numeric_audit import', 'from ncd.numeric_audit import')
s+='''

def test_cached_numeric_readout_and_audit_match_direct_paths():
    from ncd.numeric_audit import (fit_readout,fit_readout_values,numeric_values,
        intervention_numeric_targets,audit_numeric,audit_numeric_values)
    from ncd.mechanism_pairs import compatible_combinations,execution_conditioned_pairs,numeric_frontier
    e=executor();rng=np.random.default_rng(99);x=rng.normal(size=(80,3));addresses=numeric_frontier(e)
    h=np.c_[x,x*x,rng.normal(size=(80,2))]
    values,visited=numeric_values(e,x,addresses)
    direct=fit_readout(h,e,x,addresses);cached=fit_readout_values(h,addresses,values,visited)
    assert direct.to_dict()==cached.to_dict()
    masks=compatible_combinations(e,addresses);pairs,_=execution_conditioned_pairs(e,x,addresses,masks,count=12,seed=9)
    targets,target_visited=intervention_numeric_targets(e,x,addresses,pairs)
    q=np.linalg.qr(rng.normal(size=(h.shape[1],len(addresses))))[0].reshape(h.shape[1],len(addresses),1)
    assert audit_numeric(direct,e,x,h,q,pairs)==audit_numeric_values(cached,h,q,pairs,values,visited,targets,target_visited)
'''
p.write_text(s,encoding='utf-8')
