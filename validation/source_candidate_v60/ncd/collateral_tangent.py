"""Tangent interventions that suppress inactive read-coordinate leakage."""
import numpy as np


def patch_collateral_tangent(base,sources,read,masks,basis,ridge=1e-3,rcond=1e-10):
    base=np.asarray(base,dtype=float);sources=np.asarray(sources,dtype=float)
    read=np.asarray(read,dtype=float);masks=np.asarray(masks,dtype=bool);basis=np.asarray(basis,dtype=float)
    if base.ndim!=2 or sources.ndim!=3 or read.ndim!=3 or masks.ndim!=2:
        raise ValueError("Collateral tangent dimensions")
    if sources.shape[:2]!=masks.shape or sources.shape[0]!=len(base) or sources.shape[2]!=base.shape[1]:
        raise ValueError("Collateral tangent source dimensions")
    if read.shape[:2]!=(base.shape[1],masks.shape[1]) or basis.ndim!=2 or basis.shape[0]!=base.shape[1]:
        raise ValueError("Collateral tangent basis dimensions")
    if not np.allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-7) or ridge<=0 or rcond<=0:
        raise ValueError("Invalid collateral tangent solver")
    out=base.copy();active_errors=[];inactive_squares=[];inactive_count=0;conditions=[]
    for mask in np.unique(masks,axis=0):
        ids=np.flatnonzero(np.all(masks==mask,axis=1));active=np.flatnonzero(mask);inactive=np.flatnonzero(~mask)
        if not len(active):continue
        ra=read[:,active,:].reshape(base.shape[1],-1);a=basis.T@ra
        desired=np.concatenate([(sources[ids,j]-base[ids])@read[:,j,:] for j in active],axis=1)
        z0=desired@np.linalg.pinv(a,rcond=rcond)
        # Null columns span vectors z with z @ a == 0.
        _,_,vt=np.linalg.svd(a.T,full_matrices=True);rank=np.linalg.matrix_rank(a.T,tol=rcond)
        null=vt[rank:].T
        z=z0
        if null.shape[1] and len(inactive):
            ri=read[:,inactive,:].reshape(base.shape[1],-1);c=basis.T@ri;d=null.T@c
            gram=d@d.T+ridge*np.eye(null.shape[1])
            q=-(z0@c)@d.T@np.linalg.inv(gram)
            z=z0+q@null.T
        delta=z@basis.T;out[ids]=base[ids]+delta
        active_errors.append(float(np.max(np.abs(delta@ra-desired))))
        if len(inactive):
            leak=delta@read[:,inactive,:].reshape(base.shape[1],-1)
            inactive_squares.append(float(np.sum(leak*leak)));inactive_count+=leak.size
        singular=np.linalg.svd(a,compute_uv=False)
        conditions.append(float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf"))
    displacement=np.linalg.norm(out-base,axis=1)
    diagnostics={"max_constraint_residual":max(active_errors,default=0.),
        "inactive_coordinate_rms":float(np.sqrt(sum(inactive_squares)/inactive_count)) if inactive_count else 0.,
        "mean_displacement_norm":float(displacement.mean()),"max_displacement_norm":float(displacement.max()),
        "maximum_constraint_condition":max(conditions,default=0.),"ridge":float(ridge)}
    return out,diagnostics
