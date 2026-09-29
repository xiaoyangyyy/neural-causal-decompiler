"""PCA-tangent constrained hidden-state interventions."""
from dataclasses import dataclass
import numpy as np


@dataclass
class TangentSubspace:
    mean: np.ndarray
    basis: np.ndarray
    explained_variance: float
    threshold: float

    def to_dict(self):
        return {"mean":self.mean.tolist(),"basis":self.basis.tolist(),
                "explained_variance":float(self.explained_variance),"threshold":float(self.threshold)}

    @classmethod
    def from_dict(cls,d):
        return cls(np.asarray(d["mean"],dtype=float),np.asarray(d["basis"],dtype=float),
                   float(d["explained_variance"]),float(d["threshold"]))


def fit_tangent_subspace(h,threshold=.95):
    h=np.asarray(h,dtype=float)
    if h.ndim!=2 or len(h)<2 or not np.isfinite(h).all() or not 0<threshold<=1:
        raise ValueError("Invalid tangent fitting data")
    mean=h.mean(0);centered=h-mean
    _,s,vt=np.linalg.svd(centered,full_matrices=False)
    variance=s*s;total=float(variance.sum())
    if total<=0:raise ValueError("Degenerate tangent fitting data")
    cumulative=np.cumsum(variance)/total
    rank=int(np.searchsorted(cumulative,threshold,side="left")+1)
    basis=vt[:rank].T
    return TangentSubspace(mean,basis,float(cumulative[rank-1]),float(threshold))


def random_subspace(dimension,rank,seed):
    if type(dimension) is not int or type(rank) is not int or not 0<rank<=dimension:
        raise ValueError("Invalid random subspace dimensions")
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension,rank)),mode="reduced")[0]


def patch_tangent(base,sources,read,masks,basis,rcond=1e-10):
    """Minimum-norm displacement in basis matching active read coordinates."""
    base=np.asarray(base,dtype=float);sources=np.asarray(sources,dtype=float)
    read=np.asarray(read,dtype=float);masks=np.asarray(masks,dtype=bool);basis=np.asarray(basis,dtype=float)
    if base.ndim!=2 or sources.ndim!=3 or read.ndim!=3 or masks.ndim!=2:
        raise ValueError("Tangent patch dimensions")
    if sources.shape[:2]!=masks.shape or sources.shape[0]!=len(base) or sources.shape[2]!=base.shape[1]:
        raise ValueError("Tangent source dimensions")
    if read.shape[:2]!=(base.shape[1],masks.shape[1]) or basis.ndim!=2 or basis.shape[0]!=base.shape[1]:
        raise ValueError("Tangent basis dimensions")
    if not np.allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-7) or rcond<=0:
        raise ValueError("Tangent basis must be orthonormal")
    out=base.copy();residuals=[];ranks=[];conditions=[]
    for mask in np.unique(masks,axis=0):
        ids=np.flatnonzero(np.all(masks==mask,axis=1));active=np.flatnonzero(mask)
        if not len(active):continue
        r=read[:,active,:].reshape(base.shape[1],-1)
        desired=np.concatenate([(sources[ids,j]-base[ids])@read[:,j,:] for j in active],axis=1)
        constraint=basis.T@r;pinv=np.linalg.pinv(constraint,rcond=rcond)
        delta=(desired@pinv)@basis.T;out[ids]=base[ids]+delta
        achieved=delta@r;singular=np.linalg.svd(constraint,compute_uv=False)
        residuals.append(float(np.max(np.abs(achieved-desired))));ranks.append(int(np.linalg.matrix_rank(constraint,tol=rcond)))
        conditions.append(float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf"))
    displacement=np.linalg.norm(out-base,axis=1)
    diagnostics={"max_constraint_residual":max(residuals,default=0.),
        "mean_displacement_norm":float(displacement.mean()),"max_displacement_norm":float(displacement.max()),
        "minimum_constraint_rank":min(ranks,default=0),"maximum_constraint_condition":max(conditions,default=0.)}
    return out,diagnostics
