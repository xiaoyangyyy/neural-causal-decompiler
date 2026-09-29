"""Permutation-equivariant relational context features for graph programs."""
import numpy as np
from .graph_model import GRAPH_FEATURES

RELATIONS=("same_source","same_target","successor","predecessor")
RELATIONAL_PROGRAM_FEATURES=tuple(GRAPH_FEATURES)+tuple(
    f"{relation}_mean_{name}" for relation in RELATIONS for name in GRAPH_FEATURES)


def relational_program_features(features):
    """Append explicit means over incidence-related directed pair tokens.

    Input has shape (..., nodes, nodes, features). Diagonal pair tokens are never
    included. Empty relation sets (only possible for very small graphs) map to zero.
    The fixed relation definitions commute with every node permutation.
    """
    x=np.asarray(features,dtype=float)
    if x.ndim<3 or x.shape[-3]!=x.shape[-2] or x.shape[-1]!=len(GRAPH_FEATURES):
        raise ValueError("Expected (..., nodes, nodes, graph_features)")
    if not np.isfinite(x).all():raise ValueError("Non-finite graph program features")
    n=x.shape[-3]
    if n<2:raise ValueError("At least two graph nodes are required")
    prefix=x.shape[:-3];out=np.zeros(prefix+(n,n,len(RELATIONAL_PROGRAM_FEATURES)),dtype=float)
    out[...,:len(GRAPH_FEATURES)]=x
    for i in range(n):
        for j in range(n):
            if i==j:continue
            sets=(x[...,i,[k for k in range(n) if k not in (i,j)],:],
                  x[...,[k for k in range(n) if k not in (i,j)],j,:],
                  x[...,j,[k for k in range(n) if k not in (i,j)],:],
                  x[...,[k for k in range(n) if k not in (i,j)],i,:])
            cursor=len(GRAPH_FEATURES)
            for values in sets:
                out[...,i,j,cursor:cursor+len(GRAPH_FEATURES)]=values.mean(axis=-2) if values.shape[-2] else 0.
                cursor+=len(GRAPH_FEATURES)
    return out
