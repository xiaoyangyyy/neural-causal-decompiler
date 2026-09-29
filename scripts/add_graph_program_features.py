from pathlib import Path
p=Path('ncd/graph_program_features.py')
p.write_text('''"""Permutation-equivariant relational context features for graph programs."""
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
''',encoding='utf-8')
Path('tests/test_graph_program_features.py').write_text('''import numpy as np
import pytest
from ncd.graph_model import GRAPH_FEATURES
from ncd.graph_program_features import RELATIONAL_PROGRAM_FEATURES,relational_program_features


def test_relational_program_features_have_exact_incidence_semantics():
    n=4;f=len(GRAPH_FEATURES);x=np.zeros((1,n,n,f))
    for i in range(n):
        for j in range(n):x[0,i,j]=100*i+j+np.arange(f)/100
    z=relational_program_features(x);i,j=0,1;others=[2,3]
    expected=np.r_[x[0,i,j],x[0,i,others].mean(0),x[0,others,j].mean(0),
                   x[0,j,others].mean(0),x[0,others,i].mean(0)]
    assert z.shape==(1,n,n,len(RELATIONAL_PROGRAM_FEATURES))
    assert np.allclose(z[0,i,j],expected)


def test_relational_program_features_are_node_permutation_equivariant():
    rng=np.random.default_rng(71);x=rng.normal(size=(3,5,5,len(GRAPH_FEATURES)))
    permutation=np.array([3,0,4,1,2]);expected=relational_program_features(x)
    permuted=x[:,permutation][:,:,permutation]
    assert np.allclose(relational_program_features(permuted),expected[:,permutation][:,:,permutation])


@pytest.mark.parametrize('shape',[(3,4),(2,3,4,20),(1,1,20),(2,2,19)])
def test_relational_program_features_reject_invalid_shape(shape):
    with pytest.raises(ValueError):relational_program_features(np.zeros(shape))


def test_relational_program_features_reject_nonfinite():
    x=np.zeros((2,2,len(GRAPH_FEATURES)));x[0,1,0]=np.nan
    with pytest.raises(ValueError):relational_program_features(x)
''',encoding='utf-8')
