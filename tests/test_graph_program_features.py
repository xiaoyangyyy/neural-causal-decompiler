import numpy as np
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
