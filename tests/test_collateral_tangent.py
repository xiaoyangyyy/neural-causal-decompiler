import numpy as np
from ncd.collateral_tangent import patch_collateral_tangent
from ncd.manifold_intervention import patch_tangent,random_subspace


def test_collateral_tangent_preserves_active_and_reduces_inactive_leakage():
    rng=np.random.default_rng(12);d=10;k=4;n=20
    basis=random_subspace(d,7,2);read=rng.normal(size=(d,k,1));base=rng.normal(size=(n,d))
    sources=rng.normal(size=(n,k,d));masks=np.zeros((n,k),bool)
    for i in range(n):masks[i,i%k]=True
    plain,_=patch_tangent(base,sources,read,masks,basis)
    fixed,diag=patch_collateral_tangent(base,sources,read,masks,basis,ridge=1e-6)
    plain_leak=[];fixed_leak=[]
    for i in range(n):
        active=np.flatnonzero(masks[i]);inactive=np.flatnonzero(~masks[i])
        np.testing.assert_allclose((fixed[i]-base[i])@read[:,active,:].reshape(d,-1),
            np.concatenate([(sources[i,j]-base[i])@read[:,j,:] for j in active]),atol=1e-8)
        plain_leak.extend(((plain[i]-base[i])@read[:,inactive,:].reshape(d,-1)).tolist())
        fixed_leak.extend(((fixed[i]-base[i])@read[:,inactive,:].reshape(d,-1)).tolist())
    assert np.mean(np.square(fixed_leak))<np.mean(np.square(plain_leak))
    assert diag["max_constraint_residual"]<1e-8
