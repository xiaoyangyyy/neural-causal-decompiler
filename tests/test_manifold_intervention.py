import numpy as np
from ncd.manifold_intervention import fit_tangent_subspace,random_subspace,patch_tangent


def test_tangent_patch_stays_in_span_and_matches_active_coordinates():
    rng=np.random.default_rng(7);h=rng.normal(size=(100,6));sub=fit_tangent_subspace(h,.8)
    basis=random_subspace(6,4,9);read=rng.normal(size=(6,3,1))
    base=h[:5];sources=np.stack([h[5:10],h[10:15],h[15:20]],axis=1)
    masks=np.array([[1,0,0],[0,1,1],[1,1,0],[0,0,1],[1,0,1]],dtype=bool)
    patched,diag=patch_tangent(base,sources,read,masks,basis)
    np.testing.assert_allclose((patched-base)@(np.eye(6)-basis@basis.T),0,atol=1e-9)
    for i in range(len(base)):
        for j in np.flatnonzero(masks[i]):
            np.testing.assert_allclose((patched[i]-base[i])@read[:,j,:],(sources[i,j]-base[i])@read[:,j,:],atol=1e-8)
    assert diag["max_constraint_residual"]<1e-8
    assert 1<=sub.basis.shape[1]<=6
