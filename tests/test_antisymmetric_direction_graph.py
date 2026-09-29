import numpy as np
import torch
from ncd.graph_model import pair_features,graph_probabilities,SWAP_LABELS
from ncd.antisymmetric_direction_graph import AntisymmetricDirectionGraphDiscoverer


def test_antisymmetric_direction_constraints_and_equivariance():
    rng=np.random.default_rng(41);data=rng.normal(size=(48,5));x=pair_features(data)[None]
    model=AntisymmetricDirectionGraphDiscoverer(16)
    h=model.representation(torch.tensor(x,dtype=torch.float32));s,o=model.raw_components(h)
    torch.testing.assert_close(o[...,0],-o.transpose(1,2)[...,0],rtol=0,atol=1e-7)
    torch.testing.assert_close(o[...,1],o.transpose(1,2)[...,0],rtol=0,atol=1e-7)
    torch.testing.assert_close(o[...,2],o.transpose(1,2)[...,2],rtol=0,atol=1e-7)
    p=graph_probabilities(model,x);np.testing.assert_allclose(p.sum(-1),1,atol=5e-7)
    np.testing.assert_allclose(p,p.transpose(0,2,1,3)[...,SWAP_LABELS],atol=1e-7)
    order=np.array([4,1,3,0,2]);xp=x[:,order][:,:,order]
    np.testing.assert_allclose(graph_probabilities(model,xp),p[:,order][:,:,order],atol=1e-6)

