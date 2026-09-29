import numpy as np
import torch
from ncd.graph_model import pair_features,graph_probabilities
from ncd.factorized_node_context_graph import FactorizedNodeContextGraphDiscoverer


def test_factorized_node_context_probabilities_and_equivariance():
    rng=np.random.default_rng(31);data=rng.normal(size=(48,5));x=pair_features(data)[None]
    model=FactorizedNodeContextGraphDiscoverer(16);p=graph_probabilities(model,x)
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-7)
    order=np.array([4,1,3,0,2]);xp=x[:,order][:,:,order]
    np.testing.assert_allclose(graph_probabilities(model,xp),p[:,order][:,:,order],atol=1e-6)
    h=model.representation(torch.tensor(x,dtype=torch.float32));s,o=model.raw_components(h)
    assert s.shape==(1,5,5) and o.shape==(1,5,5,3)
