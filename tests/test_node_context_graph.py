import numpy as np
from ncd.graph_model import pair_features,graph_probabilities
from ncd.node_context_graph import NodeContextGraphDiscoverer


def test_node_context_graph_is_node_permutation_equivariant():
    rng=np.random.default_rng(21);data=rng.normal(size=(48,5));x=pair_features(data)[None]
    model=NodeContextGraphDiscoverer(16);p=graph_probabilities(model,x)
    order=np.array([3,0,4,1,2]);xp=x[:,order][:,:,order]
    np.testing.assert_allclose(graph_probabilities(model,xp),p[:,order][:,:,order],atol=1e-6)
