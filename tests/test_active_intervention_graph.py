import numpy as np
from ncd.multiverse import GraphWorld,Term
from ncd.active_intervention_graph import intervention_response_features,padded_observational_features,ACTIVE_FEATURE_COUNT,ActiveFactorizedGraphDiscoverer,load_active_factorized_graph
from ncd.graph_model import graph_probabilities


def test_active_features_detect_ordered_effect_and_model_runs():
    graph=((0,1,0),(0,0,0),(0,0,0));equations=((),(Term("linear",(0,),.8),),())
    world=GraphWorld(73,"train",graph,equations,"laplace",.2,64,(1.,1.,1.),"nonlinear",False)
    data=world.sample();active=intervention_response_features(world,data);control=padded_observational_features(data)
    assert active.shape==(3,3,ACTIVE_FEATURE_COUNT) and control.shape==active.shape
    np.testing.assert_array_equal(control[..., -4:],0)
    assert active[0,1,-3]>0.5
    np.testing.assert_allclose(active[1,0,-4:],0,atol=1e-12)
    model=ActiveFactorizedGraphDiscoverer(16);p=graph_probabilities(model,active[None])
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-7)

def test_active_graph_checkpoint_loader(tmp_path):
    model=ActiveFactorizedGraphDiscoverer(16);path=tmp_path/"model.pt"
    torch_state={"width":16,"state_dict":model.state_dict(),"architecture":"active_input_intervention_v1"}
    import torch
    torch.save(torch_state,path);loaded=load_active_factorized_graph(path)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key])
