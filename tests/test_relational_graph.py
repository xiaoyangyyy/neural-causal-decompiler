import numpy as np
import pytest
import torch
from ncd.graph_model import GraphDiscoverer,GRAPH_FEATURES,SWAP_LABELS,graph_labels
from ncd.relational_graph import RelationalGraphDiscoverer,edge_relations,train_relational_graph,load_relational_graph
from ncd.model import set_seed

def test_incidence_relations_have_expected_direction():
    r=edge_relations(3)
    # Query (0,1), source (1,2) is a successor; (2,0) is a predecessor.
    assert r[4,1,5] and not r[5,1,5]
    assert r[5,1,6] and not r[4,1,6]
    assert r[1,1,3] and r[2,1,2] and r[3,1,7]
    with pytest.raises(ValueError):edge_relations(1)

def test_nonzero_relation_bias_preserves_variable_equivariance():
    set_seed(493);m=RelationalGraphDiscoverer(16).eval()
    with torch.no_grad():m.relation_bias.normal_()
    for n in (3,5,8):
        x=torch.randn(2,n,n,len(GRAPH_FEATURES));p=torch.randperm(n)
        original=m(x)
        torch.testing.assert_close(m(x[:,p][:,:,p]),original[:,p][:,:,p],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(original,original.transpose(1,2)[...,SWAP_LABELS])
        assert torch.isfinite(original).all()

def test_disabled_model_matches_legacy_attention_and_parameter_count():
    set_seed(10);legacy=GraphDiscoverer(16).eval()
    disabled=RelationalGraphDiscoverer(16,False).eval()
    state=legacy.state_dict();missing,unexpected=disabled.load_state_dict(state,strict=False)
    assert missing==["relation_bias"] and not unexpected
    x=torch.randn(3,5,5,len(GRAPH_FEATURES))
    torch.testing.assert_close(legacy(x),disabled(x),atol=1e-6,rtol=1e-5)
    enabled=RelationalGraphDiscoverer(16,True)
    assert sum(p.numel() for p in enabled.parameters())==sum(p.numel() for p in disabled.parameters())

def test_incidence_information_changes_attention_beyond_token_multiset():
    set_seed(13);m=RelationalGraphDiscoverer(16).eval()
    with torch.no_grad():m.relation_bias[:,2]=5.
    x=torch.randn(1,4,4,len(GRAPH_FEATURES));y=x.clone()
    # Same token multiset and query/reverse-query values, different incidence.
    for a,b in (((0,2),(2,3)),((2,0),(3,2))):
        y[:,a[0],a[1]]=x[:,b[0],b[1]];y[:,b[0],b[1]]=x[:,a[0],a[1]]
    assert not torch.allclose(m(x)[:,0,1],m(y)[:,0,1],atol=1e-5)
    m.relations_enabled=False
    torch.testing.assert_close(m(x)[:,0,1],m(y)[:,0,1],atol=1e-6,rtol=1e-5)

def test_real_training_checkpoint_and_enabled_bias_update(tmp_path):
    rng=np.random.default_rng(1);groups=[]
    for n in (3,5):
        x=rng.normal(size=(5,n,n,len(GRAPH_FEATURES)))
        y=np.stack([graph_labels(np.triu(rng.random((n,n))<.3,1)) for _ in range(5)])
        groups.append((x,y))
    model=train_relational_graph(groups,groups,tmp_path,epochs=2,width=16,seed=19)
    restored=load_relational_graph(tmp_path/"graph_teacher.pt")
    assert torch.count_nonzero(model.relation_bias)>0
    x=torch.tensor(groups[0][0],dtype=torch.float32)
    torch.testing.assert_close(model(x),restored(x),atol=0,rtol=0)
