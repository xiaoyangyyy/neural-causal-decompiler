import numpy as np
import pytest
import torch
from ncd.graph_model import GraphDiscoverer,graph_probabilities,pair_features,graph_labels,train_graph_model,load_graph_model,decode_graph,decode_graph_threshold,dag_completion,GRAPH_FEATURES
from ncd.multiverse import generate_graph_worlds
from ncd.mechanisms import sparse_symbolic_fit,ExplicitSCM,recover_mechanisms,evaluate_recovery,load_mechanism,neural_values
from ncd.cdir import Node
from ncd.rules import fit_rule,Rule

def test_graph_teacher_variable_and_sample_equivariance(tmp_path):
    worlds=generate_graph_worlds("train",8,3,samples=32)
    x=np.stack([pair_features(w.sample()) for w in worlds]);y=np.stack([graph_labels(w.target_graph) for w in worlds])
    model=GraphDiscoverer(16)
    p=graph_probabilities(model,x)
    permutation=[2,0,1]
    xp=x[:,permutation][:,:,permutation]
    np.testing.assert_allclose(graph_probabilities(model,xp),p[:,permutation][:,:,permutation],atol=1e-6)
    d=worlds[1].sample()
    np.testing.assert_allclose(pair_features(d[::-1]),pair_features(d),atol=1e-8)
    np.testing.assert_allclose(pair_features(d[:,permutation]),pair_features(d)[permutation][:,permutation],atol=1e-8)
    fitted=train_graph_model([(x,y)],[(x,y)],tmp_path,epochs=1,width=16)
    np.testing.assert_array_equal(graph_probabilities(fitted,x),graph_probabilities(load_graph_model(tmp_path/"graph_teacher.pt"),x))

def test_generic_rules_learn_conditional_edge_primitive():
    rng=np.random.default_rng(2);x=rng.normal(size=(128,len(GRAPH_FEATURES)))
    y=np.where(x[:,14]<0,1,0)
    rule,_=fit_rule(x,y,GRAPH_FEATURES,max_splits=3,beam_width=2)
    assert np.mean(rule.predict(x)==y)>.95
    restored=Rule.from_dict(rule.to_dict())
    np.testing.assert_array_equal(restored.predict(x),rule.predict(x))
    assert "min_abs_partial_1" in rule.text()

def test_symbolic_neural_targets_recover_interaction_and_sine():
    rng=np.random.default_rng(9);x=rng.uniform(-2,2,(1024,3))
    neural_output=.7*x[:,0]*x[:,1]+np.sin(x[:,2])
    e,atoms,_=sparse_symbolic_fit(x,neural_output,[0,1,2])
    validation=rng.uniform(-2,2,(128,3))
    np.testing.assert_allclose(e.evaluate(validation),.7*validation[:,0]*validation[:,1]+np.sin(validation[:,2]),atol=1e-6)
    assert {"interaction:0,1","sin:2"}<={a["atom"] for a in atoms}

def test_explicit_scm_intervention_and_roundtrip():
    x=Node("var",index=0)
    scm=ExplicitSCM([Node("constant",value=0),Node("mul",(Node("constant",value=2),x))],
                     [np.array([-1.,1.]),np.array([0.])],np.array([[0,1],[0,0]],bool))
    recovered=ExplicitSCM.from_dict(scm.to_dict())
    result=recovered.sample(32,interventions={0:3})
    np.testing.assert_allclose(result[:,1],6)
    np.testing.assert_array_equal(scm.sample(),recovered.sample())

def test_actual_neural_to_scm_pipeline(tmp_path):
    world=generate_graph_worlds("mechanism_test",2,3,samples=96)[1]
    data=world.sample(samples=192)
    scm,models,records=recover_mechanisms(data,world.graph,tmp_path,epochs=15,query_count=128)
    evaluation=evaluate_recovery(world,scm,models,records,samples=64)
    assert len(evaluation["nodes"])==3
    assert len(evaluation["interventions"])==6
    assert np.isfinite(evaluation["mean_intervention_effect_mae"])
    for j,model in enumerate(models):
        loaded=load_mechanism(tmp_path/f"mechanism_{j}.pt")
        np.testing.assert_array_equal(neural_values(loaded,data),neural_values(model,data))
    assert scm.to_dict()["symbolic_supervision"]=="frozen_neural_mechanism_predictions_only"


def test_threshold_decoder_controls_edges_and_preserves_acyclicity():
    p=np.zeros((3,3,4),float);p[...,0]=1
    p[0,1]=[.2,.7,.05,.05];p[1,0]=[.2,.05,.7,.05]
    p[1,2]=[.45,.5,.03,.02];p[2,1]=[.45,.03,.5,.02]
    low,_=decode_graph_threshold(p,.5);high,_=decode_graph_threshold(p,.7)
    assert low[0,1] and low[1,2]
    assert high[0,1] and not high[1,2]
    from ncd.graphs import topological_order
    topological_order(high&~high.T)