import numpy as np
from ncd.rules import Rule
from ncd.graph_global_program import GraphGlobalRankingProgram,count_features

def program():
    orientation=Rule(("f",),{"expr":{"op":"var","index":0},"threshold":0.,"left":{"label":2},"right":{"label":1}})
    return GraphGlobalRankingProgram(("min_f","max_f"),(0.,0.),(1.,1.),(0.,1.),0.,(0.,0.,0.,0.,0.,0.,0.,0.),(1.,1.,1.,1.,1.,1.,1.,1.),(0.,0.,0.,0.,0.,0.,0.,0.),2.,orientation,.1,1.)

def test_graph_global_program_round_trip_and_count():
    p=program();x=np.zeros((3,3,1));x[0,1]=2;x[1,0]=-1;x[0,2]=4;x[2,0]=-3;x[1,2]=6;x[2,1]=-5
    assert p.predicted_count(p.edge_scores(x),3)==2
    q=GraphGlobalRankingProgram.from_dict(p.to_dict());np.testing.assert_array_equal(q.predict(x),p.predict(x));assert q.edge_nonzero==1 and q.count_nonzero==0

def test_graph_global_program_uses_deterministic_top_scores():
    p=program();x=np.zeros((3,3,1));x[0,1]=1;x[1,0]=-.5;x[0,2]=3;x[2,0]=-2;x[1,2]=2;x[2,1]=-1
    g=p.predict(x);assert g[0,2] and g[1,2];assert not g[0,1]
    np.testing.assert_allclose(count_features(np.array([1.,2.,3.]),3),[3,2,np.std([1.,2.,3.]),1,3,1.5,2,2.5])
