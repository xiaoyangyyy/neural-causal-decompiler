from itertools import permutations,combinations
import numpy as np
import pytest
from ncd.graphs import topological_order,d_separated,cpdag,orient_colliders,pc_stable,acyclic_projection
from ncd.multiverse import generate_graph_worlds,save_graph_dataset,load_graph_worlds

def test_chain_and_collider_distinguished():
    chain=np.array([[0,1,0],[0,0,1],[0,0,0]],bool)
    collider=np.array([[0,0,1],[0,0,1],[0,0,0]],bool)
    assert not d_separated(chain,0,2,())
    assert d_separated(chain,0,2,(1,))
    assert d_separated(collider,0,1,())
    assert not d_separated(collider,0,1,(2,))
    np.testing.assert_array_equal(cpdag(chain),chain|chain.T)
    np.testing.assert_array_equal(cpdag(collider),collider)
    sk=chain|chain.T
    np.testing.assert_array_equal(orient_colliders(sk,{(0,2):(1,)}),sk)
    np.testing.assert_array_equal(orient_colliders(sk,{}),sk)

def test_cpdag_matches_enumerated_markov_equivalence_classes():
    # Exhaustively enumerate 3-node DAGs; orientations shared by every DAG in
    # the same skeleton/v-structure equivalence class must match CPDAG output.
    dags={}
    for order in permutations(range(3)):
        pairs=list(combinations(order,2))
        for mask in range(8):
            a=np.zeros((3,3),bool)
            for bit,(i,j) in enumerate(pairs):a[i,j]=bool(mask&(1<<bit))
            dags[a.tobytes()]=a
    groups={}
    for a in dags.values():
        sk=a|a.T
        colliders=tuple((i,k,j) for k in range(3) for i,j in combinations(range(3),2)
                        if i!=k and j!=k and a[i,k] and a[j,k] and not sk[i,j])
        groups.setdefault((sk.tobytes(),colliders),[]).append(a)
    for group in groups.values():
        expected=np.logical_or.reduce(group)
        for a in group:np.testing.assert_array_equal(cpdag(a),expected)

def test_scm_roundtrip_and_surgical_intervention(tmp_path):
    for nodes in (3,5,8):
        ws=generate_graph_worlds("train",6,nodes,samples=32)
        path=tmp_path/str(nodes);save_graph_dataset(path,ws)
        loaded=load_graph_worlds(path)
        assert ws==loaded
        for w in ws:
            data,u=w.sample(return_exogenous=True)
            intervened=w.sample(interventions={0:1.25},exogenous=u)
            np.testing.assert_allclose(intervened[:,0],1.25)
            assert not w.intervention_graph([0])[:,0].any()
            assert np.isfinite(data).all()
            assert w.metadata()["equation_ast"]
    with pytest.raises(ValueError):topological_order(np.array([[0,1],[1,0]]))

def test_projection_reports_removed_cycles():
    p=np.array([[0,.9,.1],[.1,0,.8],[.7,.1,0]])
    a,rejected=acyclic_projection(p)
    assert len(topological_order(a))==3
    assert rejected==[[2,0,.7]]

def test_four_node_cpdag_matches_all_dag_equivalence_classes():
    dags={}
    for order in permutations(range(4)):
        pairs=list(combinations(order,2))
        for mask in range(64):
            a=np.zeros((4,4),bool)
            for bit,(i,j) in enumerate(pairs):a[i,j]=bool(mask&(1<<bit))
            dags[a.tobytes()]=a
    groups={}
    for a in dags.values():
        sk=a|a.T
        colliders=tuple((i,k,j) for k in range(4) for i,j in combinations(range(4),2)
                        if i!=k and j!=k and a[i,k] and a[j,k] and not sk[i,j])
        groups.setdefault((sk.tobytes(),colliders),[]).append(a)
    assert len(dags)==543
    for group in groups.values():
        expected=np.logical_or.reduce(group)
        for a in group:np.testing.assert_array_equal(cpdag(a),expected)


def test_pc_trace_exposes_skeleton_colliders_and_meek_result():
    worlds=generate_graph_worlds("train",1,3,321,samples=96)
    result=pc_stable(worlds[0].sample(),alpha=.01,max_condition=2)
    assert set(("skeleton","collider_pdag","pdag","sepsets","tests"))<=set(result)
    np.testing.assert_array_equal(result["skeleton"],result["skeleton"].T)
    assert result["pdag"].shape==(3,3)