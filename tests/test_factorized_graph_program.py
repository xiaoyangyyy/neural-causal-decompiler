import numpy as np
from ncd.rules import Rule
from ncd.factorized_graph_program import FactorizedGraphProgram


def leaf(label):return Rule(("f",),{"label":label})


def oriented_rule():
    return Rule(("f",),{"expr":{"op":"var","index":0},"threshold":0.,
                         "left":{"label":2},"right":{"label":1}})


def test_factorized_program_is_swap_consistent_and_acyclic():
    program=FactorizedGraphProgram(leaf(1),oriented_rule(),"and")
    x=np.zeros((3,3,1))
    for i in range(3):
        for j in range(i+1,3):x[i,j,0]=1.;x[j,i,0]=-1.
    graph=program.predict(x)
    assert graph[0,1] and graph[0,2] and graph[1,2]
    assert not graph[1,0]
    permutation=[2,0,1];xp=x[permutation][:,permutation]
    np.testing.assert_array_equal(program.predict(xp),graph[permutation][:,permutation])


def test_conflicting_orientation_votes_become_undirected():
    graph=FactorizedGraphProgram(leaf(1),leaf(1),"and").predict(np.zeros((2,2,1)))
    assert graph[0,1] and graph[1,0]
