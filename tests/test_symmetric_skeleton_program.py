import numpy as np
from ncd.rules import Rule
from ncd.symmetric_skeleton_program import symmetric_pair_features,SymmetricSkeletonGraphProgram

def leaf(names,label):return Rule(tuple(names),{"label":label})

def test_symmetric_features_and_program_are_permutation_equivariant():
    x=np.zeros((3,3,1))
    x[0,1,0]=2;x[1,0,0]=-1;x[0,2,0]=4;x[2,0,0]=-3;x[1,2,0]=6;x[2,1,0]=-5
    np.testing.assert_array_equal(symmetric_pair_features(x),[[-1,2],[-3,4],[-5,6]])
    skeleton=leaf(("min_f","max_f"),1)
    orientation=Rule(("f",),{"expr":{"op":"var","index":0},"threshold":0.,"left":{"label":2},"right":{"label":1}})
    p=SymmetricSkeletonGraphProgram(skeleton,orientation);g=p.predict(x);perm=[2,0,1]
    np.testing.assert_array_equal(p.predict(x[perm][:,perm]),g[perm][:,perm])

def test_symmetric_program_round_trip():
    p=SymmetricSkeletonGraphProgram(leaf(("min_f","max_f"),0),leaf(("f",),3))
    assert SymmetricSkeletonGraphProgram.from_dict(p.to_dict())==p
