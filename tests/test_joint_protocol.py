import numpy as np
from ncd.cdir import Node
from ncd.rules import Rule
from ncd.program_trace import ProgramExecutor
from ncd.joint_experiment import protocol,worlds_for,JointConfig

def test_combinations_are_fixed_disjoint_and_cover_both_types():
    leaf={"label":0};expr=Node("var",index=0).to_dict()
    tree={"expr":expr,"threshold":0.,"left":leaf,"right":{
        "expr":expr,"threshold":1.,"left":leaf,"right":{
        "expr":expr,"threshold":2.,"left":leaf,"right":{"label":1}}}}
    e=ProgramExecutor(Rule(("x",),tree))
    for mode in ("expression","predicate"):
        addresses,fit,test,held=protocol(e,mode)
        assert len(addresses)==3 and len(held)==2
        fit_set={tuple(m) for m in fit};held_set={tuple(m) for m in held}
        assert not fit_set&held_set
        assert {tuple(m) for m in test}==fit_set|held_set
        assert len(test)==len(np.unique(test,axis=0))
        assert all(tuple(m) in fit_set for m in np.eye(3,dtype=bool))
        assert all(a.endswith("/expr" if mode=="expression" else "/predicate") for a in addresses)

def test_alignment_worlds_have_no_identity_overlap():
    c=JointConfig.quick()
    splits=("train","dev","extraction","alignment_fit","alignment_validation","alignment_test")
    identities=set()
    for split in splits:
        worlds=worlds_for(split,24,c)
        assert worlds==worlds_for(split,24,c)
        ids={w.identity for w in worlds}
        assert len(ids)==24 and not ids&identities
        identities|=ids
