import numpy as np
import pytest
from ncd.worlds import World,generate_worlds,save_dataset,load_worlds,FAMILIES,UNKNOWN,INDEPENDENT

def test_worlds_reproduce_and_split_isolate(tmp_path):
    a=generate_worlds("train",24,n=32)
    b=generate_worlds("test_id",24,n=32)
    assert not {w.identity for w in a}&{w.identity for w in b}
    assert {w.family for w in a}==set(FAMILIES)
    save_dataset(tmp_path,a)
    restored=load_worlds(tmp_path)
    for x,y in zip(a,restored):
        assert x==y
        np.testing.assert_array_equal(x.sample(),y.sample())
        assert np.isfinite(x.sample(True)).all()
        assert x.ast()["internal_variables"]

def test_labels_and_ood():
    ws=generate_worlds("test_function",60,n=32)
    for w in ws:
        if w.family=="linear_gaussian": assert w.label==UNKNOWN
        if w.family=="independent": assert w.label==INDEPENDENT
        if w.family in ("anm","multiplicative","heteroskedastic"):
            assert w.function in ("cos","piecewise","sigmoid")
    with pytest.raises(ValueError): generate_worlds("typo",10)
    with pytest.raises(ValueError): generate_worlds("train",10,n=4)

def test_swapping_world_scales_preserves_orientation():
    from dataclasses import replace
    w=generate_worlds("train",12,n=32)[2]
    swapped=replace(w,direction=1-w.direction,scale_x=w.scale_y,scale_y=w.scale_x)
    np.testing.assert_array_equal(w.sample()[:,::-1],swapped.sample())

def test_intervention_preserves_exogenous_effect_noise():
    w=World(7,"train","anm",0,"linear","gaussian","gaussian",1.2,.3,n=64)
    observational,intervened=w.sample(False),w.sample(True)
    np.testing.assert_allclose(observational[:,1]-1.2*observational[:,0],
                               intervened[:,1]-1.2*intervened[:,0],atol=5e-7)
    assert not np.array_equal(observational[:,0],intervened[:,0])
