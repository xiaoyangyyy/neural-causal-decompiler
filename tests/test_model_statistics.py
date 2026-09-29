import numpy as np
import torch
import pytest
from ncd.model import Discoverer,predict,set_seed,train,load_model
from ncd.statistics import extract_one,residual,dependence,FEATURES
from ncd.worlds import generate_worlds

def test_model_invariances_and_freezing(tmp_path):
    set_seed(3)
    data=np.stack([w.sample() for w in generate_worlds("train",32,n=32)])
    model=Discoverer(12)
    p=predict(model,data)
    np.testing.assert_allclose(p,predict(model,data[:,::-1].copy()),atol=1e-6)
    np.testing.assert_allclose(p[:,[1,0,2,3]],predict(model,data[:,:,::-1].copy()),atol=1e-7)
    labels=np.array([w.label for w in generate_worlds("train",32,n=32)])
    trained=train(data,labels,data,labels,tmp_path,epochs=1,width=12)
    np.testing.assert_array_equal(predict(trained,data),predict(load_model(tmp_path/"discoverer.pt"),data))

def test_statistical_permutation_and_direction():
    d=generate_worlds("train",3,n=48)[2].sample()
    a=extract_one(d)
    np.testing.assert_allclose(a,extract_one(d[::-1]),atol=1e-9)
    b=extract_one(d[:,::-1])
    assert a[FEATURES.index("resdep_xy")]==pytest.approx(b[FEATURES.index("resdep_yx")])
    assert a[FEATURES.index("var_log_ratio")]==pytest.approx(-b[FEATURES.index("var_log_ratio")])
    with pytest.raises(ValueError): extract_one(np.full((48,2),np.nan))

def test_residuals_use_held_out_targets():
    rng=np.random.default_rng(10)
    x=np.linspace(-2,2,64); y=x*x+rng.normal(0,.1,64)
    before=residual(y,x)
    # Changing this test target cannot change its fitted prediction.
    after_y=y.copy(); after_y[10]+=100
    after=residual(after_y,x)
    assert (after[10]-before[10])==pytest.approx(100)
    assert dependence(x,x)>.999
