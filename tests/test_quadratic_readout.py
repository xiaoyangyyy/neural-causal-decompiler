import numpy as np
import torch
from ncd.quadratic_readout import fit_quadratic_readout_values,QuadraticNumericReadout

def test_quadratic_readout_round_trip_and_torch_match():
    rng=np.random.default_rng(3);h=rng.normal(size=(40,4));y=np.c_[h[:,0]+h[:,1]**2,h[:,2]**2-h[:,3]];visited=np.ones_like(y,dtype=bool);r=fit_quadratic_readout_values(h,("a","b"),y,visited,ridge=1e-6)
    q=QuadraticNumericReadout.from_dict(r.to_dict());np.testing.assert_allclose(q.predict(h),y,atol=1e-5);np.testing.assert_allclose(q.predict(h),q.torch_predict(torch.tensor(h,dtype=torch.float64)).numpy(),atol=1e-10)

