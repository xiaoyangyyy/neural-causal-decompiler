"""Diagnostic MILP for log-size grid synthesis under a float Lipschitz relaxation."""
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import certify_weighted
from ncd.io import read_json
from validation.explore_automatic_grid import absolute_jacobian_bound


def solve(system, epsilon=.17, action_bins=128, max_bin=16, seconds=30):
    d=system.state_dim
    mat=absolute_jacobian_bound(system.transition)
    mx,ma=mat[:,:d],mat[:,d:]
    obs=absolute_jacobian_bound(system.observation)
    transfer=np.linalg.solve(np.eye(d)-mx,np.eye(d))
    influence=obs@transfer
    base=influence@(ma.sum(axis=1)/(2*action_bins))
    nvars=d*max_bin
    rows=[];cols=[];data=[]
    for j in range(d):
        for k in range(1,max_bin+1):
            ix=j*max_bin+k-1
            rows.append(j);cols.append(ix);data.append(1.)
            for i in range(obs.shape[0]):
                rows.append(d+i);cols.append(ix);data.append(influence[i,j]/(2*k))
    matrix=coo_matrix((data,(rows,cols)),shape=(d+obs.shape[0],nvars)).tocsr()
    lo=np.r_[np.ones(d),np.full(obs.shape[0],-np.inf)]
    hi=np.r_[np.ones(d),np.full(obs.shape[0],epsilon*.98-base)]
    costs=np.tile(np.log(np.arange(1,max_bin+1)),d)
    result=milp(costs,integrality=np.ones(nvars),bounds=Bounds(np.zeros(nvars),np.ones(nvars)),constraints=LinearConstraint(matrix,lo,hi),options={"time_limit":seconds,"mip_rel_gap":0.0})
    if result.x is None:
        return result.status,None,None
    bins=tuple(int(np.argmax(result.x[j*max_bin:(j+1)*max_bin])+1) for j in range(d))
    r=transfer@(ma.sum(axis=1)/(2*action_bins)+1/(2*np.asarray(bins)))
    r=np.ceil(r*1.005*1e8)/1e8
    cert=certify_weighted(system,bins,tuple(f"{x:.8f}" for x in r),action_bins=action_bins,epsilon=str(epsilon),packing_axes=min(4,d))
    return result.status,bins,cert


if __name__=="__main__":
    for d in (8,32):
        p=Path("runs/learned_local_global_v1")/"seed_7201"/f"d_{d}"/"system.json"
        sys=ContinuousReLUSystem.from_dict(read_json(p))
        status,bins,cert=solve(sys)
        print(d,status,bins,cert["status"] if cert else None,cert["upper_bound"] if cert else None,flush=True)

