"""Development probe for automatic coordinate-grid synthesis from frozen ReLU weights."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np

from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import certify_weighted
from ncd.io import read_json


def absolute_jacobian_bound(network):
    result=np.eye(network.input_dim)
    for weight in network.weights:
        result=np.abs(np.asarray(weight,dtype=float))@result
    return result


def propose(system,epsilon=.17,action_bins=128,max_bin=32):
    d=system.state_dim
    matrix=absolute_jacobian_bound(system.transition)
    state=matrix[:,:d]
    action=matrix[:,d:]
    observation=absolute_jacobian_bound(system.observation)
    transfer=np.linalg.solve(np.eye(d)-state,np.eye(d))
    impact=observation@transfer
    action_error=action.sum(axis=1)/(2*action_bins)
    base=impact@action_error
    bins=np.ones(d,dtype=int)
    q=1/(2*bins)
    output=base+impact@q
    target=epsilon*.98
    steps=[]
    while np.max(output)>target:
        deficit=np.maximum(output-target,0)
        best=None
        for j in range(d):
            if bins[j]>=max_bin: continue
            change=(1/(2*bins[j])-1/(2*(bins[j]+1)))*impact[:,j]
            gain=np.minimum(deficit,change).sum()
            cost=math.log((bins[j]+1)/bins[j])
            score=gain/cost
            if best is None or score>best[0]:
                best=(score,j,change)
        if best is None or best[0]<=0:
            raise RuntimeError(f"No candidate bin increment; max output bound {max(output)}")
        _,j,change=best
        bins[j]+=1
        output-=change
        steps.append(int(j))
    for j in reversed(steps):
        if bins[j]>1:
            change=(1/(2*(bins[j]-1))-1/(2*bins[j]))*impact[:,j]
            if np.max(output+change)<=target:
                bins[j]-=1
                output+=change
    q=1/(2*bins)
    radii=transfer@(action_error+q)
    radii=np.ceil(radii*1.005*1e8)/1e8
    return tuple(map(int,bins)),tuple(f"{x:.8f}" for x in radii),output,steps


if __name__=="__main__":
    for d in (8,32,64,128):
        path=Path("runs/learned_local_global_v1")/"seed_7201"/f"d_{d}"/"system.json"
        system=ContinuousReLUSystem.from_dict(read_json(path))
        bins,radii,output,steps=propose(system)
        cert=certify_weighted(system,bins,radii,action_bins=128,epsilon="0.17",packing_axes=4)
        print(d,cert["status"],cert["upper_bound"],"grid",bins,"max_output",max(output),"steps",len(steps),flush=True)

