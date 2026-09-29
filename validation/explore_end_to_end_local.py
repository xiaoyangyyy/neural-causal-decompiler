"""Development probe: train local ReLU directions, thresholds and output weights."""
from __future__ import annotations
import numpy as np
import torch
from ncd.trained_nonlinear_realization import NonlinearTrainingConfig, _samples, phase_witnesses, candidate_relation
from ncd.continuous_scale import _tuple_network
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import certify_weighted


def train(seed=7101, d=8, epochs=600):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    config=NonlinearTrainingConfig(seed=seed,state_dim=d,train_samples=2048,selection_samples=512,test_samples=1024)
    tx,ta,ty,_=_samples(config,1,config.train_samples)
    vx,va,vy,_=_samples(config,2,config.selection_samples)
    ex,ea,ey,_=_samples(config,3,config.test_samples)
    def local(x,a):
        return torch.stack((x,torch.roll(x,1,dims=1),a[:,0,None].expand_as(x),a[:,1,None].expand_as(x)),dim=-1)
    T=local(torch.tensor(tx,dtype=torch.float64),torch.tensor(ta,dtype=torch.float64))
    V=local(torch.tensor(vx,dtype=torch.float64),torch.tensor(va,dtype=torch.float64))
    E=local(torch.tensor(ex,dtype=torch.float64),torch.tensor(ea,dtype=torch.float64))
    target=torch.tensor(ty,dtype=torch.float64)
    val_target=torch.tensor(vy,dtype=torch.float64)
    base=np.array([[-1,1,0,0],[0,0,1,-1]],dtype=float)
    W1=torch.nn.Parameter(torch.tensor(np.tile(base[None,:,:],(d,1,1))+np.random.default_rng(seed).normal(0,.03,(d,2,4)),dtype=torch.float64))
    b1=torch.nn.Parameter(torch.tensor(np.tile(np.array([-.03,-.06]),(d,1)),dtype=torch.float64))
    W2=torch.nn.Parameter(torch.tensor(np.tile(np.array([.025,.015]),(d,1)),dtype=torch.float64))
    direct=torch.nn.Parameter(torch.tensor(np.tile(np.array([.25,.05,.11,-.04]),(d,1)),dtype=torch.float64))
    bias=torch.nn.Parameter(torch.full((d,),.08,dtype=torch.float64))
    params=[W1,b1,W2,direct,bias]
    optimizer=torch.optim.Adam(params,lr=.02)
    best=None
    best_val=float('inf')
    def predict(X):
        z=torch.einsum('bdf,dhf->bdh',X,W1)+b1
        return (torch.relu(z)*W2).sum(-1)+(X*direct).sum(-1)+bias
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=((predict(T)-target)**2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val=float(torch.sqrt(((predict(V)-val_target)**2).mean()))
            if val<best_val:
                best_val=val
                best=[p.detach().clone() for p in params]
    with torch.no_grad():
        for p,q in zip(params,best): p.copy_(q)
        test_error=predict(E).numpy()-ey
    h=d+2+2*d
    first=np.zeros((h,d+2))
    first[:d+2,:]=np.eye(d+2)
    first_bias=np.zeros(h)
    second=np.zeros((d,h))
    second_bias=bias.detach().numpy().copy()
    for i in range(d):
        second[i,i]=direct.detach().numpy()[i,0]
        second[i,(i-1)%d]+=direct.detach().numpy()[i,1]
        second[i,d]=direct.detach().numpy()[i,2]
        second[i,d+1]=direct.detach().numpy()[i,3]
        for k in range(2):
            row=d+2+2*i+k
            first[row,i]=W1.detach().numpy()[i,k,0]
            first[row,(i-1)%d]=W1.detach().numpy()[i,k,1]
            first[row,d]=W1.detach().numpy()[i,k,2]
            first[row,d+1]=W1.detach().numpy()[i,k,3]
            first_bias[row]=b1.detach().numpy()[i,k]
            second[i,row]=W2.detach().numpy()[i,k]
    system=ContinuousReLUSystem(d,2,_tuple_network([first,second],[first_bias,second_bias]),_tuple_network([np.eye(4,d)],[np.zeros(4)]),("demand","signal"))
    bins,radii=candidate_relation(d)
    cert=certify_weighted(system,bins,radii,action_bins=128,epsilon="0.17",packing_axes=4)
    return system,dict(val_rmse=best_val,test_rmse=float(np.sqrt(np.mean(test_error**2))),test_max=float(np.max(np.abs(test_error))),cert=cert["status"],upper=cert["upper_bound"],phase=phase_witnesses(system))


if __name__=="__main__":
    for d in (8,32):
        _,result=train(d=d)
        print(d,result)

