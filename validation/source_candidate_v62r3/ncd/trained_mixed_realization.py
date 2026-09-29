"""All allowed hidden/output parameters learn mixed local neural dynamics.

The architecture has two state inputs and two controls per scalar output.
Projected path norms and box-image constraints are part of the frozen
optimizer protocol, not teacher features supplied to the certificate.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib
import numpy as np
import torch
from .continuous_scale import _tuple_network
from .continuous_separation import ContinuousReLUSystem,_digest
from .continuous_compositional_realization import certify_weighted
from .trained_nonlinear_realization import candidate_relation
from .io import save_json


@dataclass
class MixedTrainingConfig:
    seed:int=7101
    state_dim:int=8
    hidden_width:int=4
    train_samples:int=2048
    selection_samples:int=512
    test_samples:int=1024
    epochs:int=600
    learning_rate:float=0.01

    def validate(self):
        if (self.state_dim<6 or self.hidden_width!=4 or self.train_samples<128
            or self.selection_samples<64 or self.test_samples<64 or self.epochs<1
            or self.learning_rate<=0):
            raise ValueError('Invalid mixed-network training configuration')


def teacher_step(states,actions):
    d=states.shape[-1]
    rng=np.random.default_rng(82000+d)
    constant=rng.uniform(.127,.133,d)
    alpha=rng.uniform(.235,.245,d)
    beta=rng.uniform(.035,.043,d)
    demand=rng.uniform(.095,.105,d)
    signal=rng.uniform(.032,.038,d)
    first=rng.uniform(.013,.017,d)
    second=rng.uniform(.010,.014,d)
    prev=np.roll(states,1,axis=-1)
    a,b=actions[:,0,None],actions[:,1,None]
    return (constant+alpha*states+beta*prev+demand*a-signal*b
            +first*np.maximum(states+.5*a-.65,0)
            +second*np.maximum(prev-states+.4*a-.3*b-.1,0))


def samples(config,tag,count):
    rng=np.random.default_rng(config.seed*100+tag)
    x=rng.uniform(0,1,(count,config.state_dim))
    u=rng.uniform(0,1,(count,2))
    y=teacher_step(x,u)
    local=np.stack((x,np.roll(x,1,axis=1),np.broadcast_to(u[:,0,None],x.shape),np.broadcast_to(u[:,1,None],x.shape)),axis=-1)
    digest=hashlib.sha256(x.tobytes()+u.tobytes()+y.tobytes()).hexdigest()
    return local,y,digest,np.concatenate((x,u),axis=1)


def _forward(parameters,local):
    first,bias,output,constant=parameters
    hidden=torch.relu(torch.einsum('ndk,dhk->ndh',local,first)+bias)
    return (hidden*output).sum(-1)+constant


def _project(parameters):
    first,bias,output,constant=parameters
    with torch.no_grad():
        limits=torch.tensor([.29,.06,.13,.06],dtype=torch.float64)
        norm=(first.abs()*output.abs().unsqueeze(-1)).sum(1)
        scale=torch.minimum(torch.ones_like(norm),limits/norm.clamp_min(1e-12))
        first.mul_(scale.unsqueeze(1))
        low=torch.relu(bias+torch.minimum(first,torch.zeros_like(first)).sum(-1))
        high=torch.relu(bias+torch.maximum(first,torch.zeros_like(first)).sum(-1))
        lower=(torch.where(output>=0,output*low,output*high)).sum(-1)
        upper=(torch.where(output>=0,output*high,output*low)).sum(-1)
        lower_bias=.01-lower
        upper_bias=.99-upper
        if not bool((lower_bias<=upper_bias).all()):
            raise RuntimeError('Projected model has no invariant output-bias interval')
        constant.copy_(torch.maximum(lower_bias,torch.minimum(upper_bias,constant)))


def _export(parameters,d):
    first,bias,output,constant=[p.detach().numpy() for p in parameters]
    width=first.shape[1]
    weights=np.zeros((d*width,d+2))
    outputs=np.zeros((d,d*width))
    for i in range(d):
        for h in range(width):
            row=i*width+h
            weights[row,i]=first[i,h,0]
            weights[row,(i-1)%d]=first[i,h,1]
            weights[row,d:]=first[i,h,2:]
            outputs[i,row]=output[i,h]
    transition=_tuple_network([weights,outputs],[bias.ravel(),constant])
    observation=_tuple_network([np.eye(4,d)],[np.zeros(4)])
    return ContinuousReLUSystem(d,2,transition,observation,('demand','signal'))


def train_mixed(config):
    config.validate()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generator=torch.Generator().manual_seed(config.seed+1000*config.state_dim)
    d,h=config.state_dim,config.hidden_width
    first=torch.rand((d,h,4),generator=generator,dtype=torch.float64)-.5
    bias=torch.rand((d,h),generator=generator,dtype=torch.float64)*.5-.25
    bias[:,:2]=.8
    output=torch.randn((d,h),generator=generator,dtype=torch.float64)*.02
    output[:,:2]=torch.rand((d,2),generator=generator,dtype=torch.float64)*.05+.05
    constant=torch.full((d,),.13,dtype=torch.float64)
    parameters=[torch.nn.Parameter(p) for p in (first,bias,output,constant)]
    _project(parameters)
    initial=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.Adam(parameters,lr=config.learning_rate)
    train,selection,test=[samples(config,tag,count) for tag,count in ((1,config.train_samples),(2,config.selection_samples),(3,config.test_samples))]
    rows=[{row.tobytes() for row in item[3]} for item in (train,selection,test)]
    if any(rows[i]&rows[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Training, selection and test inputs overlap')
    tx,ty=torch.from_numpy(train[0]),torch.from_numpy(train[1])
    history=[]
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        error=_forward(parameters,tx)-ty
        loss=error.square().mean()
        loss.backward()
        optimizer.step()
        _project(parameters)
        if epoch in (0,config.epochs-1):
            history.append({'epoch':epoch+1,'training_mse':float(loss.detach())})
    def metrics(item):
        with torch.no_grad():
            error=(_forward(parameters,torch.from_numpy(item[0]))-torch.from_numpy(item[1])).numpy()
        return {'rmse':float(np.sqrt(np.mean(error**2))),'max_error':float(np.max(np.abs(error)))}
    system=_export(parameters,d)
    changes=[]
    for name,before,after in zip(('hidden_weights','hidden_biases','output_weights','output_biases'),initial,parameters):
        delta=(after.detach()-before).abs()
        changes.append({'parameter_group':name,'allowed_parameter_count':delta.numel(),
            'changed_parameter_count':int((delta>1e-12).sum()),'maximum_change':float(delta.max())})
    checkpoint={name:p.detach().numpy() for name,p in zip(('hidden_weights','hidden_biases','output_weights','output_biases'),parameters)}
    record={'schema':'ncd.mixed-all-parameter-training.v1','config':asdict(config),
        'architecture':'four local hidden ReLUs per output; two state supports and two controls; all allowed transition parameters train',
        'projection':{'absolute_path_norm_limits':[.29,.06,.13,.06],'interval_image_margin':.01},
        'optimizer':'Adam, float64, full batch, deterministic CPU, fixed final epoch',
        'torch_version':torch.__version__,'numpy_version':np.__version__,
        'teacher_used_only_for_sample_targets':True,'splits_disjoint':True,
        'split_sha256':{name:item[2] for name,item in zip(('train','selection','test'),(train,selection,test))},
        'checkpoint_selection':'fixed final epoch; no test-based selection',
        'training':metrics(train),'selection':metrics(selection),'test':metrics(test),
        'optimization_history':history,'parameter_changes':changes,
        'system_sha256':_digest(system.to_dict())}
    return system,record,checkpoint


def run_training(config,target):
    target=Path(target)
    target.mkdir(parents=True,exist_ok=True)
    system,record,checkpoint=train_mixed(config)
    save_json(target/'system.json',system.to_dict())
    save_json(target/'training.json',record)
    np.savez(target/'checkpoint.npz',**checkpoint)
    bins,radii=candidate_relation(config.state_dim)
    recurrent=certify_weighted(system,bins,radii,action_bins=128,epsilon='17/100',packing_axes=4)
    save_json(target/'recurrent_certificate.json',recurrent)
    return system,record,recurrent


def mixed_witness(system,threshold='1/10000'):
    """Exact rectangle difference from actual observed neural outputs."""
    from fractions import Fraction as Q
    from .joint_phase_initial_handoff import _extract
    from .rational_polytope import evaluate
    grid=tuple(Q(j,4) for j in range(5))
    fixed=(Q(0),Q(1,2),Q(1))
    attempts=0
    for coordinate in range(4):
        axes,padded,terms,mixed=_extract(system,coordinate)
        for action_axis in range(2):
            for previous in fixed:
                for other in fixed:
                    for a,b in zip(grid,grid[1:]):
                        for c,d in zip(grid,grid[1:]):
                            values=[]
                            points=[]
                            for state_value,action_value in ((a,c),(a,d),(b,c),(b,d)):
                                state=[Q(0)]*system.state_dim
                                state[coordinate]=state_value
                                state[(coordinate-1)%system.state_dim]=previous
                                action=[other,other]
                                action[action_axis]=action_value
                                local=tuple(state[k] if k is not None else Q(0) for k in padded)+tuple(action)
                                result=Q(system.transition.biases[-1][coordinate])+sum((w*max(Q(0),evaluate(form,local)) for w,form in terms),Q(0))
                                values.append(result)
                                points.append(state+action)
                            difference=values[0]+values[3]-values[1]-values[2]
                            attempts+=1
                            if abs(difference)>Q(threshold):
                                return {'observed_coordinate':coordinate,'state_axis':coordinate,
                                    'control_axis':action_axis,'four_inputs':[[str(x) for x in point] for point in points],
                                    'four_scalar_outputs':[str(x) for x in values],
                                    'mixed_rectangle_difference':str(difference),'minimum_absolute_difference':threshold,
                                    'rectangles_checked':attempts,'mixed_hidden_units':mixed}
    raise ValueError('No observed mixed state/control rectangle witness found')
