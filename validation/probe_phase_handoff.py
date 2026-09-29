"""Exploratory phase-aware handoff for an 81-cell initial grid."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json,save_json,digest
from ncd.nonlinear_action_closure import SQUARE,_clip,_evaluate

ROOT=Path(__file__).resolve().parents[1]


def scalar_pieces(terms,rectangle,bias):
    polygons=[rectangle]
    crossing=sorted({f for w,f in terms
        if min(_evaluate(f,p) for p in rectangle)<0 and max(_evaluate(f,p) for p in rectangle)>0})
    for form in crossing:
        split=[]
        for poly in polygons:
            scores=[_evaluate(form,p) for p in poly]
            if min(scores)>=0 or max(scores)<=0:
                split.append(poly)
            else:
                split.extend((_clip(poly,form),_clip(poly,tuple(-x for x in form))))
        polygons=split
    result=[]
    for poly in polygons:
        center=tuple(sum((p[k] for p in poly),Q(0))/len(poly) for k in range(2))
        output=[Q(0),Q(0),bias]
        for weight,form in terms:
            scores=[_evaluate(form,p) for p in poly]
            if _evaluate(form,center)>=0:
                if min(scores)<0:raise ValueError('State phase mismatch')
                for k,x in enumerate(form):output[k]+=weight*x
            elif max(scores)>0:
                raise ValueError('State phase mismatch')
        result.append((poly,tuple(output)))
    return tuple(result)


def main():
    source=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/'d_128'
    model=source/'system.json'
    system=ContinuousReLUSystem.from_dict(read_json(model))
    recurrent=read_json(source/'certificate.json')
    assert verify_weighted(system,recurrent)['status']=='certified'
    net=system.transition
    rows=[]
    for i in range(system.state_dim):
        hidden=[j for j,w in enumerate(net.weights[1][i]) if w]
        axes=sorted({k for j in hidden for k,w in enumerate(net.weights[0][j][:-2]) if w})
        if len(axes)!=2:raise ValueError('Two state inputs per output required for this probe')
        state_terms=[];action_terms=[]
        for j in hidden:
            raw=net.weights[0][j]
            weight=Q(net.weights[1][i][j])
            bias=Q(net.biases[0][j])
            state=any(raw[:-2]);action=any(raw[-2:])
            if state and action:raise ValueError('Mixed state/action hidden unit unsupported by probe')
            if action:
                action_terms.append((weight,(Q(raw[-2]),Q(raw[-1]),bias)))
            else:
                state_terms.append((weight,(Q(raw[axes[0]]),Q(raw[axes[1]]),bias)))
        action_pieces=scalar_pieces(action_terms,SQUARE,Q(0))
        action_lipschitz=max(abs(a)+abs(b) for poly,(a,b,c) in action_pieces)
        bins=tuple(3 if k<4 else 1 for k in axes)
        maximum=Q(0)
        for indices in product(*(range(n) for n in bins)):
            low=tuple(Q(q,n) for q,n in zip(indices,bins))
            high=tuple(Q(q+1,n) for q,n in zip(indices,bins))
            center=tuple(Q(2*q+1,2*n) for q,n in zip(indices,bins))
            rectangle=((low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1]))
            bias=Q(net.biases[1][i])
            at_center=bias+sum((w*max(Q(0),_evaluate(f,center)) for w,f in state_terms),Q(0))
            pieces=scalar_pieces(state_terms,rectangle,bias)
            maximum=max(maximum,max(abs(_evaluate(form,p)-at_center) for poly,form in pieces for p in poly))
        bound=maximum+action_lipschitz/Q(2*recurrent['action_bins'])+Q(1,2*recurrent['coordinate_bins'][i])
        radius=Q(recurrent['coordinate_radii'][i])
        rows.append({'coordinate':i,'state_axes':axes,'initial_state_cell_error':str(maximum),
            'action_lipschitz':str(action_lipschitz),'handoff_error_upper':str(bound),
            'recurrent_radius':str(radius),'slack':str(radius-bound)})
    margin=min(Q(r['slack']) for r in rows)
    report={'status':'exploratory','model_sha256':digest(model),'state_dim':128,'seed':6101,
        'initial_states':81,'initial_output_error':'1/6','phase_handoff_all_coordinates_pass':margin>=0,
        'minimum_handoff_slack':str(margin),'rows':rows,
        'boundary':'exact exploratory bounds; not yet integrated into formal initial/closure/program certificates'}
    save_json(ROOT/'validation'/'phase_handoff_probe.json',report)
    print('phase-aware 81-cell handoff:',margin>=0,'minimum slack',margin,float(margin),flush=True)


if __name__=='__main__':main()
