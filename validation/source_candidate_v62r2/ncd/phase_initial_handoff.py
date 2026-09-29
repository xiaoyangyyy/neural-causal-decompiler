"""Exact phase-aware initial handoff into a certified recurrent relation.

The frozen network must separate state-only and action-only hidden units.
Each scalar output may depend on at most two state coordinates. Exact
state-cell extrema preserve signed weight cancellations before a global
piecewise-affine action gradient bound accounts for action quantization.
"""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import product
from math import prod

from .affine_observability import _affine_network
from .continuous_compositional_realization import verify_weighted
from .continuous_separation import ContinuousReLUSystem,_digest
from .nonlinear_action_closure import SQUARE,_clip,_evaluate
from .reachable_two_stage import _reachable_indices


def _observation(system):
    if system.state_dim<4:
        raise ValueError('Four direct observed coordinates required')
    matrix,bias,_=_affine_network(system.observation)
    identity=tuple(tuple(Q(int(i==j)) for j in range(system.state_dim)) for i in range(4))
    if matrix!=identity or any(bias):
        raise ValueError('Four direct observed coordinates with zero bias required')


def _decompose(system,coordinate):
    net=system.transition
    if system.action_dim!=2 or len(net.weights)!=2:
        raise ValueError('Two actions and one hidden ReLU layer required')
    relevant=tuple(j for j,w in enumerate(net.weights[1][coordinate]) if w)
    axes=tuple(sorted({k for j in relevant for k,w in enumerate(net.weights[0][j][:-2]) if w}))
    if len(axes)>2:
        raise ValueError('At most two state support axes per scalar output required')
    padded=axes+(None,)*(2-len(axes))
    state_terms=[]
    action_terms=[]
    for j in relevant:
        raw=net.weights[0][j]
        weight=Q(net.weights[1][coordinate][j])
        bias=Q(net.biases[0][j])
        state=any(raw[:-2])
        action=any(raw[-2:])
        if state and action:
            raise ValueError('Mixed state/action hidden unit is not separable')
        if action:
            action_terms.append((weight,(Q(raw[-2]),Q(raw[-1]),bias)))
        else:
            state_terms.append((weight,tuple(Q(raw[k]) if k is not None else Q(0) for k in padded)+(bias,)))
    return axes,padded,tuple(state_terms),tuple(action_terms),Q(net.biases[1][coordinate])


def _scalar_pieces(terms,rectangle,bias,cache):
    crossing=tuple(sorted({form for weight,form in terms
        if min(_evaluate(form,p) for p in rectangle)<0
        and max(_evaluate(form,p) for p in rectangle)>0}))
    key=(rectangle,crossing)
    if key not in cache:
        polygons=[rectangle]
        for form in crossing:
            split=[]
            for poly in polygons:
                scores=tuple(_evaluate(form,p) for p in poly)
                if min(scores)>=0 or max(scores)<=0:
                    split.append(poly)
                else:
                    first=_clip(poly,form)
                    second=_clip(poly,tuple(-x for x in form))
                    if not first or not second:
                        raise ValueError('State/action phase split lost a region')
                    split.extend((first,second))
            polygons=split
        cache[key]=tuple(polygons)
    pieces=[]
    for poly in cache[key]:
        center=tuple(sum((p[k] for p in poly),Q(0))/len(poly) for k in range(2))
        affine=[Q(0),Q(0),bias]
        for weight,form in terms:
            scores=tuple(_evaluate(form,p) for p in poly)
            if _evaluate(form,center)>=0:
                if min(scores)<0:
                    raise ValueError('Positive phase not fixed on polygon')
                for k,x in enumerate(form):
                    affine[k]+=weight*x
            elif max(scores)>0:
                raise ValueError('Negative phase not fixed on polygon')
        pieces.append((poly,tuple(affine)))
    return tuple(pieces)


def _fingerprint(pieces):
    return _digest([{'vertices':[[str(x) for x in p] for p in poly],
                     'affine_form':[str(x) for x in form]} for poly,form in pieces])


def _proof(system: ContinuousReLUSystem,recurrent: dict):
    _observation(system)
    if Q(recurrent['epsilon'])!=Q(17,100):
        raise ValueError('This initial-label theorem requires epsilon=17/100')
    prior=verify_weighted(system,recurrent)
    if prior['status']!='certified':
        raise ValueError('Certified recurrent weighted relation required')
    bins=tuple(3 if i<4 else 1 for i in range(system.state_dim))
    rows=[]
    cache={}
    for coordinate in range(system.state_dim):
        axes,padded,state_terms,action_terms,bias=_decompose(system,coordinate)
        action_pieces=_scalar_pieces(action_terms,SQUARE,Q(0),cache)
        action_lipschitz=max(abs(a)+abs(b) for poly,(a,b,c) in action_pieces)
        pair_bins=tuple(bins[k] if k is not None else 1 for k in padded)
        cells=[]
        maximum=Q(0)
        for indices in product(*(range(n) for n in pair_bins)):
            low=tuple(Q(q,n) for q,n in zip(indices,pair_bins))
            high=tuple(Q(q+1,n) for q,n in zip(indices,pair_bins))
            center=tuple(Q(2*q+1,2*n) for q,n in zip(indices,pair_bins))
            rectangle=((low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1]))
            at_center=bias+sum((w*max(Q(0),_evaluate(f,center)) for w,f in state_terms),Q(0))
            pieces=_scalar_pieces(state_terms,rectangle,bias,cache)
            values=tuple(_evaluate(form,p) for poly,form in pieces for p in poly)
            minimum,upper=min(values),max(values)
            if not minimum<=at_center<=upper:
                raise ValueError('Cell midpoint outside exact phase range')
            deviation=max(abs(minimum-at_center),abs(upper-at_center))
            maximum=max(maximum,deviation)
            cells.append({'indices':list(indices),'center_output':str(at_center),
                'minimum_output':str(minimum),'maximum_output':str(upper),
                'deviation':str(deviation),'phase_region_count':len(pieces),
                'phase_partition_sha256':_fingerprint(pieces)})
        control_error=action_lipschitz/Q(2*recurrent['action_bins'])
        half=Q(1,2*recurrent['coordinate_bins'][coordinate])
        bound=maximum+control_error+half
        radius=Q(recurrent['coordinate_radii'][coordinate])
        if bound>radius:
            raise ValueError(f'Phase-aware initial handoff fails coordinate {coordinate}')
        rows.append({'coordinate':coordinate,'state_axes':list(axes),
            'pair_coordinate_bins':list(pair_bins),'state_cells':cells,
            'state_cell_error_upper':str(maximum),
            'action_lipschitz':str(action_lipschitz),
            'action_region_count':len(action_pieces),
            'action_partition_sha256':_fingerprint(action_pieces),
            'control_quantization_error_upper':str(control_error),
            'recurrent_quantization_half':str(half),
            'initial_to_recurrent_error_upper':str(bound),
            'recurrent_relation_radius':str(radius),'handoff_slack':str(radius-bound)})
    ranges=_reachable_indices(recurrent)
    rec_count=prod(b-a+1 for a,b in ranges)
    return {'schema':'ncd.phase-initial-handoff.v1','status':'certified',
        'system_sha256':_digest(system.to_dict()),'recurrent_certificate_sha256':_digest(recurrent),
        'state_dim':system.state_dim,'action_dim':system.action_dim,
        'epsilon':recurrent['epsilon'],'action_bins':recurrent['action_bins'],
        'initial_coordinate_bins':list(bins),'recurrent_coordinate_bins':recurrent['coordinate_bins'],
        'initial_output_error_upper':['1/6']*4,
        'initial_to_recurrent_error_upper':[r['initial_to_recurrent_error_upper'] for r in rows],
        'recurrent_relation_radii':[str(Q(x)) for x in recurrent['coordinate_radii']],
        'recurrent_index_low':[a for a,b in ranges],'recurrent_index_high':[b for a,b in ranges],
        'initial_state_count':81,'recurrent_state_count':rec_count,'upper_bound':81+rec_count,
        'minimum_initial_label_count':81,
        'initial_packing':{'axis_values':['0','1/2','1'],'observed_axes':[0,1,2,3],
            'point_count':81,'minimum_output_separation':'1/2','twice_epsilon':'17/50',
            'scope':'any epsilon-accurate deterministic output machine initial selector'},
        'coordinate_proofs':rows,
        'state_pair_cells_checked':sum(len(r['state_cells']) for r in rows),
        'minimum_handoff_slack':str(min(Q(r['handoff_slack']) for r in rows)),
        'scope':'full unit initial/action cubes and every finite horizon through the fixed recurrent relation',
        'boundary':'minimum initialization label count 81; not the globally minimum whole machine or causal program'}


def certify_phase_handoff(system,recurrent):
    return _proof(system,recurrent)


def verify_phase_handoff(system,recurrent,certificate):
    if certificate.get('schema')!='ncd.phase-initial-handoff.v1':
        raise ValueError('Unsupported phase-aware initial certificate')
    expected=_proof(system,recurrent)
    if certificate!=expected:
        raise ValueError('Phase-aware initial handoff replay mismatch')
    return {'status':'verified','initial_state_count':81,
        'recurrent_state_count':expected['recurrent_state_count'],'upper_bound':expected['upper_bound'],
        'minimum_initial_label_count':81}
