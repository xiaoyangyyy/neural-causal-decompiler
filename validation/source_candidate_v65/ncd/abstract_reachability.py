"""Exact graph closure of the two-stage machine under all continuous actions.

Each outgoing edge range encloses all action-quantized transitions from one
abstract center. A least fixed point prunes the prior coordinate image box.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from fractions import Fraction as Q
from itertools import product

from .affine_observability import _affine_network
from .continuous_compositional_realization import _quantize
from .continuous_separation import ContinuousReLUSystem, _digest
from .reachable_two_stage import (
    initial_bins, two_stage_initial, two_stage_output, two_stage_step,
    verify_two_stage)


def _active_axes(initial, recurrent):
    if len(initial)!=len(recurrent):
        raise ValueError('Mismatched initial/recurrent dimensions')
    active=tuple(i for i,(a,b) in enumerate(zip(initial,recurrent))
                 if a>1 or b>1)
    if not active or len(active)>8 or any(
            initial[j]!=1 or recurrent[j]!=1
            for j in range(len(initial)) if j not in active):
        raise ValueError('Unsupported active-axis projection')
    return active


def _successor_forms(system: ContinuousReLUSystem, active):
    d=system.state_dim
    transition,offset,_=_affine_network(system.transition)
    forms=[]
    for i in active:
        row=transition[i]
        if len(row)!=d+system.action_dim:
            raise ValueError('Malformed exact affine transition')
        fixed=offset[i]+sum((row[j]*Q(1,2)
                            for j in range(d) if j not in active),Q(0))
        action_low=sum((min(Q(0),v) for v in row[d:]),Q(0))
        action_high=sum((max(Q(0),v) for v in row[d:]),Q(0))
        forms.append((tuple(row[j] for j in active),
                      fixed+action_low,fixed+action_high))
    return tuple(forms)


def _successors(indices, source_bins, target_bins, forms):
    ranges=[]
    for (coeff,low,high),n in zip(forms,target_bins):
        shift=sum((v*Q(2*q+1,2*m)
                   for v,q,m in zip(coeff,indices,source_bins)),Q(0))
        left=_quantize(low+shift,n)
        right=_quantize(high+shift,n)
        if left>right:
            raise ValueError('Empty action-box successor range')
        ranges.append(range(left,right+1))
    return product(*ranges)


def _proof(system: ContinuousReLUSystem,
           recurrent_certificate: dict,
           two_stage_certificate: dict):
    prior=verify_two_stage(system,recurrent_certificate,two_stage_certificate)
    if prior['status']!='verified' or prior['upper_bound'] is None:
        raise ValueError('A valid two-stage machine is required')
    initial=initial_bins(system.state_dim)
    recurrent=tuple(recurrent_certificate['coordinate_bins'])
    active=_active_axes(initial,recurrent)
    initial_active=tuple(initial[i] for i in active)
    recurrent_active=tuple(recurrent[i] for i in active)
    forms=_successor_forms(system,active)
    seed=set()
    for indices in product(*(range(n) for n in initial_active)):
        seed.update(_successors(indices,initial_active,recurrent_active,forms))
    seen=set(seed)
    queue=deque(sorted(seed))
    checked_edges=0
    while queue:
        state=queue.popleft()
        for target in _successors(state,recurrent_active,recurrent_active,forms):
            checked_edges+=1
            if target not in seen:
                seen.add(target)
                queue.append(target)
    retained=sorted(seen)
    outer_low=two_stage_certificate['recurrent_index_low']
    outer_high=two_stage_certificate['recurrent_index_high']
    if any(not outer_low[axis]<=row[k]<=outer_high[axis]
           for row in retained for k,axis in enumerate(active)):
        raise ValueError('Graph closure escaped the certified recurrent grid')
    initial_count=prior['initial_state_count']
    total=initial_count+len(retained)
    return {
        'schema':'ncd.abstract-action-box-closure.v1',
        'system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent_certificate),
        'two_stage_certificate_sha256':_digest(two_stage_certificate),
        'epsilon':recurrent_certificate['epsilon'],
        'state_dim':system.state_dim,
        'action_dim':system.action_dim,
        'active_axes':list(active),
        'initial_active_bins':list(initial_active),
        'recurrent_active_bins':list(recurrent_active),
        'action_enclosure':'full continuous unit cube; contains all quantized action centers',
        'initial_successor_count':len(seed),
        'closure_edge_count':checked_edges,
        'recurrent_active_indices':[list(row) for row in retained],
        'initial_state_count':initial_count,
        'recurrent_state_count':len(retained),
        'upper_bound':total,
        'scope':'all unit initial states, all continuous unit action words, every finite horizon',
        'status':'certified',
    }


def certify_abstract_closure(system: ContinuousReLUSystem,
                             recurrent_certificate: dict,
                             two_stage_certificate: dict):
    certificate=_proof(system,recurrent_certificate,two_stage_certificate)
    verify_abstract_closure(
        system,recurrent_certificate,two_stage_certificate,certificate)
    return certificate


def verify_abstract_closure(system: ContinuousReLUSystem,
                            recurrent_certificate: dict,
                            two_stage_certificate: dict,
                            certificate: dict):
    if certificate.get('schema')!='ncd.abstract-action-box-closure.v1':
        raise ValueError('Unsupported abstract closure certificate')
    expected=_proof(system,recurrent_certificate,two_stage_certificate)
    if certificate!=expected:
        raise ValueError('Abstract closure replay mismatch')
    return {'status':'verified',
            'initial_state_count':expected['initial_state_count'],
            'initial_successor_count':expected['initial_successor_count'],
            'recurrent_state_count':expected['recurrent_state_count'],
            'closure_edge_count':expected['closure_edge_count'],
            'upper_bound':expected['upper_bound']}


def _validate_graph_state(system, recurrent_certificate, certificate, state):
    if not isinstance(state,tuple) or len(state)!=2:
        raise ValueError('Malformed graph state')
    if state[0]=='recurrent':
        axes=certificate['active_axes']
        indices=state[1]
        if len(indices)!=system.state_dim:
            raise ValueError('Malformed recurrent graph indices')
        key=[indices[i] for i in axes]
        rows=certificate['recurrent_active_indices']
        position=bisect_left(rows,key)
        if position==len(rows) or rows[position]!=key:
            raise ValueError('Recurrent state not in certified graph closure')
    elif state[0]!='initial':
        raise ValueError('Unknown graph phase')
    # The original exact machine validates every coordinate and tag.
    two_stage_output(system,recurrent_certificate,state)


def graph_initial(system: ContinuousReLUSystem, point):
    return two_stage_initial(system,point)


def graph_output(system: ContinuousReLUSystem,
                 recurrent_certificate: dict,certificate: dict,state):
    _validate_graph_state(system,recurrent_certificate,certificate,state)
    return two_stage_output(system,recurrent_certificate,state)


def graph_step(system: ContinuousReLUSystem,
               recurrent_certificate: dict,certificate: dict,state,action):
    _validate_graph_state(system,recurrent_certificate,certificate,state)
    target=two_stage_step(system,recurrent_certificate,state,action)
    _validate_graph_state(system,recurrent_certificate,certificate,target)
    return target
