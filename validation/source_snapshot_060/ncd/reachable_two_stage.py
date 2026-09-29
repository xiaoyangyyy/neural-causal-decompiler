"""Exact two-stage finite realization with reachable recurrent states.

A small initial grid covers the full unit cube. Every first transition
enters the invariant reachable subset of an existing weighted grid.
"""
from __future__ import annotations

from fractions import Fraction as Q
from math import prod

from .continuous_compositional_realization import (
    _quantize, sensitivity, value, verify_weighted)
from .continuous_separation import ContinuousReLUSystem, _digest


def initial_bins(dimension: int) -> tuple[int, ...]:
    if dimension < 5:
        raise ValueError('Five distinct active coordinates required')
    bins=[1]*dimension
    for i in (0,1,2,3,dimension-1):
        bins[i]=3
    return tuple(bins)


def _reachable_indices(recurrent_certificate: dict):
    ranges=[]
    for n,raw_low,raw_high in zip(
            recurrent_certificate['coordinate_bins'],
            recurrent_certificate['transition_image_low'],
            recurrent_certificate['transition_image_high']):
        low,high=Q(raw_low),Q(raw_high)
        if not 0<=low<=high<=1:
            raise ValueError('Invalid full-cube transition image enclosure')
        left=_quantize(low,n)
        right=_quantize(high,n)
        if left>right:
            raise ValueError('Empty recurrent index range')
        ranges.append((left,right))
    return tuple(ranges)


def _center(indices, bins):
    return [Q(2*q+1,2*n) for q,n in zip(indices,bins)]


def two_stage_initial(system: ContinuousReLUSystem, point):
    bins=initial_bins(system.state_dim)
    if len(point)!=system.state_dim or any(
            not 0<=Q(x)<=1 for x in point):
        raise ValueError('Initial point outside full unit cube')
    return ('initial',tuple(_quantize(Q(x),n) for x,n in zip(point,bins)))


def _validate_state(system, recurrent_certificate, state):
    if not isinstance(state,tuple) or len(state)!=2:
        raise ValueError('Malformed two-stage state')
    phase,indices=state
    if phase=='initial':
        bins=initial_bins(system.state_dim)
        valid=all(isinstance(q,int) and not isinstance(q,bool)
                  and 0<=q<n for q,n in zip(indices,bins))
    elif phase=='recurrent':
        bins=tuple(recurrent_certificate['coordinate_bins'])
        ranges=_reachable_indices(recurrent_certificate)
        valid=all(isinstance(q,int) and not isinstance(q,bool)
                  and left<=q<=right
                  for q,(left,right) in zip(indices,ranges))
    else:
        raise ValueError('Unknown two-stage phase')
    if len(indices)!=system.state_dim or not valid:
        raise ValueError('Two-stage state outside its declared grid')
    return bins


def two_stage_output(system: ContinuousReLUSystem,
                     recurrent_certificate: dict, state):
    bins=_validate_state(system,recurrent_certificate,state)
    return tuple(value(system.observation,_center(state[1],bins)))


def two_stage_step(system: ContinuousReLUSystem,
                   recurrent_certificate: dict, state, action):
    bins=_validate_state(system,recurrent_certificate,state)
    m=recurrent_certificate['action_bins']
    if len(action)!=system.action_dim or any(
            not 0<=Q(a)<=1 for a in action):
        raise ValueError('Action outside full unit cube')
    controls=[Q(2*_quantize(Q(a),m)+1,2*m) for a in action]
    image=value(system.transition,_center(state[1],bins)+controls)
    target_bins=recurrent_certificate['coordinate_bins']
    target=('recurrent',tuple(_quantize(x,n) for x,n in zip(image,target_bins)))
    _validate_state(system,recurrent_certificate,target)
    return target


def _proof(system: ContinuousReLUSystem, recurrent_certificate: dict):
    prior=verify_weighted(system,recurrent_certificate)
    if prior['status']!='certified' or prior['upper_bound'] is None:
        raise ValueError('Certified recurrent weighted grid required')
    d=system.state_dim
    initial=initial_bins(d)
    recurrent=tuple(recurrent_certificate['coordinate_bins'])
    if len(recurrent)!=d or any(not isinstance(n,int) or n<1 for n in recurrent):
        raise ValueError('Malformed recurrent coordinate bins')
    m=recurrent_certificate['action_bins']
    epsilon=Q(recurrent_certificate['epsilon'])
    radii=tuple(Q(x) for x in recurrent_certificate['coordinate_radii'])
    initial_half=[Q(1,2*n) for n in initial]
    recurrent_half=[Q(1,2*n) for n in recurrent]
    initial_output=sensitivity(system.observation,initial_half)
    handoff=tuple(delta+half for delta,half in zip(
        sensitivity(system.transition,
                    initial_half+[Q(1,2*m)]*system.action_dim),
        recurrent_half))
    if (any(error>epsilon for error in initial_output)
            or any(error>radius for error,radius in zip(handoff,radii))):
        raise ValueError('Initial stage does not simulate into recurrent grid')
    ranges=_reachable_indices(recurrent_certificate)
    initial_count=prod(initial)
    recurrent_count=prod(right-left+1 for left,right in ranges)
    total=initial_count+recurrent_count
    return {
        'schema':'ncd.reachable-two-stage.v1',
        'system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent_certificate),
        'epsilon':str(epsilon),
        'state_dim':d,'action_dim':system.action_dim,
        'initial_coordinate_bins':list(initial),
        'recurrent_coordinate_bins':list(recurrent),
        'recurrent_action_bins':m,
        'initial_output_error_upper':[str(x) for x in initial_output],
        'initial_to_recurrent_error_upper':[str(x) for x in handoff],
        'recurrent_relation_radii':[str(x) for x in radii],
        'recurrent_index_low':[left for left,_ in ranges],
        'recurrent_index_high':[right for _,right in ranges],
        'initial_state_count':initial_count,
        'recurrent_state_count':recurrent_count,
        'upper_bound':total,
        'scope':'full unit initial and action cubes, every finite horizon',
        'machine':'initial-grid states transition once into the invariant reachable subset of the weighted recurrent grid',
        'status':'certified',
    }


def certify_two_stage(system: ContinuousReLUSystem,
                      recurrent_certificate: dict):
    certificate=_proof(system,recurrent_certificate)
    verify_two_stage(system,recurrent_certificate,certificate)
    return certificate


def verify_two_stage(system: ContinuousReLUSystem,
                     recurrent_certificate: dict,certificate: dict):
    if certificate.get('schema')!='ncd.reachable-two-stage.v1':
        raise ValueError('Unsupported two-stage certificate')
    expected=_proof(system,recurrent_certificate)
    if certificate!=expected:
        raise ValueError('Two-stage certificate replay mismatch')
    return {'status':'verified',
            'initial_state_count':expected['initial_state_count'],
            'recurrent_state_count':expected['recurrent_state_count'],
            'upper_bound':expected['upper_bound']}
