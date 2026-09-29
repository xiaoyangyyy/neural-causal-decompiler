"""Exact minimal initial coordinate grid for a fixed recurrent simulation.

Direct observations force three bins per observed axis. Either the 81-state
baseline or a 108-state single-axis refinement closes the exact handoff.
"""
from __future__ import annotations
from fractions import Fraction as Q
from math import prod

from .affine_observability import _affine_network
from .continuous_compositional_realization import (
    _quantize,sensitivity,value,verify_weighted)
from .continuous_separation import ContinuousReLUSystem,_digest
from .reachable_two_stage import _center,_reachable_indices


def _bounds(system,recurrent,bins):
    half=[Q(1,2*n) for n in bins]
    output=tuple(sensitivity(system.observation,half))
    handoff=tuple(a+Q(1,2*n) for a,n in zip(
        sensitivity(system.transition,half+[Q(1,2*recurrent['action_bins'])]*system.action_dim),
        recurrent['coordinate_bins']))
    epsilon=Q(recurrent['epsilon'])
    radii=tuple(Q(x) for x in recurrent['coordinate_radii'])
    valid=all(x<=epsilon for x in output) and all(x<=r for x,r in zip(handoff,radii))
    return output,handoff,valid


def _proof(system: ContinuousReLUSystem,recurrent: dict):
    previous=verify_weighted(system,recurrent)
    if previous['status']!='certified':
        raise ValueError('Certified recurrent relation required')
    if system.state_dim<5 or Q(recurrent['epsilon'])!=Q(17,100):
        raise ValueError('This exact initial-grid theorem requires epsilon=17/100')
    observation,bias,_=_affine_network(system.observation)
    d=system.state_dim
    identity=tuple(tuple(Q(int(i==j)) for j in range(d)) for i in range(4))
    if observation!=identity or any(bias):
        raise ValueError('Four direct observed coordinates required')
    baseline=[1]*d
    baseline[:4]=[3]*4
    baseline=tuple(baseline)
    baseline_output,baseline_handoff,baseline_valid=_bounds(system,recurrent,baseline)
    candidates=[baseline]
    for axis in range(4):
        trial=list(baseline)
        trial[axis]=4
        candidates.append(tuple(trial))
    feasible=[]
    for bins in candidates:
        output,handoff,valid=_bounds(system,recurrent,bins)
        if valid:feasible.append((prod(bins),bins,output,handoff))
    if not feasible:
        raise ValueError('No certified initial grid at the 81/108 theorem boundary')
    count,bins,output,handoff=min(feasible,key=lambda x:(x[0],x[1]))
    if count not in (81,108) or (count==81)!=baseline_valid:
        raise ValueError('Invalid exact initial-grid optimality branch')
    radii=tuple(Q(x) for x in recurrent['coordinate_radii'])
    exclusions=[i for i,(x,r) in enumerate(zip(baseline_handoff,radii)) if x>r]
    if count==108 and not exclusions:
        raise ValueError('Missing 81-state handoff exclusion')
    ranges=_reachable_indices(recurrent)
    rec_count=prod(b-a+1 for a,b in ranges)
    return {
        'schema':'ncd.optimal-initial-grid.v1',
        'system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent),
        'epsilon':recurrent['epsilon'],
        'state_dim':d,'action_dim':system.action_dim,
        'minimum_observed_axis_bins':[3]*4,
        'baseline_output_error_upper':[str(x) for x in baseline_output],
        'initial_coordinate_bins':list(bins),
        'recurrent_coordinate_bins':recurrent['coordinate_bins'],
        'action_bins':recurrent['action_bins'],
        'initial_output_error_upper':[str(x) for x in output],
        'initial_to_recurrent_error_upper':[str(x) for x in handoff],
        'recurrent_relation_radii':[str(x) for x in radii],
        'recurrent_index_low':[a for a,b in ranges],
        'recurrent_index_high':[b for a,b in ranges],
        'initial_state_count':count,
        'recurrent_state_count':rec_count,
        'upper_bound':count+rec_count,
        'direct_observation_initial_grid_lower':81,
        'baseline_initial_state_count':81,
        'baseline_handoff_excluded_coordinates':exclusions,
        'baseline_handoff_error_upper':[str(x) for x in baseline_handoff],
        'exact_minimum_initial_grid_count':count,
        'initial_optimality_scope':'all positive integer coordinate bins satisfying output tolerance and exact sensitivity handoff into this fixed recurrent relation',
        'initial_optimality_reason':'each direct observed coordinate needs at least 3 bins; below 108 states the only possible grid is 81 with all unobserved bins equal to 1',
        'scope':'full unit initial/action cubes and every finite horizon',
        'status':'certified',
    }


def certify_initial_grid(system: ContinuousReLUSystem,recurrent: dict):
    certificate=_proof(system,recurrent)
    verify_initial_grid(system,recurrent,certificate)
    return certificate


def verify_initial_grid(system: ContinuousReLUSystem,recurrent: dict,certificate: dict):
    if certificate.get('schema')!='ncd.optimal-initial-grid.v1':
        raise ValueError('Unsupported initial-grid certificate')
    expected=_proof(system,recurrent)
    if certificate!=expected:
        raise ValueError('Initial-grid replay mismatch')
    return {'status':'verified','initial_state_count':expected['initial_state_count'],
            'exact_minimum_initial_grid_count':expected['exact_minimum_initial_grid_count'],
            'recurrent_state_count':expected['recurrent_state_count'],
            'upper_bound':expected['upper_bound']}


def _state_bins(system,recurrent,certificate,state):
    if not isinstance(state,tuple) or len(state)!=2:
        raise ValueError('Malformed initial-grid machine state')
    phase,indices=state
    if not isinstance(indices,(tuple,list)) or len(indices)!=system.state_dim:
        raise ValueError('Malformed grid indices')
    if phase=='initial':
        bins=tuple(certificate['initial_coordinate_bins'])
        ranges=tuple((0,n-1) for n in bins)
    elif phase=='recurrent':
        bins=tuple(recurrent['coordinate_bins'])
        ranges=_reachable_indices(recurrent)
    else:
        raise ValueError('Unknown grid phase')
    if not all(isinstance(q,int) and not isinstance(q,bool) and a<=q<=b
               for q,(a,b) in zip(indices,ranges)):
        raise ValueError('State outside certified initial/recurrent grid')
    return bins


def initial_grid_initial(system: ContinuousReLUSystem,certificate: dict,point):
    if len(point)!=system.state_dim or any(not 0<=Q(x)<=1 for x in point):
        raise ValueError('Initial point outside unit cube')
    bins=certificate['initial_coordinate_bins']
    return ('initial',tuple(_quantize(Q(x),n) for x,n in zip(point,bins)))


def initial_grid_output(system: ContinuousReLUSystem,recurrent: dict,certificate: dict,state):
    bins=_state_bins(system,recurrent,certificate,state)
    return tuple(value(system.observation,_center(state[1],bins)))


def initial_grid_step(system: ContinuousReLUSystem,recurrent: dict,certificate: dict,state,action):
    bins=_state_bins(system,recurrent,certificate,state)
    if len(action)!=system.action_dim or any(not 0<=Q(x)<=1 for x in action):
        raise ValueError('Action outside unit cube')
    m=recurrent['action_bins']
    controls=[Q(2*_quantize(Q(a),m)+1,2*m) for a in action]
    image=value(system.transition,_center(state[1],bins)+controls)
    target=('recurrent',tuple(_quantize(x,n) for x,n in zip(image,recurrent['coordinate_bins'])))
    _state_bins(system,recurrent,certificate,target)
    return target
