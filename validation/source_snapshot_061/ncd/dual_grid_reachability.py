"""Exact reachability co-design across state and action grid resolutions.

The jointly optimized 512-action-bin recurrent grid has fewer closed graph
states than the 128-action-bin grid despite a larger outer image rectangle.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from fractions import Fraction as Q
from itertools import product

from .abstract_reachability import _active_axes,_successor_forms,_successors
from .continuous_separation import ContinuousReLUSystem,_digest
from .correlated_reachability import (
    _could_reach,_forms_and_pairs,_source_pair_limits)
from .reachable_two_stage import (
    initial_bins,two_stage_initial,two_stage_output,two_stage_step,
    verify_two_stage)


def _proof(system: ContinuousReLUSystem,recurrent_certificate: dict,
           two_stage_certificate: dict):
    previous=verify_two_stage(system,recurrent_certificate,two_stage_certificate)
    if previous['status']!='verified':
        raise ValueError('Valid joint-grid two-stage certificate required')
    initial=initial_bins(system.state_dim)
    recurrent=tuple(recurrent_certificate['coordinate_bins'])
    active=_active_axes(initial,recurrent)
    initial_active=tuple(initial[i] for i in active)
    recurrent_active=tuple(recurrent[i] for i in active)
    candidates=_successor_forms(system,active)
    coefficients,fixed,pairs=_forms_and_pairs(system,active)
    seed=set()
    initial_candidates=0
    initial_retained=0
    for source in product(*(range(n) for n in initial_active)):
        limits=_source_pair_limits(source,initial_active,coefficients,fixed,pairs)
        for target in _successors(source,initial_active,recurrent_active,candidates):
            initial_candidates+=1
            if _could_reach(target,recurrent_active,limits):
                initial_retained+=1
                seed.add(target)
    seen=set(seed)
    queue=deque(sorted(seed))
    recurrent_candidates=0
    recurrent_retained=0
    while queue:
        source=queue.popleft()
        limits=_source_pair_limits(source,recurrent_active,coefficients,fixed,pairs)
        for target in _successors(source,recurrent_active,recurrent_active,candidates):
            recurrent_candidates+=1
            if _could_reach(target,recurrent_active,limits):
                recurrent_retained+=1
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
    outer_low=two_stage_certificate['recurrent_index_low']
    outer_high=two_stage_certificate['recurrent_index_high']
    if any(not outer_low[axis]<=row[k]<=outer_high[axis]
           for row in seen for k,axis in enumerate(active)):
        raise ValueError('Graph escaped certified joint-grid image rectangle')
    retained=sorted(seen)
    return {
        'schema':'ncd.dual-grid-correlated-closure.v1',
        'system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent_certificate),
        'two_stage_certificate_sha256':_digest(two_stage_certificate),
        'epsilon':recurrent_certificate['epsilon'],
        'action_bins':recurrent_certificate['action_bins'],
        'active_axes':list(active),
        'initial_active_bins':list(initial_active),
        'recurrent_active_bins':list(recurrent_active),
        'pair_separator_count':len(pairs),
        'initial_candidate_edges':initial_candidates,
        'initial_retained_edges':initial_retained,
        'initial_successor_count':len(seed),
        'recurrent_candidate_edges':recurrent_candidates,
        'recurrent_retained_edges':recurrent_retained,
        'recurrent_active_indices':[list(row) for row in retained],
        'initial_state_count':previous['initial_state_count'],
        'recurrent_state_count':len(retained),
        'upper_bound':previous['initial_state_count']+len(retained),
        'scope':'all unit initial states, all continuous unit action words, every finite horizon',
        'boundary':'pairwise action support is conservative, not exact action feasibility',
        'status':'certified',
    }


def certify_dual_grid_closure(system: ContinuousReLUSystem,
                              recurrent_certificate: dict,
                              two_stage_certificate: dict):
    certificate=_proof(system,recurrent_certificate,two_stage_certificate)
    verify_dual_grid_closure(system,recurrent_certificate,
                             two_stage_certificate,certificate)
    return certificate


def verify_dual_grid_closure(system: ContinuousReLUSystem,
                             recurrent_certificate: dict,
                             two_stage_certificate: dict,certificate: dict):
    if certificate.get('schema')!='ncd.dual-grid-correlated-closure.v1':
        raise ValueError('Unsupported dual-grid closure certificate')
    expected=_proof(system,recurrent_certificate,two_stage_certificate)
    if certificate!=expected:
        raise ValueError('Dual-grid closure replay mismatch')
    return {key:expected[key] for key in (
        'status','action_bins','pair_separator_count',
        'initial_state_count','initial_candidate_edges',
        'initial_retained_edges','initial_successor_count',
        'recurrent_candidate_edges','recurrent_retained_edges',
        'recurrent_state_count','upper_bound')}


def _validate_state(system,recurrent_certificate,certificate,state):
    if not isinstance(state,tuple) or len(state)!=2:
        raise ValueError('Malformed dual-grid state')
    if state[0]=='recurrent':
        indices=state[1]
        if len(indices)!=system.state_dim:
            raise ValueError('Malformed recurrent indices')
        key=[indices[i] for i in certificate['active_axes']]
        rows=certificate['recurrent_active_indices']
        position=bisect_left(rows,key)
        if position==len(rows) or rows[position]!=key:
            raise ValueError('Recurrent state not in dual-grid closure')
    elif state[0]!='initial':
        raise ValueError('Unknown graph phase')
    two_stage_output(system,recurrent_certificate,state)


def dual_grid_initial(system: ContinuousReLUSystem,point):
    return two_stage_initial(system,point)


def dual_grid_output(system: ContinuousReLUSystem,
                     recurrent_certificate: dict,certificate: dict,state):
    _validate_state(system,recurrent_certificate,certificate,state)
    return two_stage_output(system,recurrent_certificate,state)


def dual_grid_step(system: ContinuousReLUSystem,
                   recurrent_certificate: dict,certificate: dict,state,action):
    _validate_state(system,recurrent_certificate,certificate,state)
    target=two_stage_step(system,recurrent_certificate,state,action)
    _validate_state(system,recurrent_certificate,certificate,target)
    return target
