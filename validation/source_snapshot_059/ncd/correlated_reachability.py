"""Exact pairwise-action correlation filter for a finite neural realization.

A target cell is kept unless every action-box necessary condition survives.
Strictly disjoint pairwise ranges prove an edge impossible without LP rounding.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from fractions import Fraction as Q
from itertools import product

from .abstract_reachability import (
    _active_axes, _successor_forms, _successors, graph_initial,
    verify_abstract_closure)
from .affine_observability import _affine_network
from .continuous_separation import ContinuousReLUSystem, _digest
from .reachable_two_stage import (
    initial_bins, two_stage_output, two_stage_step)


def _forms_and_pairs(system: ContinuousReLUSystem, active):
    matrix,offset,_=_affine_network(system.transition)
    d=system.state_dim
    coefficients=[]
    fixed=[]
    actions=[]
    for axis in active:
        row=matrix[axis]
        coefficients.append(tuple(row[j] for j in active))
        fixed.append(offset[axis]+sum(
            (row[j]*Q(1,2) for j in range(d) if j not in active),Q(0)))
        actions.append(row[d:])
    pairs=[]
    for i in range(len(active)):
        for k in range(i+1,len(active)):
            first=sum(actions[i],Q(0))
            second=sum(actions[k],Q(0))
            if first<=0 or second<=0:
                raise ValueError('Positive total action influence required')
            ratios={first/second}
            ratios.update(x/y for x,y in zip(actions[i],actions[k])
                          if y and x/y>0)
            for ratio in sorted(ratios):
                difference=tuple(x-ratio*y for x,y in zip(actions[i],actions[k]))
                pairs.append((i,k,ratio,
                              sum((min(Q(0),x) for x in difference),Q(0)),
                              sum((max(Q(0),x) for x in difference),Q(0))))
    return tuple(coefficients),tuple(fixed),tuple(pairs)


def _source_pair_limits(indices, source_bins, coefficients, fixed, pairs):
    centers=tuple(Q(2*q+1,2*n) for q,n in zip(indices,source_bins))
    base=tuple(f+sum((v*c for v,c in zip(row,centers)),Q(0))
               for f,row in zip(fixed,coefficients))
    return tuple((i,k,ratio,base[i]-ratio*base[k]+low,
                  base[i]-ratio*base[k]+high)
                 for i,k,ratio,low,high in pairs)


def _could_reach(indices, target_bins, limits):
    # Closed target cells deliberately include boundary ties: rejecting only
    # strict disjointness is sound for floor-and-clamp quantization.
    for i,k,ratio,low,high in limits:
        target_low=Q(indices[i],target_bins[i])-ratio*Q(indices[k]+1,target_bins[k])
        target_high=Q(indices[i]+1,target_bins[i])-ratio*Q(indices[k],target_bins[k])
        if target_low>high or target_high<low:
            return False
    return True


def _proof(system: ContinuousReLUSystem,recurrent_certificate: dict,
           two_stage_certificate: dict,abstract_certificate: dict):
    previous=verify_abstract_closure(
        system,recurrent_certificate,two_stage_certificate,abstract_certificate)
    if previous['status']!='verified':
        raise ValueError('Prior abstract graph must verify')
    initial=initial_bins(system.state_dim)
    recurrent=tuple(recurrent_certificate['coordinate_bins'])
    active=_active_axes(initial,recurrent)
    initial_active=tuple(initial[i] for i in active)
    recurrent_active=tuple(recurrent[i] for i in active)
    candidates=_successor_forms(system,active)
    coefficients,fixed,pairs=_forms_and_pairs(system,active)
    old={tuple(row) for row in abstract_certificate['recurrent_active_indices']}
    seed=set()
    initial_candidates=0
    for source in product(*(range(n) for n in initial_active)):
        limits=_source_pair_limits(source,initial_active,coefficients,fixed,pairs)
        for target in _successors(source,initial_active,recurrent_active,candidates):
            initial_candidates+=1
            if _could_reach(target,recurrent_active,limits):
                seed.add(target)
    seen=set(seed)
    queue=deque(sorted(seed))
    recurrent_candidates=0
    recurrent_edges=0
    while queue:
        source=queue.popleft()
        limits=_source_pair_limits(source,recurrent_active,coefficients,fixed,pairs)
        for target in _successors(source,recurrent_active,recurrent_active,candidates):
            recurrent_candidates+=1
            if _could_reach(target,recurrent_active,limits):
                recurrent_edges+=1
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
    retained=sorted(seen)
    if not seen<=old:
        raise ValueError('Correlated closure escaped the prior certified graph')
    return {
        'schema':'ncd.correlated-action-closure.v1',
        'system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent_certificate),
        'two_stage_certificate_sha256':_digest(two_stage_certificate),
        'abstract_certificate_sha256':_digest(abstract_certificate),
        'epsilon':recurrent_certificate['epsilon'],
        'active_axes':list(active),
        'initial_active_bins':list(initial_active),
        'recurrent_active_bins':list(recurrent_active),
        'pair_separator':'positive total-action and coordinate action-slope ratios; exact pairwise support of shared action cube',
        'pair_separator_count':len(pairs),
        'initial_candidate_edges':initial_candidates,
        'initial_successor_count':len(seed),
        'recurrent_candidate_edges':recurrent_candidates,
        'recurrent_retained_edges':recurrent_edges,
        'recurrent_active_indices':[list(row) for row in retained],
        'initial_state_count':previous['initial_state_count'],
        'recurrent_state_count':len(retained),
        'upper_bound':previous['initial_state_count']+len(retained),
        'scope':'all unit initial states, all continuous unit action words, every finite horizon',
        'status':'certified',
    }


def certify_correlated_closure(system: ContinuousReLUSystem,
                               recurrent_certificate: dict,
                               two_stage_certificate: dict,
                               abstract_certificate: dict):
    certificate=_proof(system,recurrent_certificate,two_stage_certificate,
                       abstract_certificate)
    verify_correlated_closure(system,recurrent_certificate,
                              two_stage_certificate,abstract_certificate,certificate)
    return certificate


def verify_correlated_closure(system: ContinuousReLUSystem,
                              recurrent_certificate: dict,
                              two_stage_certificate: dict,
                              abstract_certificate: dict,certificate: dict):
    if certificate.get('schema')!='ncd.correlated-action-closure.v1':
        raise ValueError('Unsupported correlated closure certificate')
    expected=_proof(system,recurrent_certificate,two_stage_certificate,
                    abstract_certificate)
    if certificate!=expected:
        raise ValueError('Correlated closure replay mismatch')
    return {key:expected[key] for key in (
        'status','initial_state_count','initial_successor_count',
        'recurrent_state_count','initial_candidate_edges',
        'recurrent_candidate_edges','recurrent_retained_edges','upper_bound')}


def _validate_state(system,recurrent_certificate,certificate,state):
    if not isinstance(state,tuple) or len(state)!=2:
        raise ValueError('Malformed correlated graph state')
    if state[0]=='recurrent':
        indices=state[1]
        if len(indices)!=system.state_dim:
            raise ValueError('Malformed recurrent indices')
        key=[indices[i] for i in certificate['active_axes']]
        rows=certificate['recurrent_active_indices']
        position=bisect_left(rows,key)
        if position==len(rows) or rows[position]!=key:
            raise ValueError('Recurrent state not in correlated graph closure')
    elif state[0]!='initial':
        raise ValueError('Unknown graph phase')
    two_stage_output(system,recurrent_certificate,state)


def correlated_initial(system: ContinuousReLUSystem,point):
    return graph_initial(system,point)


def correlated_output(system: ContinuousReLUSystem,
                      recurrent_certificate: dict,certificate: dict,state):
    _validate_state(system,recurrent_certificate,certificate,state)
    return two_stage_output(system,recurrent_certificate,state)


def correlated_step(system: ContinuousReLUSystem,
                    recurrent_certificate: dict,certificate: dict,state,action):
    _validate_state(system,recurrent_certificate,certificate,state)
    target=two_stage_step(system,recurrent_certificate,state,action)
    _validate_state(system,recurrent_certificate,certificate,target)
    return target
