"""Numerical integer-grid refinement with exact independent upper replay.

The search score is untrusted. Only verify_weighted can certify a bound.
"""
from __future__ import annotations
from itertools import product
from fractions import Fraction as Q
import math
import numpy as np
from .automatic_grid_realization import absolute_influence
from .continuous_compositional_realization import certify_weighted, verify_weighted
from .continuous_separation import ContinuousReLUSystem, _digest


def propose_refinement(system: ContinuousReLUSystem, baseline: dict, *,
                       action_multipliers: tuple[int, ...] = (1,4)) -> dict:
    if (not action_multipliers or len(action_multipliers)>5
            or any(not isinstance(x,int) or isinstance(x,bool) or x<1
                   for x in action_multipliers)
            or tuple(sorted(set(action_multipliers))) != action_multipliers):
        raise ValueError('Invalid action-bin multiplier search')
    prior = verify_weighted(system,baseline)
    if prior['status'] != 'certified' or prior['upper_bound'] is None:
        raise ValueError('Refinement requires a certified baseline upper')
    old_bins = tuple(int(x) for x in baseline['coordinate_bins'])
    d,u = system.state_dim,system.action_dim
    active = tuple(i for i,n in enumerate(old_bins) if n>1)
    if len(active) < 2 or len(active) > 8:
        raise ValueError('Refinement supports 2 to 8 active coordinates')
    influence = absolute_influence(system.transition)
    state = influence[:,:d]
    control = influence[:,d:]
    output = absolute_influence(system.observation)
    if max(abs(np.linalg.eigvals(state))) >= 1-1e-9:
        raise ValueError('Absolute influence is not contractive')
    transfer = np.linalg.solve(np.eye(d)-state,np.eye(d))
    if not np.isfinite(transfer).all() or np.min(transfer)<-1e-8:
        raise ValueError('Influence transfer unresolved')
    transfer = np.maximum(transfer,0.0)
    impact = output@transfer
    pivot = min(active,key=lambda j:(-float(np.max(impact[:,j])),j))
    others = tuple(j for j in active if j!=pivot)
    ranges = {j:range(max(1,old_bins[j]//2),
                      max(old_bins[j]+2,math.ceil(1.5*old_bins[j]))+1)
              for j in others}
    old_cost = math.prod(old_bins)
    best = None
    searched = 0
    for action_bins in (int(baseline['action_bins'])*m
                        for m in action_multipliers):
        control_error = control.sum(axis=1)/(2*action_bins)
        base = impact@control_error
        background = sum((impact[:,j]/(2*old_bins[j])
                          for j in range(d) if j not in active),
                         np.zeros(output.shape[0]))
        for values in product(*(ranges[j] for j in others)):
            searched += 1
            rest = base+background
            for j,n in zip(others,values):
                rest = rest+impact[:,j]/(2*n)
            residual = float(Q(baseline['epsilon']))-rest
            if np.min(residual)<=0:
                continue
            required = np.max(impact[:,pivot]/(2*residual))
            if not math.isfinite(float(required)):
                continue
            n_pivot = max(1,math.ceil(float(required)+1e-10))
            bins = list(old_bins)
            bins[pivot] = n_pivot
            for j,n in zip(others,values):
                bins[j] = n
            cost = math.prod(bins)
            if cost >= old_cost or (best is not None and cost >= best[0]):
                continue
            half = np.asarray([1/(2*n) for n in bins])
            radii = np.linalg.solve(np.eye(d)-state,half+control_error)
            radii = np.ceil(radii*1.0005*1e10)/1e10
            if (not np.isfinite(radii).all() or np.min(radii)<=0
                    or np.max(output@radii)>float(Q(baseline['epsilon']))):
                continue
            best = (cost,action_bins,tuple(bins),tuple(f'{x:.10f}' for x in radii))
    if best is None:
        return {'schema':'ncd.integer-grid-refinement-proposal.v1',
                'system_sha256':_digest(system.to_dict()),
                'baseline_sha256':_digest(baseline),
                'action_multipliers':list(action_multipliers),
                'status':'unresolved','reason':'no smaller numerical grid in declared search'}
    cost,action_bins,bins,radii = best
    return {'schema':'ncd.integer-grid-refinement-proposal.v1',
            'system_sha256':_digest(system.to_dict()),
            'baseline_sha256':_digest(baseline),
            'action_multipliers':list(action_multipliers),
            'status':'candidate',
            'old_upper_bound':str(old_cost),
            'candidate_upper_bound':str(cost),
            'active_coordinates':list(active),
            'pivot_coordinate':pivot,
            'search_ranges':{str(j):[ranges[j].start,ranges[j].stop-1] for j in others},
            'numeric_tuples_searched':searched,
            'coordinate_bins':list(bins),
            'coordinate_radii':list(radii),
            'action_bins':action_bins,
            'epsilon':baseline['epsilon'],
            'packing_axes':baseline['packing_axes'],
            'boundary':'numerical proposal only; exact weighted verifier decides certification'}


def certify_refinement(system: ContinuousReLUSystem, baseline: dict, *,
                       action_multipliers: tuple[int, ...] = (1,4)):
    proposal = propose_refinement(
        system,baseline,action_multipliers=action_multipliers)
    if proposal['status'] != 'candidate':
        return proposal,None
    certificate = certify_weighted(
        system,tuple(proposal['coordinate_bins']),tuple(proposal['coordinate_radii']),
        action_bins=proposal['action_bins'],epsilon=proposal['epsilon'],
        packing_axes=proposal['packing_axes'])
    if (certificate['status'] != 'certified'
            or certificate['upper_bound'] != proposal['candidate_upper_bound']):
        raise ValueError('Numerical grid failed exact certification')
    return proposal,certificate


def verify_refinement(system: ContinuousReLUSystem, baseline: dict,
                      proposal: dict, certificate: dict):
    if (Q(certificate['epsilon']) != Q(proposal['epsilon'])
            or certificate['packing_axes'] != proposal['packing_axes']):
        raise ValueError('Refined certificate parameter mismatch')
    multipliers = tuple(proposal.get('action_multipliers',()))
    expected = propose_refinement(
        system,baseline,action_multipliers=multipliers)
    if proposal != expected or proposal['status'] != 'candidate':
        raise ValueError('Integer-grid proposal replay mismatch')
    result = verify_weighted(system,certificate)
    if (result['status'] != 'certified'
            or result['upper_bound'] != proposal['candidate_upper_bound']
            or certificate['coordinate_bins'] != proposal['coordinate_bins']
            or tuple(Q(x) for x in certificate['coordinate_radii']) !=
               tuple(Q(x) for x in proposal['coordinate_radii'])
            or certificate['action_bins'] != proposal['action_bins']):
        raise ValueError('Refined exact upper proof mismatch')
    return {'status':'verified','old_upper_bound':proposal['old_upper_bound'],
            'new_upper_bound':result['upper_bound'],
            'action_bins':proposal['action_bins'],
            'numeric_tuples_searched':proposal['numeric_tuples_searched']}
