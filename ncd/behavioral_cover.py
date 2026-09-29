"""Exact all-horizon behavioral covering number for affine ring networks.

The centers are concrete initial states whose true output trajectories cover
every initial state's trajectory for every common action word. This is not
an executable finite-state transition system.
"""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product

from .affine_observability import _affine_network
from .continuous_separation import ContinuousReLUSystem, _digest
from .one_step_packing_capacity import verify_one_step_capacity


OBSERVED_CENTERS = (Q(1,6),Q(1,2),Q(5,6))
FEEDBACK_CENTERS = (Q(1,4),Q(3,4))


def _covers_unit_interval(centers: tuple[Q, ...], radius: Q) -> bool:
    """Check a finite closed-interval cover without sampling."""
    intervals = sorted((center-radius,center+radius) for center in centers)
    if not intervals or intervals[0][0] > 0:
        return False
    reached = intervals[0][1]
    for left,right in intervals[1:]:
        if left > reached:
            return False
        reached = max(reached,right)
    return reached >= 1


def representative_states(dimension: int):
    if dimension < 5:
        raise ValueError('Four observed and one feedback coordinate required')
    centers = []
    for observed in product(OBSERVED_CENTERS, repeat=4):
        for feedback in FEEDBACK_CENTERS:
            state = [Q(1,2)]*dimension
            state[:4] = observed
            state[-1] = feedback
            centers.append(tuple(state))
    return tuple(centers)


def _proof(system: ContinuousReLUSystem, packing_certificate: dict,
           capacity_certificate: dict):
    capacity = verify_one_step_capacity(
        system,packing_certificate,capacity_certificate)
    if (capacity['status'] != 'verified'
            or capacity['exact_all_horizon_packing_capacity'] != 162):
        raise ValueError('Matching all-horizon packing lower proof required')
    epsilon = Q(float(packing_certificate['epsilon']))
    if Q(capacity_certificate['epsilon_exact_float']) != epsilon:
        raise ValueError('Behavioral cover tolerance mismatch')
    d=system.state_dim
    transition,_,_=_affine_network(system.transition)
    state=tuple(tuple(row[:d]) for row in transition)
    a=abs(state[0][0])
    b=abs(state[0][d-1])
    observed_gain=max(sum((abs(value) for value in state[i]),Q(0))
                      for i in (1,2,3))
    hidden_gain=max(sum((abs(value) for value in state[i]),Q(0))
                    for i in range(4,d))
    centers=representative_states(d)
    if len(centers)!=162 or any(
            not 0<=value<=1 for center in centers for value in center):
        raise ValueError('Invalid concrete representatives')
    # Nearest-center quantization gives these coordinate error limits
    # everywhere on the unit state cube.
    initial_observed=Q(1,6)
    initial_feedback=Q(1,4)
    initial_other_hidden=Q(1,2)
    initial_hidden=max(initial_feedback,initial_other_hidden)
    if (not _covers_unit_interval(OBSERVED_CENTERS,initial_observed)
            or not _covers_unit_interval(FEEDBACK_CENTERS,initial_feedback)
            or not _covers_unit_interval((Q(1,2),),initial_other_hidden)):
        raise ValueError('Representative coordinates do not cover the unit cube')
    first_observed0=a*initial_observed+b*initial_feedback
    first_observed_other=observed_gain*initial_observed
    hidden_after_first=hidden_gain*initial_hidden
    future_observed0=a*epsilon+b*hidden_after_first
    future_observed_other=observed_gain*epsilon
    future_hidden=hidden_gain*max(epsilon,hidden_after_first)
    if (initial_observed>epsilon or first_observed0>epsilon
            or first_observed_other>epsilon or future_observed0>epsilon
            or future_observed_other>epsilon
            or future_hidden>hidden_after_first):
        raise ValueError('Concrete representative trajectories do not cover')
    return {
        'schema':'ncd.behavioral-cover.v1',
        'system_sha256':_digest(system.to_dict()),
        'packing_certificate_sha256':_digest(packing_certificate),
        'capacity_certificate_sha256':_digest(capacity_certificate),
        'epsilon':packing_certificate['epsilon'],
        'epsilon_exact_float':str(epsilon),
        'state_dim':d,
        'action_dim':system.action_dim,
        'observed_centers':[str(x) for x in OBSERVED_CENTERS],
        'feedback_centers':[str(x) for x in FEEDBACK_CENTERS],
        'other_hidden_center':'1/2',
        'center_set_sha256':_digest([[str(x) for x in point] for point in centers]),
        'concrete_representative_count':len(centers),
        'initial_observed_error_upper':str(initial_observed),
        'initial_hidden_error_upper':str(initial_hidden),
        'first_step_output0_error_upper':str(first_observed0),
        'first_step_other_output_error_upper':str(first_observed_other),
        'first_step_hidden_error_upper':str(hidden_after_first),
        'future_output0_error_upper':str(future_observed0),
        'future_other_output_error_upper':str(future_observed_other),
        'future_hidden_error_upper':str(future_hidden),
        'all_horizon_packing_lower':capacity['exact_all_horizon_packing_capacity'],
        'exact_all_horizon_behavioral_cover_number':162,
        'scope':'all unit initial states, every shared continuous unit action word, all finite horizons',
        'boundary':'trajectory cover is not a deterministic finite-state realization',
        'status':'certified',
    }


def certify_behavioral_cover(system: ContinuousReLUSystem,
                              packing_certificate: dict,
                              capacity_certificate: dict):
    certificate=_proof(system,packing_certificate,capacity_certificate)
    verify_behavioral_cover(
        system,packing_certificate,capacity_certificate,certificate)
    return certificate


def verify_behavioral_cover(system: ContinuousReLUSystem,
                             packing_certificate: dict,
                             capacity_certificate: dict,
                             certificate: dict):
    if certificate.get('schema')!='ncd.behavioral-cover.v1':
        raise ValueError('Unsupported behavioral-cover certificate')
    expected=_proof(system,packing_certificate,capacity_certificate)
    if certificate!=expected:
        raise ValueError('Behavioral-cover replay mismatch')
    return {'status':'verified',
            'cover_number':expected['exact_all_horizon_behavioral_cover_number'],
            'representative_count':expected['concrete_representative_count'],
            'packing_lower':expected['all_horizon_packing_lower']}
