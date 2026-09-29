"""Exact one-step output-trace packing capacity for affine feedback rings.

The capacity bounds the strength of every one-step common-action packing
argument. A matching point set also lower-bounds arbitrary finite machines.
"""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product

from .affine_observability import _affine_network
from .continuous_separation import ContinuousReLUSystem, _digest
from .dynamic_packing import certify_dynamic_packing, verify_dynamic_packing


def six_pair_ring_points(dimension: int):
    """Six interleaved feedback pairs times the observed-coordinate triples."""
    if dimension < 5:
        raise ValueError('Distinct feedback coordinate required')
    pairs = ((Q(0),Q(0)),(Q(1,10),Q(1)),
             (Q(9,20),Q(0)),(Q(11,20),Q(1)),
             (Q(9,10),Q(0)),(Q(1),Q(1)))
    points = []
    for middle in product((Q(0),Q(1,2),Q(1)),repeat=3):
        for first,last in pairs:
            point = [Q(0)]*dimension
            point[0] = first
            point[1:4] = middle
            point[-1] = last
            points.append(tuple(point))
    return tuple(points)


def certify_six_pair_packing(system: ContinuousReLUSystem):
    action = (tuple(Q(0) for _ in range(system.action_dim)),)
    certificate = certify_dynamic_packing(
        system,six_pair_ring_points(system.state_dim),action)
    result = verify_dynamic_packing(system,certificate)
    if (result['lower_bound'] != 162 or result['pair_count'] != 13041
            or result['first_witness_counts'] != [12960,81]):
        raise ValueError('Six-pair packing pattern changed')
    return certificate


def _proof(system: ContinuousReLUSystem, packing_certificate: dict):
    d = system.state_dim
    if d < 5 or system.observation.output_dim != 4:
        raise ValueError('Four-output feedback ring required')
    raw_epsilon = packing_certificate['epsilon']
    if raw_epsilon != repr(float(raw_epsilon)):
        raise ValueError('Packing tolerance must be canonical binary64')
    epsilon = Q(float(raw_epsilon))
    threshold = 2*epsilon
    transition,offset,_ = _affine_network(system.transition)
    observation,_,_ = _affine_network(system.observation)
    if any(tuple(row) != tuple(Q(int(j==i)) for j in range(d))
           for i,row in enumerate(observation)):
        raise ValueError('First four coordinates must be observed exactly')
    if len(transition) != d or any(len(row) != d+system.action_dim
                                    for row in transition):
        raise ValueError('Malformed affine transition')
    for row,bias in zip(transition,offset):
        low = bias+sum((min(Q(0),value) for value in row),Q(0))
        high = bias+sum((max(Q(0),value) for value in row),Q(0))
        if low < 0 or high > 1:
            raise ValueError('Transition does not preserve the unit cube')
    state = tuple(tuple(row[:d]) for row in transition)
    if (any(state[0][j] for j in range(d) if j not in (0,d-1))
            or any(state[i][j] for i in (1,2,3)
                   for j in range(d) if j not in (i-1,i))):
        raise ValueError('Observed affine dependency support changed')
    width = Q(1,3)
    if width > threshold:
        raise ValueError('Observation cells are too wide')
    other_spans = tuple((abs(state[i][i-1])+abs(state[i][i]))*width
                        for i in (1,2,3))
    if any(span > threshold for span in other_spans):
        raise ValueError('Other next-output coordinates can separate within a cell')
    first_span = abs(state[0][0])*width+abs(state[0][d-1])
    if first_span > 2*threshold:
        raise ValueError('Two-point capacity within a cell is not proved')
    # For two runs under the same action word, their affine action terms
    # cancel. Let M_t bound the first four coordinate differences and H_t
    # bound the hidden coordinates. If M_0,M_1<=threshold, then H_1<=gamma.
    # The checked rectangle is forward invariant for every later time.
    hidden_gain = max(sum((abs(x) for x in state[i]),Q(0))
                      for i in range(4,d))
    other_gain = max(sum((abs(x) for x in state[i]),Q(0))
                     for i in (1,2,3))
    future_first = abs(state[0][0])*threshold+abs(state[0][d-1])*hidden_gain
    future_other = other_gain*threshold
    future_hidden = hidden_gain*max(threshold,hidden_gain)
    if (threshold > 1 or hidden_gain > 1
            or future_first > threshold or future_other > threshold
            or future_hidden > hidden_gain):
        raise ValueError('All-horizon threshold propagation failed')
    packing = verify_dynamic_packing(system,packing_certificate)
    if (packing['status'] != 'verified' or packing['lower_bound'] != 162
            or packing['pair_count'] != 13041 or packing['horizon'] != 1
            or packing['evaluation_mode'] != 'globally-affine'
            or Q(packing['minimum_pair_distance']) <= threshold):
        raise ValueError('Matching one-step lower witness failed')
    return {
        'schema':'ncd.one-step-packing-capacity.v1',
        'system_sha256':_digest(system.to_dict()),
        'packing_certificate_sha256':_digest(packing_certificate),
        'epsilon':raw_epsilon,
        'epsilon_exact_float':str(epsilon),
        'threshold':str(threshold),
        'state_dim':d,
        'action_dim':system.action_dim,
        'partition_axes':[0,1,2,3],
        'intervals_per_axis':3,
        'cell_width':str(width),
        'cell_count':81,
        'maximum_points_per_cell':2,
        'row0_next_output_span_bound':str(first_span),
        'other_next_output_span_bounds':[str(x) for x in other_spans],
        'lower_witness_points':packing['lower_bound'],
        'lower_witness_pairs':packing['pair_count'],
        'lower_first_witness_counts':packing['first_witness_counts'],
        'exact_one_step_packing_capacity':162,
        'hidden_gain':str(hidden_gain),
        'other_observed_gain':str(other_gain),
        'future_first_output_bound':str(future_first),
        'future_other_output_bound':str(future_other),
        'future_hidden_bound':str(future_hidden),
        'all_horizon_collapse':'unseparated at times 0 and 1 implies unseparated at every later time',
        'exact_all_horizon_packing_capacity':162,
        'scope':'any full-cube initial states and any common continuous action word of arbitrary finite length',
        'status':'certified',
    }


def certify_one_step_capacity(system: ContinuousReLUSystem,
                              packing_certificate: dict):
    certificate = _proof(system,packing_certificate)
    verify_one_step_capacity(system,packing_certificate,certificate)
    return certificate


def verify_one_step_capacity(system: ContinuousReLUSystem,
                             packing_certificate: dict,
                             certificate: dict):
    if certificate.get('schema') != 'ncd.one-step-packing-capacity.v1':
        raise ValueError('Unsupported one-step capacity certificate')
    expected = _proof(system,packing_certificate)
    if certificate != expected:
        raise ValueError('One-step capacity replay mismatch')
    return {'status':'verified',
            'exact_one_step_packing_capacity':expected['exact_one_step_packing_capacity'],
            'exact_all_horizon_packing_capacity':expected['exact_all_horizon_packing_capacity'],
            'lower_witness_points':expected['lower_witness_points'],
            'lower_witness_pairs':expected['lower_witness_pairs']}
