"""Exact finite-horizon intervention packing for arbitrary frozen ReLU systems.

A shared action word gives a lower bound for any deterministic finite-state
realization: two concrete states whose output traces differ by more than
2 epsilon cannot be paired with the same abstract state.
"""
from __future__ import annotations
from fractions import Fraction as Q
import math
from itertools import combinations, product
from .continuous_compositional_realization import value
from .continuous_separation import ContinuousReLUSystem, _digest
from .affine_observability import _affine_network, Unresolved as AffineUnresolved


class Unresolved(ValueError):
    """The proposed finite witness set does not separate at tolerance."""


def _parse_points(raw, dimension):
    if not isinstance(raw, list) or len(raw) < 2 or len(raw) > 4096:
        raise ValueError('Invalid packing point count')
    points = []
    for sparse in raw:
        if not isinstance(sparse, list):
            raise ValueError('Invalid sparse packing point')
        point = [Q(0)] * dimension
        previous = -1
        for entry in sparse:
            if (not isinstance(entry, list) or len(entry) != 2
                    or not isinstance(entry[0], int) or isinstance(entry[0], bool)
                    or not previous < entry[0] < dimension
                    or not isinstance(entry[1], str)):
                raise ValueError('Malformed sparse packing coordinate')
            index = entry[0]
            value_exact = Q(entry[1])
            if not 0 < value_exact <= 1:
                raise ValueError('Stored sparse coordinate outside (0,1]')
            point[index] = value_exact
            previous = index
        points.append(tuple(point))
    if len(set(points)) != len(points):
        raise ValueError('Duplicate packing point')
    return points


def _parse_actions(raw, dimension):
    if not isinstance(raw, list) or len(raw) > 256:
        raise ValueError('Invalid action-word length')
    actions = []
    for item in raw:
        if (not isinstance(item, list) or len(item) != dimension
                or any(not isinstance(x, str) for x in item)):
            raise ValueError('Malformed action word')
        action = tuple(Q(x) for x in item)
        if any(not 0 <= x <= 1 for x in action):
            raise ValueError('Action outside unit cube')
        actions.append(action)
    return actions


def _evaluate(system, points, actions, epsilon):
    # A fixed-phase proof gives an exact sparse affine evaluator. Every other
    # network falls back to direct exact ReLU evaluation at the witness points.
    try:
        transition, transition_bias, _ = _affine_network(system.transition)
        observation, observation_bias, _ = _affine_network(system.observation)
    except AffineUnresolved:
        evaluation_mode = 'direct-exact-ReLU'
        step = lambda state, action: value(system.transition,state+list(action))
        output = lambda state: value(system.observation,state)
    else:
        evaluation_mode = 'globally-affine'
        transition_sparse = [tuple((j,a) for j,a in enumerate(row) if a)
                             for row in transition]
        observation_sparse = [tuple((j,a) for j,a in enumerate(row) if a)
                              for row in observation]
        def apply(rows, biases, argument):
            return [bias + sum((a*argument[j] for j,a in row),Q(0))
                    for row,bias in zip(rows,biases)]
        step = lambda state, action: apply(transition_sparse,transition_bias,
                                           state+list(action))
        output = lambda state: apply(observation_sparse,observation_bias,state)
    traces = []
    for point in points:
        state = list(point)
        trace = []
        for time in range(len(actions)+1):
            if any(not 0 <= x <= 1 for x in state):
                raise ValueError('Witness trajectory leaves unit state cube')
            trace.append(tuple(output(state)))
            if time < len(actions):
                state = step(state,actions[time])
        traces.append(trace)
    threshold = 2*epsilon
    counts = [0]*(len(actions)+1)
    minimum = None
    for left,right in combinations(range(len(points)),2):
        distances = [max(abs(a-b) for a,b in zip(traces[left][time],traces[right][time]))
                     for time in range(len(actions)+1)]
        separation = max(distances)
        minimum = separation if minimum is None or separation < minimum else minimum
        first = next((time for time, distance in enumerate(distances)
                      if distance > threshold),None)
        if first is None:
            raise Unresolved(f'Packing pair {left},{right} not separated: {separation} <= {threshold}')
        counts[first] += 1
    return {'lower_bound':len(points),
            'pair_count':len(points)*(len(points)-1)//2,
            'first_witness_counts':counts,
            'minimum_pair_distance':str(minimum),
            'threshold_exact_float':str(threshold),
            'horizon':len(actions),
            'evaluation_mode':evaluation_mode}


def verify_dynamic_packing(system: ContinuousReLUSystem, certificate: dict):
    if (certificate.get('schema') != 'ncd.dynamic-packing.v1'
            or certificate.get('system_sha256') != _digest(system.to_dict())
            or certificate.get('state_dim') != system.state_dim
            or certificate.get('action_dim') != system.action_dim
            or certificate.get('claim') !=
            'minimum deterministic realization states for all unit initial states and action words'):
        raise ValueError('Dynamic packing system or tolerance mismatch')
    raw_epsilon = certificate.get('epsilon')
    if not isinstance(raw_epsilon,str):
        raise ValueError('Dynamic packing tolerance missing')
    try:
        epsilon_float = float(raw_epsilon)
    except ValueError as error:
        raise ValueError('Invalid dynamic packing tolerance') from error
    if (not math.isfinite(epsilon_float) or epsilon_float <= 0
            or raw_epsilon != repr(epsilon_float)):
        raise ValueError('Invalid dynamic packing tolerance')
    points = _parse_points(certificate.get('points'),system.state_dim)
    actions = _parse_actions(certificate.get('action_word'),system.action_dim)
    expected = _evaluate(system,points,actions,Q(epsilon_float))
    if certificate.get('metrics') != expected:
        raise ValueError('Dynamic packing metrics replay mismatch')
    return {'status':'verified', **expected}


def certify_dynamic_packing(system: ContinuousReLUSystem,
                            points, action_word, epsilon: float = 0.17):
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid dynamic packing tolerance')
    serialized = []
    for point in points:
        if len(point) != system.state_dim:
            raise ValueError('Packing point dimension mismatch')
        serialized.append([[j,str(Q(x))] for j,x in enumerate(point) if Q(x) != 0])
    actions = [[str(Q(x)) for x in action] for action in action_word]
    parsed_points = _parse_points(serialized,system.state_dim)
    parsed_actions = _parse_actions(actions,system.action_dim)
    certificate = {'schema':'ncd.dynamic-packing.v1',
                   'system_sha256':_digest(system.to_dict()),
                   'state_dim':system.state_dim,
                   'action_dim':system.action_dim,
                   'epsilon':repr(epsilon),
                   'points':serialized,
                   'action_word':actions,
                   'metrics':_evaluate(system,parsed_points,parsed_actions,Q(epsilon)),
                   'claim':'minimum deterministic realization states for all unit initial states and action words'}
    verify_dynamic_packing(system,certificate)
    return certificate


def staggered_ring_points(dimension):
    """Five (x_0,x_last) choices times 27 observed-coordinate triples."""
    if dimension < 5:
        raise ValueError('Staggered ring witness requires distinct last coordinate')
    pairs = ((Q(0),Q(0)),(Q(1,2),Q(0)),(Q(1),Q(0)),
             (Q(1,10),Q(1)),(Q(3,5),Q(1)))
    points = []
    for middle in product((Q(0),Q(1,2),Q(1)),repeat=3):
        for first,last in pairs:
            point = [Q(0)]*dimension
            point[0] = first
            point[1:4] = middle
            point[-1] = last
            points.append(tuple(point))
    return points


def certify_staggered_ring_packing(system: ContinuousReLUSystem):
    points = staggered_ring_points(system.state_dim)
    action = (tuple(Q(0) for _ in range(system.action_dim)),)
    certificate = certify_dynamic_packing(system,points,action)
    if (certificate['metrics']['lower_bound'] != 135
            or certificate['metrics']['first_witness_counts'] != [8991,54]):
        raise Unresolved('Staggered-ring witness pattern changed')
    return certificate
