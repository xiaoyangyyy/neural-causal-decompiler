"""Exact compositional finite-state bounds for separable ReLU products.

The verifier checks every serialized network coefficient. It composes a
replayed scalar eight-state exclusion and exact nine-state upper without
enumerating the exponential product state space.
"""
from __future__ import annotations
from fractions import Fraction as Q
from .continuous_separation import ContinuousReLUSystem, _digest
from .affine_observability import _affine_network, Unresolved as AffineUnresolved
from .exact_interval_lower import (
    verify_interval_chain_exclusion, verify_exact_nine_state_upper,
)


class Unresolved(ValueError):
    """Network or scalar proof does not satisfy the product theorem."""


def _check_semantic_product(system: ContinuousReLUSystem):
    """Prove the realized function is the product, regardless of hidden order."""
    d = system.state_dim
    if system.action_dim != d:
        raise Unresolved('State and action dimensions differ')
    try:
        transition, transition_bias, transition_phase = _affine_network(system.transition)
        observation, observation_bias, observation_phase = _affine_network(system.observation)
    except AffineUnresolved as error:
        raise Unresolved(str(error)) from error
    lam, beta = Q(1,2), Q(0.4)
    if lam+beta > 1:
        raise Unresolved('Exact unit-domain invariance failed')
    if len(transition) != d or len(observation) != d:
        raise Unresolved('Network output dimensions do not match the product')
    for i in range(d):
        if transition_bias[i] != 0:
            raise Unresolved('Transition has a nonzero offset')
        for j, value in enumerate(transition[i]):
            expected = lam if j == i else beta if j == d+i else Q(0)
            if value != expected:
                raise Unresolved('Transition coordinates are coupled or changed')
        if observation_bias[i] != 0:
            raise Unresolved('Observation has a nonzero offset')
        for j, value in enumerate(observation[i]):
            if value != (1 if i == j else 0):
                raise Unresolved('Observation is not coordinate identity')
    coefficients = sum(
        sum(len(row) for layer in network.weights for row in layer)
        + sum(len(layer) for layer in network.biases)
        for network in (system.transition, system.observation)
    )
    return lam, beta, coefficients, transition_phase, observation_phase


def _compute(system: ContinuousReLUSystem, scalar_exclusion: dict):
    lam, beta, coefficients, transition_phase, observation_phase = _check_semantic_product(system)
    if verify_interval_chain_exclusion(scalar_exclusion)['excluded_states'] != 8:
        raise Unresolved('Scalar eight-state exclusion did not replay')
    if verify_exact_nine_state_upper()['states'] != 9:
        raise Unresolved('Scalar nine-state upper did not replay')
    d = system.state_dim
    epsilon = Q(0.101)
    fixed_high = beta / (1-lam)
    if not (0 < fixed_high <= 1 and fixed_high / 3 > 2*epsilon):
        raise Unresolved('Four controlled fixed points are not separated')
    fixed = [fixed_high*i/3 for i in range(4)]
    actions = [((1-lam)*value)/beta for value in fixed]
    if any(not 0 <= value <= 1 for value in actions):
        raise Unresolved('Fixed actions leave unit domain')
    return {
        'schema': 'ncd.exact-separable-product-bounds.v1',
        'system_sha256': _digest(system.to_dict()),
        'scalar_exclusion_sha256': _digest(scalar_exclusion),
        'dimension': d,
        'epsilon': repr(0.101),
        'lambda': str(lam), 'beta': str(beta),
        'coefficients_checked': coefficients,
        'transition_phase': transition_phase,
        'observation_phase': observation_phase,
        'fixed_second_values': [str(x) for x in fixed],
        'fixed_second_actions': [str(x) for x in actions],
        'independent_slices': str(4**(d-1)),
        'lower_bound': str(9*4**(d-1)),
        'upper_bound': str(9**d),
        'horizon': 'unbounded',
        'action_domain': '[0,1]^dimension',
        'lower_proof': 'disjoint fixed-coordinate slices, each requiring nine scalar states',
        'upper_proof': 'implicit coordinate product of exact nine-state realization',
    }


def certify_product_bounds(system: ContinuousReLUSystem, scalar_exclusion: dict):
    """Produce a compact exact certificate for any verified product dimension."""
    return _compute(system, scalar_exclusion)


def verify_product_bounds(system: ContinuousReLUSystem,
                          scalar_exclusion: dict, certificate: dict):
    expected = _compute(system, scalar_exclusion)
    if certificate != expected:
        raise ValueError('Separable-product certificate replay mismatch')
    return {'status':'verified', 'dimension':expected['dimension'],
            'coefficients_checked':expected['coefficients_checked'],
            'lower_bound':expected['lower_bound'],
            'upper_bound':expected['upper_bound'],
            'horizon':'unbounded'}
