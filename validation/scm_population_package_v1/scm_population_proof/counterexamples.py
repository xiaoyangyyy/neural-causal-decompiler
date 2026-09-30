"""Exact Student-5 square-chain population transport counterexample.

This is an ideal real-law family witness, not a claim about a particular
seeded confirmation world or a finite device PRNG.
"""
from fractions import Fraction as Q
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def nonnegative_fraction(value):
    if type(value) not in (str, int):
        raise ValueError('Expected a rational string or integer')
    result = Q(value)
    if result < 0 or max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError('Invalid nonnegative rational')
    return result


def derive(n_nodes=5, estimated_first_moment_upper='0', requested_cost_lower='1'):
    if type(n_nodes) is not int or n_nodes not in (5, 8):
        raise ValueError('The chain needs a separate intervention isolate')
    B = nonnegative_fraction(estimated_first_moment_upper)
    K = nonnegative_fraction(requested_cost_lower)
    alpha = Q(4, 9) * Q(5, 69) ** 3
    growth = Q(1, 16) ** 7
    beta = (2 * alpha) ** 3 * alpha * growth / 3
    ratio = (B + K + 1) / beta
    M = 16 + (ratio.numerator + ratio.denominator - 1) // ratio.denominator
    finite_lower = beta * (M ** 3 - 16 ** 3) - B
    if finite_lower <= K:
        raise ArithmeticError('The strict finite cutoff bound failed')
    model = {
        'nodes': n_nodes,
        'edges': [[0, 1], [1, 2], [2, 3]],
        'extra_nodes': 'independent isolated noise variables',
        'equations': ['X0=U0', 'X1=a1*X0^2+U1', 'X2=a2*X1^2+U2', 'X3=a3*X2^2+U3'],
        'coefficient_interval': ['1/8', '1/4'],
        'witness_coefficient': '3/16',
        'noise': {'family': 'independent Student t, df=5', 'form': 'U=sigma*T5/kappa',
                  'sigma_interval': ['1/4', '1/2'], 'kappa_interval': ['1', '2'],
                  'benchmark_sigma': 'stored IEEE-754 value 0.35',
                  'benchmark_kappa': 'sqrt(5/3)'},
        'observed_scales': 'all one',
        'intervention': {'target': n_nodes - 1, 'observed_value': '1'},
        'semantic_scope': 'ideal real-valued mathematical population; finite device PRNG excluded',
    }
    return {
        'schema': 'ncd.student5-transport-counterexample.v1', 'status': 'refuted-scoped',
        'claim': 'All allowed ideal population SCMs admit finite extended L1 coupling cost to a finite-first-moment empirical estimate',
        'model': model, 'model_sha256': hashlib.sha256(canonical(model).encode()).hexdigest(),
        'density': '8*kappa/(3*sigma*sqrt(5)*pi)*(1+kappa^2*u^2/(5*sigma^2))^(-3)',
        'density_minorant_alpha': str(alpha),
        'density_minorant': 'f(u)>=alpha*u^(-6) for u>=1; f(u)>=alpha for |u|<=1',
        'growth_lower': str(growth),
        'growth_statement': 'On U0>=16 and |U1|,|U2|,|U3|<=1, X3>=growth_lower*U0^8',
        'other_noise_event_probability_lower': str((2 * alpha) ** 3),
        'truncated_moment_beta': str(beta),
        'truncated_moment_inequality': 'E[min(max(X3,0),growth_lower*M^8)]>=beta*(M^3-16^3) for M>=16',
        'finite_witness': {'estimated_first_moment_upper_B': str(B),
                           'requested_cost_lower_K': str(K), 'cutoff_M': str(M),
                           'truncated_output_cap': str(growth * M ** 8),
                           'strict_coupling_cost_lower': str(finite_lower)},
        'conclusion': 'For every finite B and K the coupling cost exceeds K; hence the true coordinate has infinite first moment and the extended L1 cost is infinite',
        'not_proved': ['any historical confirmation world has infinite first moment',
                       'bounded or weak distribution metrics fail',
                       'finite device PRNG exactly realizes the ideal law',
                       'the original compound causal-decompilation objective is resolved'],
        'original_claim_closed': False, 'original_objective_achieved': False,
    }


def verify(certificate):
    if type(certificate) is not dict:
        raise ValueError('Invalid counterexample certificate')
    model = certificate.get('model', {})
    witness = certificate.get('finite_witness', {})
    expected = derive(model.get('nodes'), witness.get('estimated_first_moment_upper_B'),
                      witness.get('requested_cost_lower_K'))
    if certificate != expected:
        raise ValueError('Counterexample certificate mismatch')
    return {'status': 'verified', 'conclusion': 'refuted-scoped',
            'original_objective_achieved': False}
