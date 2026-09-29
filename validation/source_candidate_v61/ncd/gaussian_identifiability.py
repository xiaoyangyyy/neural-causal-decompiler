"""A checked observational non-identifiability witness, not a fitted failure."""
from fractions import Fraction as Q


def certify_gaussian_nonidentifiability(coefficient='1/2', do_value='1'):
    a, intervention = Q(coefficient), Q(do_value)
    if not 0 < abs(a) < 1 or not intervention:
        raise ValueError("Need nonzero correlation below one and nonzero intervention")
    residual = 1 - a*a
    return {
        'schema': 'ncd.gaussian-nonidentifiability.v1',
        'status': 'refuted',
        'claim': 'Every observational linear Gaussian SCM admits unique causal direction recovery',
        'quantifiers': 'any observational estimator; any finite sample count, or the population law',
        'assumptions': ['acyclic', 'no hidden confounding', 'independent centered Gaussian exogenous variables'],
        'forward': {'graph': [[0,1],[0,0]], 'coefficient':str(a), 'root_variance':'1', 'residual_variance':str(residual)},
        'reverse': {'graph': [[0,0],[1,0]], 'coefficient':str(a), 'root_variance':'1', 'residual_variance':str(residual)},
        'observational_mean': ['0','0'],
        'observational_covariance': [['1',str(a)],[str(a),'1']],
        'do_X': str(intervention),
        'forward_do_Y_mean': str(a*intervention),
        'reverse_do_Y_mean': '0',
        'finite_sample_minimax_error_lower': '1/2',
        'argument': 'Both linear images of independent Gaussians have the same mean and positive definite covariance, hence the same joint law. Their n-fold observational laws also coincide. The two correct-direction output probabilities sum to at most one, so one error probability is at least one half, including randomized estimators. do(X=c) yields different Y means.',
        'does_not_refute': ['identification under additional restricted noise assumptions', 'identification with informative interventions', 'recovery of an observational equivalence class']}


def verify_gaussian_nonidentifiability(certificate):
    if certificate.get('schema') != 'ncd.gaussian-nonidentifiability.v1':
        raise ValueError("Unsupported Gaussian certificate")
    expected = certify_gaussian_nonidentifiability(certificate['forward']['coefficient'], certificate['do_X'])
    if certificate != expected:
        raise ValueError("Gaussian certificate mismatch")
    a = Q(certificate['forward']['coefficient'])
    # Independently compute both structural covariance matrices.
    v = Q(certificate['forward']['residual_variance'])
    forward = [[Q(1),a],[a,a*a+v]]
    reverse = [[a*a+v,a],[a,Q(1)]]
    claimed = [[Q(x) for x in row] for row in certificate['observational_covariance']]
    if forward != reverse or forward != claimed or 1-a*a <= 0:
        raise ValueError("Unequal or degenerate observational laws")
    if a*Q(certificate['do_X']) == 0:
        raise ValueError("Intervention fails to distinguish the SCMs")
    return {'status':'verified', 'conclusion':'refuted', 'finite_sample_error_lower':'1/2'}
