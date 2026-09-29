"""Conditional DAG error propagation; local evidence is a separate obligation."""
from fractions import Fraction as Q
from .graphs import topological_order


def certify_scm_error(graph, local_errors, lipschitz, noise_distances, interventions=(), premises=None):
    if any(x not in (0,1,False,True) for row in graph for x in row):raise ValueError('Expected unweighted DAG adjacency')
    n = len(graph)
    order = topological_order(graph)
    eta, nu = [Q(x) for x in local_errors], [Q(x) for x in noise_distances]
    L = [[Q(x) for x in row] for row in lipschitz]
    if len(eta)!=n or len(nu)!=n or len(L)!=n or any(len(row)!=n for row in L):
        raise ValueError("SCM error dimensions")
    if any(x<0 for x in eta+nu+[x for row in L for x in row]):
        raise ValueError("Negative error or Lipschitz bound")
    if any(L[i][j] and not graph[i][j] for i in range(n) for j in range(n)):
        raise ValueError("Lipschitz dependence outside graph")
    do = sorted(set(interventions))
    if any(type(i) is not int or not 0<=i<n for i in do):
        raise ValueError("Invalid intervention index")
    errors = [Q(0)]*n
    for j in order:
        if j not in do:
            errors[j] = eta[j]+nu[j]+sum((L[i][j]*errors[i] for i in range(n)), Q(0))
    return {'schema':'ncd.conditional-scm-error.v1','status':'proved-conditionally',
        'graph':graph,'local_errors':list(map(str,eta)), 'lipschitz':[[str(x) for x in r] for r in L],
        'noise_distances':list(map(str,nu)), 'interventions':do, 'order':order,
        'coordinate_error_bounds':list(map(str,errors)), 'joint_wasserstein_l1_upper':str(sum(errors)),
        'premises':premises or [],
        'assumptions':['same correct DAG and parent sets','same intervention values on both SCMs',
            'coupled product exogenous laws with E|U_j-V_j| bounded by noise_distances[j]',
            'uniform local mechanism error on a domain containing both coupled executions',
            'coordinatewise Lipschitz bounds on that domain'],
        'boundary':'This checks the conditional induction arithmetic; it does not prove supplied local errors, graph correctness, noise independence, or the existence of the claimed noise coupling.'}


def verify_scm_error(certificate):
    expected = certify_scm_error(certificate['graph'],certificate['local_errors'],certificate['lipschitz'],
        certificate['noise_distances'],certificate['interventions'],certificate['premises'])
    if certificate != expected:
        raise ValueError("Conditional SCM certificate mismatch")
    return {'status':'verified','conclusion':'proved-conditionally','premises_verified':False}
