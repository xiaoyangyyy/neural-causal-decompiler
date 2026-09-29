"""Exact intervention coupling bounds for specified linear Gaussian SCMs.

This proves a scoped foundational lemma, not recovery by the learned pipeline.
"""
from fractions import Fraction as Q
from itertools import combinations


def _model(spec):
    a=[[Q(v) for v in row] for row in spec['coefficients']];n=len(a)
    b=[Q(v) for v in spec['intercepts']];mu=[Q(v) for v in spec['noise_means']];sd=[Q(v) for v in spec['noise_stds']]
    if not 1<=n<=8 or any(len(row)!=n for row in a) or any(len(v)!=n for v in (b,mu,sd)) or any(v<0 for v in sd):raise ValueError('SCM dimensions or noise scales')
    if spec.get('noise_joint')!='independent_gaussian':raise ValueError('Independent noise law must be specified; residual marginals are insufficient')
    if any(a[i][i] for i in range(n)):raise ValueError('SCM self loop')
    incoming={i:{j for j in range(n) if a[i][j]} for i in range(n)};order=[]
    while len(order)<n:
        ready=[i for i in range(n) if i not in order and incoming[i].issubset(order)]
        if not ready:raise ValueError('SCM graph is cyclic')
        order.append(min(ready))
    return a,b,mu,sd,order


def certify_linear_gaussian_coupling(reference,realization):
    a,b,mu,sd,order=_model(reference);aa,bb,mm,ss,other_order=_model(realization);n=len(a)
    if a!=aa:raise ValueError('This scoped certificate requires identical known coefficients and correct parents')
    local=[abs(b[i]-bb[i]) for i in range(n)]
    noise=[abs(mu[i]-mm[i])+abs(sd[i]-ss[i]) for i in range(n)]
    interventions=[]
    for count in range(n+1):
        for fixed in combinations(range(n),count):
            d=[Q(0)]*n
            for i in order:
                if i not in fixed:d[i]=local[i]+noise[i]+sum((abs(a[i][j])*d[j] for j in range(n)),Q(0))
            interventions.append({'targets':list(fixed),'coordinate_w1_upper':[str(v) for v in d],'joint_w1_l1_upper':str(sum(d,Q(0)))})
    return {'schema':'ncd.linear-gaussian-intervention-coupling.v1','status':'proved',
        'reference':reference,'realization':realization,'topological_order':order,
        'local_uniform_mechanism_error':[str(v) for v in local],
        'noise_coordinate_coupling_upper':[str(v) for v in noise],'interventions':interventions,
        'input_domain':'all real parent inputs and all real common do-values',
        'intervention_family':'every subset of nodes, arbitrary identical real intervention values in both models',
        'coupling':'Use independent common Z_i~N(0,1), U_i=mu_i+sd_i*Z_i and V_i=mm_i+ss_i*Z_i.',
        'derivation':'E|Z_i|<=sqrt(E Z_i^2)=1; E|U_i-V_i|<=|mu_i-mm_i|+|sd_i-ss_i|. Topological induction gives d_i<=epsilon_i+delta_i+sum_j |a_ij|d_j; common do-values set d_i=0. This coupling bounds coordinate W1 and joint W1 under l1 cost.',
        'correct_parent_sets':'verified for these two explicit SCMs by coefficient equality',
        'noise_law':'explicit independent Gaussian product laws, not empirical residual fits',
        'learned_model_recovery_covered':False,'original_requirement_not_automatically_closed':True}


def verify_linear_gaussian_coupling(certificate):
    expected=certify_linear_gaussian_coupling(certificate['reference'],certificate['realization'])
    if certificate!=expected:raise ValueError('SCM coupling certificate mismatch')
    return {'status':'verified','conclusion':'proved','scope':'two specified explicit SCMs; all shared do-values and target subsets'}
