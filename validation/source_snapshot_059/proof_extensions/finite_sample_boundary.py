"""Population-distinct Gaussian SCMs can still be uniformly hard at finite n."""
from fractions import Fraction as Q
from ncd.proof_intervals import Interval


def certify_finite_sample_boundary(samples=96,coefficient='1/100',nodes=3,error_target='1/100'):
    beta=Q(coefficient);target=Q(error_target)
    if type(samples)is not int or samples<1 or type(nodes)is not int or nodes not in (2,3,5,8) or not 0<abs(beta)<=1 or not 0<=target<Q(1,2):raise ValueError('Finite-sample boundary contract')
    covariance_a=[[Q(int(i==j)) for j in range(nodes)] for i in range(nodes)]
    covariance_b=[row[:] for row in covariance_a];covariance_b[0][1]=covariance_b[1][0]=beta;covariance_b[1][1]=1+beta**2
    # Under B, X0~N(0,1), X1|X0~N(beta*X0,1). KL(B||A)=beta^2/2.
    kl=Q(samples)*beta**2/2;tv_upper=min(Q(1),(Interval.point(kl)/2).sqrt().hi)
    lower=(1-tv_upper)/2
    return {'schema':'ncd.finite-sample-separation-boundary.v1','status':'refuted' if lower>target else 'unresolved',
        'samples':samples,'nodes':nodes,'coefficient':str(beta),'error_target':str(target),
        'model_a':{'edge_0_1':False,'covariance':[[str(v) for v in row] for row in covariance_a]},
        'model_b':{'edge_0_1':True,'covariance':[[str(v) for v in row] for row in covariance_b]},
        'noise':'all exogenous noises mutually independent N(0,1); extra nodes independent roots',
        'population_laws_distinct':True,'single_row_kl_b_to_a':str(beta**2/2),'sample_kl_b_to_a':str(kl),
        'total_variation_upper':str(tv_upper),'minimax_graph_error_lower':str(lower),
        'refuted_statement':'an observational estimator has error at most error_target for both specified graph models at this sample size',
        'scope':'all randomized estimators with only these n observations and model-independent side information',
        'dependencies':['Gaussian conditional KL calculation','iid KL tensorization','Pinsker inequality','two-point testing lower bound'],
        'does_not_refute_benchmarks_excluding_this_small_edge':True,'interventional_recovery_covered':False}


def verify_finite_sample_boundary(certificate):
    expected=certify_finite_sample_boundary(certificate['samples'],certificate['coefficient'],certificate['nodes'],certificate['error_target'])
    if certificate!=expected:raise ValueError('Finite-sample separation boundary certificate mismatch')
    return {'status':'verified','conclusion':certificate['status'],'minimax_graph_error_lower':certificate['minimax_graph_error_lower'],
        'original_scope_not_automatically_closed':True}
