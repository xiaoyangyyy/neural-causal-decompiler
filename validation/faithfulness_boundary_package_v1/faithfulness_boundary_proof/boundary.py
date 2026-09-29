"""Construction and independently recomputed rational Gaussian certificates.

All noise/SCM statements use ideal real arithmetic and Gaussian probability laws.
No claim is made about discrete PRNG laws, rounded GraphWorld parameters,
almost-sure world performance, identifying interventions, or original closure.
"""
from fractions import Fraction as Q
from itertools import combinations, product
import copy

REQUIREMENTS_SHA = '04d608c94e83c8b77adaca81fcf4942826d3aa7d5d61152d6e895b010fcc724c'
GENERATOR_SHA = '9746dd2c23015e5138c65398065dfc20d1bb34cc8aaca2f5109e326eb1f19883'
SCHEMA = 'ncd.faithfulness-boundary-certificate.v1'
SCOPE = {
    'nodes': [3, 5, 8], 'noise': 'independent centered Gaussian, common standard deviation 7/20',
    'canonical_coefficient_magnitude': ['2/5', '11/10'],
    'observed_scales': 'inside exp([-3/2,3/2]); unknown to observational estimator',
    'arithmetic': 'exact rational model parameters; ideal real SCM and Gaussian laws',
    'information': 'only IID observational rows, no true graph/scales/noises/do data',
    'estimator_family': 'all measurable estimators, including randomized estimators',
    'target': 'one correct graph Markov equivalence class (CPDAG); abstention counts as nonrecovery',
    'not_claimed': ['hardware floating execution', 'exact rounded generator realization',
        'almost-sure or 99-percent random-world performance', 'all estimators given interventions',
        'failure under a justified positive strong-faithfulness floor', 'original R0-R13 closure'],
    'known_math_is_not_project_innovation': True,
}


def validate(n, samples, delta):
    if type(n) is not int or n not in (3, 5, 8):
        raise ValueError('Expected 3/5/8 nodes')
    if type(samples) is not int or not 1 <= samples <= 10**9:
        raise ValueError('Positive bounded integer sample count required')
    if isinstance(delta, (bool, float)):
        raise ValueError('Use exact rational failure probability')
    d = Q(delta)
    if not 0 < d < Q(1, 2):
        raise ValueError('Failure threshold must be strictly between zero and one half')
    return d


def model(n, kind, t):
    weights = [[Q(0) for _ in range(n)] for _ in range(n)]
    scales = [Q(1)] * n
    if kind == 'triangle':
        weights[0][1] = Q(3, 4)
        weights[1][2] = Q(3, 4)
        weights[0][2] = -Q(9, 16) + t
    elif kind == 'collider':
        weights[0][1] = Q(15, 16)
        weights[2][1] = Q(3, 4)
        scales[1], scales[2] = Q(4, 5), Q(5, 4)
    else:
        raise ValueError('Unknown model template')
    return {'weights': [[str(x) for x in row] for row in weights],
        'graph': [[int(bool(x)) for x in row] for row in weights],
        'scales': list(map(str, scales)), 'noise_std': ['7/20'] * n,
        'noise_law': 'product-centered-Gaussian', 'latent_confounding': False,
        'extra_nodes': 'isolated independent Gaussian roots'}


def construct(n=3, samples=96, delta='1/100', mode='population'):
    d = validate(n, samples, delta)
    if mode not in ('population', 'faithful-finite-sample'):
        raise ValueError('Unknown boundary')
    q = (1 - 2*d)/2
    t = Q(0) if mode == 'population' else q / (8*samples)
    a, b = model(n, 'triangle', t), model(n, 'collider', Q(0))
    # Proposed records are calculated by a shared-independent rechecker below;
    # verification recomputes all arithmetic from the structural matrices.
    record = {'schema': SCHEMA, 'mode': mode, 'nodes': n, 'samples': samples,
        'requested_error_upper': str(d), 'perturbation': str(t),
        'triangle': a, 'collider': b, 'scope': copy.deepcopy(SCOPE),
        'requirements_sha256': REQUIREMENTS_SHA, 'generator_sha256': GENERATOR_SHA,
        'original_objective_achieved': False}
    record.update(derive(record))
    return record


def order(graph):
    n = len(graph)
    if any(len(row) != n for row in graph) or any(graph[i][i] for i in range(n)):
        raise ValueError('Invalid graph')
    todo = list(range(n)); result = []
    while todo:
        ready = [j for j in todo if not any(graph[i][j] for i in todo)]
        if not ready:
            raise ValueError('Cycle')
        j = min(ready); todo.remove(j); result.append(j)
    return result


def structural(m):
    w = [[Q(x) for x in row] for row in m['weights']]
    n = len(w); g = m['graph']; s = list(map(Q, m['scales'])); u = list(map(Q, m['noise_std']))
    if len(g) != n or any(len(row) != n for row in w+g) or len(s) != n or len(u) != n:
        raise ValueError('Dimension mismatch')
    if any(type(v) is not int or v not in (0, 1) for row in g for v in row):
        raise ValueError('Binary graph required')
    if g != [[int(bool(v)) for v in row] for row in w]:
        raise ValueError('Wrong structural parent set')
    if m['noise_law'] != 'product-centered-Gaussian' or m['latent_confounding'] is not False:
        raise ValueError('Gaussian independent-noise contract changed')
    if any(x <= 0 for x in s+u) or any(x != Q(7,20) for x in u):
        raise ValueError('Positive frozen independent Gaussian noises required')
    if any(v and not Q(2,5) <= abs(v) <= Q(11,10) for row in w for v in row):
        raise ValueError('Coefficient outside original generator range')
    # Rational inner box suffices: exp(3/2) >= 1+3/2 = 5/2.
    if any(not Q(2,5) < x < Q(5,2) for x in s):
        raise ValueError('Scale outside certified inner box')
    if any(w[i][j] for i in range(n) for j in range(n) if i >= 3 or j >= 3):
        raise ValueError('Extra nodes must remain independent roots')
    transport = [[Q(0) for _ in range(n)] for _ in range(n)]
    for j in order(g):
        transport[j][j] = 1
        for i in range(n):
            for k in range(n): transport[j][k] += w[i][j]*transport[i][k]
    cov = [[s[i]*s[j]*sum(transport[i][k]*transport[j][k]*u[k]**2 for k in range(n))
        for j in range(n)] for i in range(n)]
    return cov, w, s


def invert_det(matrix):
    n = len(matrix); a = [[Q(x) for x in row]+[Q(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    det = Q(1)
    for j in range(n):
        pivot = next((i for i in range(j,n) if a[i][j]), None)
        if pivot is None: raise ValueError('Singular covariance')
        if pivot != j: a[pivot],a[j]=a[j],a[pivot];det=-det
        v=a[j][j];det*=v;a[j]=[x/v for x in a[j]]
        for i in range(n):
            if i!=j:
                v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[j])]
    return [row[n:] for row in a],det


def colliders(g):
    out=[]
    for middle in range(len(g)):
        for i,j in combinations(range(len(g)),2):
            if g[i][middle] and g[j][middle] and not(g[i][j] or g[j][i]):out.append([i,middle,j])
    return out


def equivalence(g):
    n=len(g);edges=[(i,j) for i,j in combinations(range(n),2) if g[i][j] or g[j][i]]
    if len(edges)>3:raise ValueError('Unsupported equivalence enumeration bound')
    dags=[]
    for orientation in product((0,1),repeat=len(edges)):
        candidate=[[0]*n for _ in range(n)]
        for (i,j),bit in zip(edges,orientation):candidate[i if bit==0 else j][j if bit==0 else i]=1
        try:order(candidate)
        except ValueError:continue
        if colliders(candidate)==colliders(g):dags.append(candidate)
    if not dags:raise ValueError('Empty Markov class')
    essential=[[int(any(a[i][j] for a in dags)) for j in range(n)] for i in range(n)]
    return {'cpdag':essential,'dag_count':len(dags),'skeleton':[[int(g[i][j] or g[j][i]) for j in range(n)] for i in range(n)],'unshielded_colliders':colliders(g)}


def conditional_numerators(cov):
    rows=[]
    for i,j in combinations(range(3),2):
        k=next(k for k in range(3) if k not in (i,j))
        rows.append({'pair':[i,j],'unconditional':str(cov[i][j]),
            'condition':[k],'conditional_numerator':str(cov[i][j]*cov[k][k]-cov[i][k]*cov[j][k])})
    return rows


def do_means(m, target, value):
    _,w,s=structural(m);n=len(w);means=[Q(0)]*n
    for j in order(m['graph']):
        means[j] = value/s[j] if j==target else sum(w[i][j]*means[i] for i in range(n))
    return [str(means[i]*s[i]) for i in range(n)]


def derive(c):
    n=c['nodes'];N=c['samples'];d=validate(n,N,c['requested_error_upper']);t=Q(c['perturbation'])
    mode=c['mode']
    if mode not in ('population','faithful-finite-sample') or not 0<=t<=Q(1,16):raise ValueError('Boundary parameter invalid')
    if (mode=='population')!=(t==0):raise ValueError('Wrong faithfulness mode')
    a,b=c['triangle'],c['collider'];A,_,_=structural(a);B,_,_=structural(b)
    if len(A)!=n or len(B)!=n:raise ValueError('Wrong node count')
    invA,detA=invert_det(A);invB,detB=invert_det(B)
    if detA<=0 or detA!=detB:raise ValueError('Unequal positive determinant')
    trace=sum(invB[i][j]*A[j][i] for i in range(n) for j in range(n))
    kl=(trace-n)/2
    if kl!=t*t/2:raise ValueError('KL does not equal exact structural calculation')
    EA,EB=equivalence(a['graph']),equivalence(b['graph'])
    if EA['cpdag']==EB['cpdag'] or EA['dag_count']!=6 or EB['dag_count']!=1:
        raise ValueError('Counterexample needs different Markov classes')
    CA,CB=conditional_numerators(A),conditional_numerators(B)
    # Extras are independent roots; adding them to any conditioning set does
    # not alter these Gaussian independence statements.
    if any(not Q(r['conditional_numerator']) for r in CA+CB):raise ValueError('Unexpected conditional independence')
    if [r['pair'] for r in CB if Q(r['unconditional'])==0]!=[[0,2]]:raise ValueError('Collider faithfulness failed')
    if mode=='population':
        if A!=B or [r['pair'] for r in CA if Q(r['unconditional'])==0]!=[[0,2]]:
            raise ValueError('Observed laws differ')
        bound={'observational_covariances_equal':True,'finite_sample_minimax_error_lower':'1/2',
            'population_unique_mec_claim':'refuted','triangle_faithful':False,'collider_faithful':True}
    else:
        if A==B or any(not Q(r['unconditional']) for r in CA):raise ValueError('Triangle must be faithful and different')
        q=(1-2*d)/2;expected=q/(8*N);tv=q/8
        if t!=expected or N*kl>tv*tv:raise ValueError('Nonuniformity witness/bound changed')
        lower=(1-tv)/2
        if lower<=d:raise ValueError('Lower bound does not strictly contradict target')
        rho2=A[0][2]**2/(A[0][0]*A[2][2])
        if rho2!=t*t/(Q(25,16)+t*t):raise ValueError('Separation formula mismatch')
        bound={'observational_covariances_equal':False,'triangle_faithful':True,'collider_faithful':True,
            'total_variation_upper':str(tv),'iid_KL':str(N*kl),'finite_sample_minimax_error_lower':str(lower),
            'strictly_exceeds_requested_error':True,'unconditional_rho_02_squared':str(rho2),
            'uniform_faithful_mec_recovery_claim':'refuted',
            'unseparated_two_class_family_minimax':'1/2',
            'minimax_argument':'Let t decrease to zero for fixed N; KL and TV vanish, so worst-case error is at least 1/2. A fair independent choice of the two CPDAGs attains upper error 1/2 on this family.',
            'all_N_delta_argument':'For every positive N and 0<delta<1/2 use q=(1-2delta)/2 and t=q/(8N); t<1/16, N*t^2/2<=q^2/64 and (1-q/8)/2>delta. All edge magnitudes exceed 1/2 while rho02 can approach zero.'}
    intervention={'target':2,'value':'1','triangle_mean':do_means(a,2,Q(1)),
        'collider_mean':do_means(b,2,Q(1)),'distinguishing_variable':1}
    if intervention['triangle_mean'][1]!='0' or intervention['collider_mean'][1]!='12/25':raise ValueError('Do responses failed to distinguish')
    return {'status':'refuted','covariance_triangle':[[str(x) for x in row] for row in A],
        'covariance_collider':[[str(x) for x in row] for row in B], 'determinant':str(detA),
        'one_row_KL':str(kl),'triangle_equivalence':EA,'collider_equivalence':EB,
        'gaussian_CI_triangle':CA,'gaussian_CI_collider':CB,'intervention':intervention,
        'bounds':bound,'evidence_class':'true_causal_correctness',
        'proof_dependencies':['linear images of independent Gaussians determined by mean and covariance',
            'Gaussian conditional independence iff corresponding conditional covariance is zero',
            'graph Markov equivalence iff same skeleton and unshielded colliders',
            'IID KL adds over rows; equal covariance determinants remove log determinant ratio',
            'two disjoint correct-output events have summed probability <=1+TV, including randomized estimators',
            'TV<=sqrt(KL), obtained from Jensen for Bhattacharyya affinity and Cauchy-Schwarz; rational bound encloses sqrt'],
        'uncovered':list(SCOPE['not_claimed'])}


def verify(c):
    if c.get('schema')!=SCHEMA or c.get('scope')!=SCOPE or c.get('original_objective_achieved') is not False:
        raise ValueError('Scope/schema changed or original claim promoted')
    if c.get('requirements_sha256')!=REQUIREMENTS_SHA or c.get('generator_sha256')!=GENERATOR_SHA:
        raise ValueError('Original source identity changed')
    # Shape/model values are frozen separately from arithmetic. The rechecker
    # uses structural covariance, Gaussian elimination, and full 3-edge DAG
    # enumeration rather than trusting proposed covariance or CPDAG fields.
    n=c['nodes'];validate(n,c['samples'],c['requested_error_upper']);t=Q(c['perturbation'])
    if c['triangle']!=model(n,'triangle',t) or c['collider']!=model(n,'collider',Q(0)):
        raise ValueError('Changed structural witness')
    recomputed=derive(c)
    keys={'schema','mode','nodes','samples','requested_error_upper','perturbation','triangle','collider',
        'scope','requirements_sha256','generator_sha256','original_objective_achieved'}
    if set(c)!=keys|set(recomputed) or any(c[k]!=v for k,v in recomputed.items()):raise ValueError('Certificate mismatch')
    return {'status':'verified','conclusion':'refuted','mode':c['mode'],'nodes':n,
        'minimax_error_lower':recomputed['bounds']['finite_sample_minimax_error_lower'],
        'original_objective_achieved':False}
