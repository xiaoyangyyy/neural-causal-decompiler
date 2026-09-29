"""Sufficient growth/moment induction; a failed sufficient bound is unresolved."""
from fractions import Fraction as Q
from .compact import order,rational,fingerprint

FINITE_ALL={'gaussian','laplace','uniform','mixture'}


def derive(world,q='1'):
    q=Q(q)
    if q<=0:raise ValueError('Positive moment order required')
    g=world['graph'];topo=order(g);n=len(g);eq=world['equations'];scales=list(map(rational,world['scales'] or [1]*n))
    if len(eq)!=n or len(scales)!=n or any(s<=0 for s in scales):raise ValueError('Invalid world dimensions/scales')
    family=world['noise_family']
    if family not in FINITE_ALL|{'student'}:raise NotImplementedError('Unknown true noise law')
    noise_laws=[('gaussian' if world.get('root_shift') and not any(g[i][j] for i in range(n)) else family) for j in range(n)]
    vectors=[[0]*n for _ in range(n)];constants=[Q(0)]*n;rows=[]
    for j in topo:
        d=[0]*n;d[j]=1;constant=Q(1);used=set()
        for term in eq[j]:
            op=term['operator'];p=term['parents'];coef=abs(rational(term['coefficient']))
            arity={'linear':1,'square':1,'sin':1,'cos':1,'tanh':1,'interaction':2}
            if op not in arity or len(p)!=arity[op] or any(type(i) is not int or not 0<=i<n or not g[i][j] for i in p):raise ValueError('Unregistered or incorrect true mechanism')
            used.update(p)
            if op in ('sin','cos','tanh'):v=[0]*n;c=Q(1)
            elif op=='linear':v=vectors[p[0]];c=constants[p[0]]
            elif op=='square':v=[2*x for x in vectors[p[0]]];c=constants[p[0]]**2
            else:v=[a+b for a,b in zip(vectors[p[0]],vectors[p[1]])];c=constants[p[0]]*constants[p[1]]
            d=[max(a,b) for a,b in zip(d,v)];constant+=coef*c
        if used!={i for i in range(n) if g[i][j]}:raise ValueError('True graph and equations disagree')
        constants[j]=max(constant,1/scales[j]);vectors[j]=d
        obligations=[{'noise_node':k,'noise_family':noise_laws[k],'required_moment':str(q*power),
            'available':noise_laws[k] in FINITE_ALL or q*power<5} for k,power in enumerate(d) if power]
        good=all(o['available'] for o in obligations)
        rows.append({'node':j,'growth_exponents':d,'canonical_growth_constant':str(constants[j]),
            'observed_growth_constant':str(scales[j]*constants[j]),'moment_obligations':obligations,
            'status':'proved-scoped' if good else 'unresolved-by-sufficient-bound'})
    return {'schema':'ncd.true-scm-moment-induction.v1','world_sha256':fingerprint(world),'moment_order':str(q),
        'status':'proved-scoped' if all(r['status']=='proved-scoped' for r in rows) else 'unresolved-by-sufficient-bound',
        'rows':rows,'noise_laws':noise_laws,
        'intervention_scope':'All compatible subsets with observed do values in [-1,1]',
        'argument':'Canonical |Z_j| is bounded by C_j times product_k (1+|U_k|)^d_jk. Independent ideal exogenous noises factor the needed moment; Student5 has absolute moments only below 5. Failure of this sufficient condition does not prove divergence.',
        'true_noise_independence':'a property of the specified mathematical generator, not inferred from data',
        'semantic_scope':'Ideal continuous mathematical noise laws; finite-state PRNG and hardware execution are not certified',
        'original_claim_closed':False,'original_objective_achieved':False}


def verify(world,certificate):
    expected=derive(world,certificate['moment_order'])
    if expected!=certificate:raise ValueError('Moment induction certificate mismatch')
    return {'status':'verified','conclusion':expected['status'],'original_objective_achieved':False}
