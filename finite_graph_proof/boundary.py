from fractions import Fraction as Q

def ceil(value):return -(-value.numerator//value.denominator)
def covariance(A):return [[sum((x*y for x,y in zip(a,b)),Q(0)) for b in A] for a in A]
def encode(matrix):return [[str(v) for v in row] for row in matrix]
def decode(matrix):return [[Q(v) for v in row] for row in matrix]

def certify(a='1/2',samples=96,delta='1/100',family_size=1):
 a=Q(a);delta=Q(delta)
 if not 0<abs(a)<=1 or max(a.numerator.bit_length(),a.denominator.bit_length())>256:raise ValueError('Nonzero |a|<=1, rational parameter budget <=256 bits required')
 if max(delta.numerator.bit_length(),delta.denominator.bit_length())>256:raise ValueError('Confidence rational budget <=256 bits required')
 if type(samples)is not int or not 1<=samples<=1000000 or type(family_size)is not int or not 1<=family_size<=1000000 or not 0<delta<1:raise ValueError('Finite sample/confidence contract')
 A=[[Q(1),Q(0),Q(0)],[a,Q(1),Q(0)],[Q(0),Q(0),Q(1)]]
 B=[[Q(1),Q(0),Q(0)],[Q(0),Q(1),Q(0)],[a,Q(0),Q(1)]]
 var=2+2*a*a;bound=var/(var+samples*a*a);local_delta=delta/family_size
 required=ceil(var/(a*a)*(1/local_delta-1))
 radius=Q(1,4);quadratic=radius*radius*(2+(1+abs(a))**2)/2
 return {'schema':'ncd.finite-gaussian-mec-boundary.v1','status':'proved-in-stated-two-model-family','a':str(a),'samples':samples,'delta':str(delta),'family_size':family_size,
 'models':[{'label':'edge01','mixing':encode(A),'covariance':encode(covariance(A)),'graph':[[0,1,0],[0,0,0],[0,0,0]]},
           {'label':'edge02','mixing':encode(B),'covariance':encode(covariance(B)),'graph':[[0,0,1],[0,0,0],[0,0,0]]}],
 'noise_law':'Independent standard centered Gaussian U0,U1,U2, independently repeated for each observed row; an ideal mathematical sampling model, not a PRNG-independence certificate.',
 'population':'The two covariance laws differ and their faithful one-edge skeletons give distinct Gaussian Markov equivalence classes. Population class identification is possible within this specified family.',
 'overlap':{'cube_radius':str(radius),'maximum_residual_quadratic':str(quadratic),'jacobian_determinant':'1','normalization_lower':'1/24','exponential_lower':'1/3','density_lower':'1/72','cube_volume':'1/8',
            'normalization_proof':'pi/4 = integral_0^1 1/(1+x*x) dx <=1; sqrt(8)<=3; hence (2*pi)^(3/2)<=24.',
            'exponential_proof':'e = sum_k 1/k! <3 using the geometric upper bound on the factorial tail; q<=1 implies exp(-q)>=1/3.',
            'joint_overlap_lower':{'base':'1/576','exponent':samples},'minimax_error_lower':{'multiplier':'1/2','base':'1/576','exponent':samples},
            'strictly_positive_for_every_finite_samples':True},
 'universal_zero_error_recovery':'refuted for every possibly randomized estimator required to return the single correct Markov equivalence class from finitely many observational rows',
 'lower_bound_reason':'On the cube product, each n-row density is >=(1/72)^n and its volume is (1/8)^n. Average two-model error is >=half their density overlap; worst-model error is at least that average.',
 'estimator':{'statistic':'T=mean(X0*(X1-X2)); return edge01 iff sign(a)*T>=0, else edge02','row_means':[str(a),str(-a)],'row_variance':str(var),
              'uniform_error_upper':str(bound),'per_claim_delta':str(local_delta),'sufficient_samples':required,
              'requested_confidence_gate':'proved' if bound<=local_delta else 'unresolved-by-this-bound',
              'tail_proof':'For centered W with variance s2, Markov on (W-t)^2 and t=s2/abs(mu) gives P(W<=-abs(mu))<=s2/(s2+mu*mu); use s2=row_variance/n and the reflected upper tail for the other model.'},
 'conditions':['Exactly the declared two SCM hypotheses, known nonzero a (including its sign)','Three observed variables; acyclic, no latent confounding, independent standard Gaussian exogenous laws','IID observations; a fixed estimator before evaluation','For K simultaneous recovery events use per-event delta/K and the union bound'],
 'not_refuted':['Population Markov equivalence recovery','Statistical recovery at declared nonzero error','Candidate sets that retain both different equivalence classes','Other information access such as noiseless or coupled interventional contrasts'],
 'not_proved':['Graph recovery for all original 3/5/8-node worlds','Performance of the actual trained neural graph classifiers','Empirical exogenous independence or arbitrary noise-shift robustness'],
 'certificate_input_budget':{'coefficient_and_confidence_bits':256,'maximum_samples_and_family_size':1000000,'theorem_formula_valid_for_every_positive_finite_n':True},
 'classical_probability_bounds_not_project_innovation':True,'original_claim_closed':False,'original_objective_achieved':False}

def verify(c):
 if c['schema']!='ncd.finite-gaussian-mec-boundary.v1':raise ValueError('Wrong theorem schema')
 a=Q(c['a']);n=c['samples'];means=[];variances=[]
 for k,m in enumerate(c['models']):
  A=decode(m['mixing']);S=covariance(A)
  if encode(S)!=m['covariance']:raise ValueError('Covariance mismatch')
  if any(A[i][i]!=1 or any(A[i][j] for j in range(i+1,3)) for i in range(3)):raise ValueError('Jacobian/positive-definiteness premise')
  mu=S[0][1]-S[0][2]
  # Independent route: Gaussian fourth moments of A=X0, B=X1-X2.
  # Var(A*B)=Var(A)*Var(B)+Cov(A,B)^2.
  variance=S[0][0]*(S[1][1]+S[2][2]-2*S[1][2])+mu*mu
  means.append(mu);variances.append(variance)
 if means!=[a,-a] or variances[0]!=variances[1] or c['estimator']['row_means']!=list(map(str,means)) or Q(c['estimator']['row_variance'])!=variances[0]:raise ValueError('Moment proof mismatch')
 upper=variances[0]/(variances[0]+n*means[0]**2)
 if str(upper)!=c['estimator']['uniform_error_upper']:raise ValueError('Probability bound mismatch')
 # Independently enclose each inverse structural residual on the common cube.
 r=Q(c['overlap']['cube_radius']);qmax=[]
 for m in c['models']:
  G=m['graph'];qmax.append(sum(((r+sum(abs(a)*r for parent in range(3) if G[parent][j]))**2 for j in range(3)),Q(0))/2)
 if max(qmax)>1 or str(max(qmax))!=c['overlap']['maximum_residual_quadratic']:raise ValueError('Density event bound mismatch')
 expected=certify(c['a'],n,c['delta'],c['family_size'])
 if c!=expected:raise ValueError('Theorem, graph, event, scope or confidence mismatch')
 return {'status':'verified','population_family_identifiable':True,'zero_error_finite_sample_equivalence_class_recovery':'refuted','all_estimators_covered':True,
 'uniform_error_upper':str(upper),'sufficient_samples':c['estimator']['sufficient_samples'],'requested_confidence_gate':c['estimator']['requested_confidence_gate'],
 'original_objective_achieved':False}
