"""Strict lower error bounds for one fixed Gaussian MEC estimator.

No experimental observations enter these certificates. Every stochastic
premise is a stated ideal mathematical law, not a learned-noise certificate.
"""
from fractions import Fraction as Q
from math import factorial
import json,hashlib
from finite_graph_proof.boundary import certify as base_certify,verify as base_verify

DERIVATIONS={
 'signed_law':'Under model edge01, sign(a)*X0*(X1-X2)=abs(a)*X0^2+sqrt(2)*X0*Z, X0,Z independent standard Gaussian; use a=1/2. The same error law holds in the reflected edge02 model up to its measure-zero tie.',
 'strict_region':'For 0<X0<=1/2 and -1<=Z<=-1/2, a*X0+sqrt(2)*Z<=1/4-1/2<0. Thus the statistic is strictly negative. The excluded endpoint X0=0 has zero Gaussian measure.',
 'region_density':'pi<4 follows by integrating 1/(1+x*x)<1 over (0,1). Joint normalization 1/(2*pi)>1/8. On the rectangle (x*x+z*z)/2<=5/8 and exp(-u)>=1-u, so density>3/64. Area=1/4.',
 'conditional_error':'For S=sum(X0_i^2), the signed statistic has mean a*S and conditional variance 2*S. S>0 almost surely, E[S]=samples. Conditional error is f(S)=Phi(-a*sqrt(S/2)).',
 'jensen':'For c=a/sqrt(2)>0 and s>0, f_second(s)=c*phi(c*sqrt(s))*(1+c*c*s)/(4*s^(3/2))>0. The bounded continuous extension to zero is convex; Jensen gives error>=Phi(-a*sqrt(samples/2)).',
 'mills_lower':'Integration by parts yields I(t)=phi(t)/t-integral_t_infinity(phi(u)/u^2)du. Since u>t almost everywhere, the integral<I(t)/t^2, hence I(t)>phi(t)*t/(t*t+1).',
 'e_upper':'sum_{k=0}^4 1/k!=65/24; the remaining positive tail is <=(1/120)/(1-1/6)=1/100. Thus e<=1631/600<11/4.',
 'normal_tail':'For samples=96,a=1/2, t*t=12, t>17/5, sqrt(2*pi)<3 and exp(-t*t/2)> (4/11)^6. Therefore error>17/195*(4/11)^6.',
 'family_counterexample':'Take family_size independent datasets, each with the declared IID rows under model edge01. This is an allowed joint experiment even if independence between events is not required by the universal family contract. Probability of at least one wrong graph is 1-(1-p)^family_size, strictly above 1-(1-error_lower)^family_size.',
}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def certify(samples=1,family_size=1,delta='1/100'):
 if (samples,family_size) not in ((1,1),(96,100)) or type(samples)is not int or type(family_size)is not int or Q(delta)!=Q(1,100):
  raise ValueError('Exactly the frozen single-row or 96-row/100-event contract is supported')
 base=base_certify('1/2',samples,delta,family_size);base_verify(base)
 q=Q(3,256) if samples==1 else Q(17,195)*Q(4,11)**6
 family_lower=1-(1-q)**family_size
 if family_lower<=Q(delta):raise ValueError('A lower bound must strictly contradict the requested success probability')
 return {'schema':'ncd.fixed-gaussian-estimator-failure.v1','status':'refuted-in-stated-fixed-estimator-scope','samples':samples,'a':'1/2','family_size':family_size,'delta':'1/100',
  'base_certificate':base,'base_certificate_sha256':digest(base),'derivations':DERIVATIONS.copy(),
  'primitive_bounds':{'pi_strict_upper':'4','e_upper':'11/4','e_series_partial':'65/24','e_series_tail_upper':'1/100','rectangle_area':'1/4','rectangle_quadratic_upper':'5/8','rectangle_density_strict_lower':'3/64','conditional_c_squared':'1/8','tail_t_squared':'12','tail_t_strict_lower':'17/5'},
  'per_event_error_strict_lower':str(q),'family_failure_strict_lower':str(family_lower),
  'counterexample_joint_law':{'model':'edge01','between_events':'independent product of exactly family_size datasets','within_events':'Exactly samples independent standard-Gaussian exogenous rows per dataset','estimator':base['estimator']['statistic']},
  'requested_contract':'For every permitted collection of family_size recovery events, probability all returned MECs are correct is at least 1-delta',
  'refutation_quantifier':'This specified estimator; an explicit independent-event experiment in the same specified Gaussian SCM family fails the requested uniform confidence contract',
  'all_estimators_refuted':False,'actual_neural_graph_recovery_refuted':False,'six_event_96_row_theorem_refuted':False,
  'sampling_independence_is_assumed_mathematical_model':True,'classical_probability_theory_not_project_innovation':True,
  'hardware_semantics_covered':False,'original_claim_closed':False,'original_objective_achieved':False}

def verify(c):
 if not isinstance(c,dict):raise ValueError('Certificate object')
 n,k=c.get('samples'),c.get('family_size')
 if type(n)is not int or type(k)is not int or (n,k) not in ((1,1),(96,100)) or c.get('a')!='1/2' or c.get('delta')!='1/100':raise ValueError('Changed recovery scope')
 base=c.get('base_certificate');base_verify(base)
 if (base['a'],base['samples'],base['family_size'],base['delta'])!=('1/2',n,k,'1/100') or digest(base)!=c['base_certificate_sha256']:raise ValueError('Changed verified dependency')
 # Independent covariance/conditional-variance derivation; no constructor call.
 covariance=[[Q(x) for x in row] for row in base['models'][0]['covariance']]
 variance_difference=covariance[1][1]+covariance[2][2]-2*covariance[1][2]
 regression=covariance[0][1]-covariance[0][2]
 if covariance[0][0]!=1 or regression!=Q(1,2) or variance_difference-regression**2!=2:raise ValueError('Gaussian conditional law changed')
 partial=sum((Q(1,factorial(j)) for j in range(5)),Q(0))
 tail=Q(1,factorial(5))/(1-Q(1,6))
 if partial!=Q(65,24) or tail!=Q(1,100) or partial+tail>=Q(11,4):raise ValueError('Invalid elementary e enclosure')
 bounds={'pi_strict_upper':'4','e_upper':str(Q(11,4)),'e_series_partial':str(partial),'e_series_tail_upper':str(tail),'rectangle_area':str(Q(1,2)**2),'rectangle_quadratic_upper':str((Q(1,2)**2+1)/2),'rectangle_density_strict_lower':str((1-Q(5,8))/8),'conditional_c_squared':str(regression**2/2),'tail_t_squared':str(regression**2*96/2),'tail_t_strict_lower':'17/5'}
 if bounds!=c['primitive_bounds'] or c['derivations']!=DERIVATIONS:raise ValueError('Proof relation or primitive bound changed')
 if Q(17,5)**2>=12 or Q(1,4)-Q(1,2)>=0:raise ValueError('Strict tail/rectangle relation')
 q=Q(3,64)*Q(1,4) if n==1 else Q(17,5)/13/3*Q(4**6,11**6)
 remaining=Q(1)
 for _ in range(k):remaining*=1-q
 family=1-remaining
 if str(q)!=c['per_event_error_strict_lower'] or str(family)!=c['family_failure_strict_lower'] or family<=Q(1,100):raise ValueError('False strict confidence counterexample')
 fixed={'schema':'ncd.fixed-gaussian-estimator-failure.v1','status':'refuted-in-stated-fixed-estimator-scope','counterexample_joint_law':{'model':'edge01','between_events':'independent product of exactly family_size datasets','within_events':'Exactly samples independent standard-Gaussian exogenous rows per dataset','estimator':base['estimator']['statistic']},
  'requested_contract':'For every permitted collection of family_size recovery events, probability all returned MECs are correct is at least 1-delta',
  'refutation_quantifier':'This specified estimator; an explicit independent-event experiment in the same specified Gaussian SCM family fails the requested uniform confidence contract',
  'all_estimators_refuted':False,'actual_neural_graph_recovery_refuted':False,'six_event_96_row_theorem_refuted':False,'sampling_independence_is_assumed_mathematical_model':True,'classical_probability_theory_not_project_innovation':True,'hardware_semantics_covered':False,'original_claim_closed':False,'original_objective_achieved':False}
 if any(c.get(key)!=value for key,value in fixed.items()) or set(c)!=set(fixed)|{'samples','a','family_size','delta','base_certificate','base_certificate_sha256','derivations','primitive_bounds','per_event_error_strict_lower','family_failure_strict_lower'}:raise ValueError('Unsupported strengthened or omitted scope')
 return {'status':'verified','conclusion':'refuted-scoped','samples':n,'family_size':k,'per_event_error_strict_lower':str(q),'family_failure_strict_lower':str(family),'requested_recovery_success_probability_gate':'refuted','independent_product_counterexample_verified':True,'all_estimators_refuted':False,'original_claim_closed':False,'original_objective_achieved':False}
