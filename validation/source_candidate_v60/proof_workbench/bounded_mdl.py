"""Finite grammar enumeration and actual approximate-minimum nonuniqueness.

No incomplete enumeration becomes a minimality certificate. Function equality
and equality of intervention-bearing internal computations stay separate.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib,json,time
from ncd.cdir import Node
from ncd.io import digest
from ncd.proof_intervals import Interval
from ncd.frozen_mechanism_proof import export_mechanism,neural_interval,certify_mechanism,verify_mechanism
from .polynomial_semantics import canonical_polynomial


def _key(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def grammar_expressions(grammar):
    n=grammar['variables'];bound=grammar['max_cost'];constants=[Q(v) for v in grammar['constants']];operations=grammar['operations']
    if type(n)is not int or not 1<=n<=8 or type(bound)is not int or not 1<=bound<=15 or len(constants)!=len(set(constants)) or any(Q(float(v))!=v for v in constants):raise ValueError('Finite dyadic grammar')
    if len(operations)!=len(set(operations)) or any(op not in ('add','sub','mul','square') for op in operations):raise ValueError('Finite polynomial operators')
    buckets={}
    for cost in range(1,bound+1):
        current=[]
        if cost==1:
            current=[{'op':'constant','value':float(v)} for v in constants]+[{'op':'var','index':j} for j in range(n)]
            for expression in current:yield cost,expression
        else:
            for operation in operations:
                if operation=='square':
                    for a in buckets[cost-1]:
                        expression={'op':operation,'args':[a]};current.append(expression);yield cost,expression
                else:
                    for left in range(1,cost-1):
                        right=cost-1-left
                        for a in buckets[left]:
                            for b in buckets[right]:
                                expression={'op':operation,'args':[a,b]};current.append(expression);yield cost,expression
        buckets[cost]=current


def certify_polynomial_mdl(target,grammar,max_candidates=50000,seconds=30,prefix_limit=None):
    if type(max_candidates)is not int or max_candidates<1 or seconds<=0:raise ValueError('Enumeration budget')
    wanted=canonical_polynomial(target,grammar['variables']);deadline=time.monotonic()+seconds
    transcript=hashlib.sha256();counts={};minimum=None;winners=[];checked=0;complete=True
    for cost,expression in grammar_expressions(grammar):
        if checked>=min(max_candidates,prefix_limit if prefix_limit is not None else max_candidates) or time.monotonic()>deadline:complete=False;break
        normal=canonical_polynomial(expression,grammar['variables'])
        transcript.update((_key(expression)+'\n'+_key(normal)+'\n').encode())
        counts[str(cost)]=counts.get(str(cost),0)+1;checked+=1
        if normal==wanted:
            if minimum is None:minimum=cost
            if cost==minimum:winners.append(expression)
    # A full frozen bounded grammar is replayed for closure; otherwise only an
    # upper bound may be stated even if the prefix happens to contain a winner.
    return {'schema':'ncd.finite-polynomial-mdl.v1','status':'proved' if complete and minimum is not None else 'refuted' if complete else 'unresolved',
        'target':target,'grammar':grammar,'max_candidates':max_candidates,'enumerated_programs':checked,'complete':complete,
        'counts_by_cost':counts,'transcript_sha256':transcript.hexdigest(),'target_normal_form':wanted,
        'minimum_cost':minimum if complete else None,'current_upper_bound':minimum,
        'minimum_programs':winners if complete else [],'minimum_syntax_unique':len(winners)==1 if complete and minimum is not None else None,
        'exact_target_function_unique':'all equal to this target polynomial; this says nothing about internal causal computations',
        'scope':'exact all-real polynomial function fidelity within this frozen finite constant/operator grammar and max AST cost',
        'canonical_normal_form_unique':True,'internal_semantic_uniqueness_proved':False}


def verify_polynomial_mdl(certificate):
    # Recreate the claimed prefix without trusting any enumerated candidate or
    # canonical form. One extra item determines whether the prefix was complete.
    replay=certify_polynomial_mdl(certificate['target'],certificate['grammar'],certificate['max_candidates'],seconds=43200,prefix_limit=certificate['enumerated_programs'])
    if certificate!=replay:raise ValueError('Finite MDL enumeration coverage or result mismatch')
    return {'status':'verified','conclusion':certificate['status'],'minimum_cost':certificate['minimum_cost'],'original_R4_R12_closed':False}


def certify_approximate_root_nonuniqueness(checkpoint,domain=None,epsilon='1/100'):
    network=export_mechanism(checkpoint)
    if network['parents']:raise ValueError('This certificate requires an actual constant root neural mechanism')
    domain=domain or [['-1','1']]*3;scale=Q(network['output_scale']);eps=Q(epsilon)
    if eps<=0:raise ValueError('Positive approximation tolerance')
    value=neural_interval(network,[Interval.point(0) for _ in domain]);mid=(value.lo+value.hi)/2
    constants=[float(mid-scale*eps/2),float(mid+scale*eps/2)]
    if constants[0]==constants[1]:raise ValueError('Distinct representable constants not separated')
    proofs=[]
    for c in constants:
        expression={'op':'constant','value':c}
        proof=certify_mechanism(network,expression,domain,epsilon,scale=str(scale),max_boxes=1,seconds=30)
        if proof['status']!='proved':raise ValueError('Candidate constant is not uniformly faithful')
        verify_mechanism(network,expression,proof);proofs.append({'expression':expression,'proof':proof})
    return {'schema':'ncd.approximate-minimum-semantic-nonuniqueness.v1','status':'refuted',
        'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'network':network,'domain':domain,'epsilon':str(eps),
        'grammar':{'variables':len(domain),'constants':[str(Q(c)) for c in constants],
            'operations':['add','sub','mul','square'],'cost':'positive unit AST-node cost; all constants and variables cost one'},
        'candidates':proofs,'minimum_cost':1,'shorter_grammar_program_count':0,
        'distinct_functions_witness':{'input':['0']*len(domain),'output_difference':str(abs(Q(constants[1])-Q(constants[0])))},
        'refuted_statement':'The shortest programs satisfying the declared approximate neural-fidelity tolerance are always unique even as mathematical functions.',
        'scope':'this actual frozen root network, this frozen grammar, the entire declared box and the positive stated tolerance',
        'exact_minimal_realization_refuted':False,'device_rounding_covered':False,'original_R12_closed':False}


def verify_approximate_root_nonuniqueness(certificate):
    expected=certify_approximate_root_nonuniqueness(certificate['checkpoint'],certificate['domain'],certificate['epsilon'])
    if expected!=certificate:raise ValueError('Approximate minimum nonuniqueness certificate mismatch')
    return {'status':'verified','conclusion':'refuted','minimum_cost':1,'original_R12_closed':False}
