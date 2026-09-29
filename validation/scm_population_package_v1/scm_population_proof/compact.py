"""Finite-support empirical SCM induction. No truth/identification inference."""
from fractions import Fraction as Q
import math,hashlib,json


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rational(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite stored number required')
    return Q(value)


def order(graph):
    n=len(graph)
    if n not in (3,5,8) or any(len(row)!=n for row in graph) or any(type(v) not in (int,bool) or v not in (0,1) for row in graph for v in row) or any(graph[j][j] for j in range(n)):
        raise ValueError('Expected loop-free 3/5/8-node DAG')
    pending=list(range(n));result=[]
    while pending:
        ready=[j for j in pending if not any(graph[i][j] for i in pending)]
        if not ready:raise ValueError('Directed cycle')
        j=min(ready);pending.remove(j);result.append(j)
    return result


def interval(expr,bounds,allowed,depth=0):
    if depth>24:raise ValueError('AST depth budget exhausted')
    op=expr.get('op')
    if op=='constant':
        if set(expr)!={'op','value'}:raise ValueError('Changed constant fields')
        v=rational(expr['value']);return (v,v)
    if op=='var':
        if set(expr)!={'op','index'} or type(expr['index']) is not int or expr['index'] not in allowed:raise ValueError('Undeclared parent')
        return bounds[expr['index']]
    arity={'add':2,'sub':2,'mul':2,'square':1,'sin':1,'cos':1,'tanh':1,'abs':1,'div':2,'sqrt':1,'log':1}
    if op not in arity:raise NotImplementedError('Unsupported SCM operator: '+str(op))
    if set(expr)!={'op','args'} or len(expr['args'])!=arity[op]:raise ValueError('Invalid mechanism AST')
    a=[interval(child,bounds,allowed,depth+1) for child in expr['args']]
    if op=='add':out=(a[0][0]+a[1][0],a[0][1]+a[1][1])
    elif op=='sub':out=(a[0][0]-a[1][1],a[0][1]-a[1][0])
    elif op=='mul':v=[x*y for x in a[0] for y in a[1]];out=(min(v),max(v))
    elif op=='square':lo,hi=a[0];out=(Q(0) if lo<=0<=hi else min(lo*lo,hi*hi),max(lo*lo,hi*hi))
    elif op in ('sin','cos','tanh'):out=(-Q(1),Q(1))
    elif op=='abs':lo,hi=a[0];out=(Q(0) if lo<=0<=hi else min(abs(lo),abs(hi)),max(abs(lo),abs(hi)))
    elif op=='div':v=max(map(abs,a[0]))/Q(1e-8);out=(-v,v)
    elif op=='sqrt':v=max(a[0][1],Q(0));out=(Q(0),max(Q(1),v)) if v else (Q(0),Q(0))
    else:v=max(max(map(abs,a[0])),1/Q(1e-12));out=(-v,v)
    if any(max(v.numerator.bit_length(),v.denominator.bit_length())>1000000 for v in out):raise ValueError('Exact-integer proof budget exhausted')
    return out


def parents(expr):
    if expr['op']=='var':return {expr['index']}
    return set().union(*(parents(e) for e in expr.get('args',[])))


def derive(model,do_bounds=('-1','1')):
    graph=model['source_graph'];topo=order(graph);n=len(graph)
    if model.get('noise_assumption')!='independent empirical additive residuals':raise ValueError('Different declared noise model')
    if len(model['equations'])!=n or len(model['noise_samples'])!=n:raise ValueError('Wrong SCM dimensions')
    do=tuple(map(Q,do_bounds))
    if len(do)!=2 or do[0]>do[1]:raise ValueError('Invalid intervention domain')
    bounds=[None]*n;noise=[];steps=[]
    for j in topo:
        values=model['noise_samples'][j]
        if not isinstance(values,list) or not values or len(values)>1000000:raise ValueError('Finite nonempty empirical noise required')
        values=list(map(rational,values));u=(min(values),max(values));noise.append({'node':j,'count':len(values),'range':list(map(str,u))})
        allowed={i for i in range(n) if graph[i][j]}
        f=interval(model['equations'][j],bounds,allowed)
        natural=(f[0]+u[0],f[1]+u[1]);out=(min(natural[0],do[0]),max(natural[1],do[1]));bounds[j]=out
        steps.append({'node':j,'mechanism_range':list(map(str,f)),'natural_range':list(map(str,natural)),'all_do_union_range':list(map(str,out))})
    effective=[[int(i in parents(model['equations'][j])) for j in range(n)] for i in range(n)]
    if 'effective_graph' in model and effective!=model['effective_graph']:raise ValueError('Serialized effective parents mismatch')
    absolute=[max(map(abs,b)) for b in bounds]
    return {'schema':'ncd.empirical-scm-support.v1','status':'proved-scoped','model_sha256':fingerprint(model),
        'program_sha256':fingerprint(model['equations']),'graph_sha256':fingerprint(graph),'noise_values_sha256':fingerprint(model['noise_samples']),
        'nodes':n,'topological_order':topo,'noise_ranges':noise,'steps':steps,'coordinate_abs_upper':list(map(str,absolute)),
        'joint_first_moment_upper':str(sum(absolute)),'intervention_bounds':list(map(str,do)),
        'scope':'Every compatible intervention subset, each assigned any real value in the declared closed interval; all empirical noise choices',
        'semantic_scope':'Ideal real CDIR operations; constants are exact stored binary values; hardware rounding is not certified',
        'assumptions':['finite nonempty empirical noise arrays','valid acyclic declared parents','registered arithmetic operator subset'],
        'not_proved':['true graph recovery','empirical residual independence in the true world','true noise-law distance','neural/program fidelity','population distribution recovery'],
        'original_claim_closed':False,'original_objective_achieved':False}


def verify(model,certificate):
    actual=derive(model,certificate['intervention_bounds'])
    if actual!=certificate:raise ValueError('Finite-support certificate mismatch')
    return {'status':'verified','conclusion':'proved-scoped','nodes':len(model['source_graph']),'original_objective_achieved':False}
