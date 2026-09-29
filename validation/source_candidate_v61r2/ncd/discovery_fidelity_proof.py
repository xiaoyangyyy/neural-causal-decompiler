"""Local full-input proof for historical Discoverer and historical Rule.

No replacement teacher or inserted discovery rule is used. Unsupported regression
features remain unresolved rather than being evaluated by an uncertified oracle.
"""
from fractions import Fraction as Q
from .proof_intervals import Interval,affine,hyperbolic_tangent,logarithm,exponential
from .frozen_mechanism_proof import expression_interval
from .io import digest


def export_discoverer(checkpoint):
    from .model import load_model
    model=load_model(checkpoint);state=model.state_dict()
    def numbers(t):
        def walk(x):return [walk(v) for v in x] if isinstance(x,list) else str(Q(x))
        return walk(t.detach().cpu().tolist())
    stacks={}
    for name in ('encoder','head'):
        indices=(0,2) if name=='encoder' else (0,2)
        stacks[name]=[{'weights':numbers(state[f'{name}.{i}.weight']),'bias':numbers(state[f'{name}.{i}.bias']),
                      'activation':'tanh' if name=='encoder' or i==0 else 'identity'} for i in indices]
    return {'schema':'ncd.frozen-discoverer.v1','checkpoint_sha256':digest(checkpoint),'width':model.width,**stacks}


def average(values):
    return sum(values,Interval.point(0))/len(values)


def variance(values):
    mean=average(values)
    return average([(v-mean).square() for v in values])


def stack(layers,values):
    h=values
    for layer in layers:
        h=affine(layer['weights'],layer['bias'],h)
        if layer['activation']=='tanh':h=[hyperbolic_tangent(x) for x in h]
        elif layer['activation']!='identity':raise ValueError('Unsupported discoverer activation')
    return h


def discoverer_logits(network,data):
    if network['schema']!='ncd.frozen-discoverer.v1' or len(data)<16 or any(len(r)!=2 for r in data):raise ValueError('Invalid raw discovery domain')
    def representation(rows):
        cols=list(zip(*rows));means=[average(c) for c in cols]
        std=[]
        for c in cols:
            v=variance(c).sqrt();floor=Q(1e-5)
            std.append(Interval(max(floor,v.lo),max(floor,v.hi)))
        hidden=[stack(network['encoder'],[((x-m)/s).clip(-20,20) for x,m,s in zip(row,means,std)]) for row in rows]
        return [average([h[j] for h in hidden]) for j in range(network['width'])]+[
            average([h[j].square() for h in hidden]) for j in range(network['width'])]+[logarithm(s) for s in std]
    normal=stack(network['head'],representation(data))
    swapped=stack(network['head'],representation([list(reversed(row)) for row in data]))
    if len(normal)!=4 or len(swapped)!=4:raise ValueError('Expected four discovery logits')
    return [(normal[j]+swapped[[1,0,2,3][j]])/2 for j in range(4)]


def order_statistic(values):
    if not values:raise ValueError('Empty order statistic')
    lower=sorted(v.lo for v in values);upper=sorted(v.hi for v in values);n=len(values)
    if n%2:return Interval(lower[n//2],upper[n//2])
    return Interval((lower[n//2-1]+lower[n//2])/2,(upper[n//2-1]+upper[n//2])/2)


def dependence_interval(x,y):
    def kernel(v):
        n=len(v);dist=[[(a-b).square() for b in v] for a in v]
        positive=[]
        for i in range(n):
            for j in range(n):
                # Equal source variables have exact zero self distance.
                if i==j:dist[i][j]=Interval.point(0);continue
                d=dist[i][j]
                if d.lo>Q(1e-12):positive.append(d)
                elif d.hi>Q(1e-12):raise ValueError('Kernel bandwidth threshold remains unresolved')
        bandwidth=order_statistic(positive) if positive else Interval.point(1)
        floor=Q(1e-8);bandwidth=Interval(max(floor,bandwidth.lo),max(floor,bandwidth.hi))
        raw=[[exponential(-d/(2*bandwidth)) for d in row] for row in dist]
        rows=[average(row) for row in raw];cols=[average(list(col)) for col in zip(*raw)];mean=average(rows)
        return [[raw[i][j]-rows[i]-cols[j]+mean for j in range(n)] for i in range(n)]
    k,l=kernel(x),kernel(y)
    numerator=sum((a*b for row,other in zip(k,l) for a,b in zip(row,other)),Interval.point(0))
    normk=sum((a.square() for row in k for a in row),Interval.point(0))
    norml=sum((a.square() for row in l for a in row),Interval.point(0))
    norm=(normk*norml).sqrt();floor=Q(1e-12)
    return numerator/Interval(max(floor,norm.lo),max(floor,norm.hi))



def interval_solve(matrix,rhs):
    """Gaussian elimination with certified nonzero interval pivots."""
    n=len(rhs)
    if len(matrix)!=n or any(len(row)!=n for row in matrix):raise ValueError('Interval solve dimensions')
    a=[list(row)+[value] for row,value in zip(matrix,rhs)]
    for j in range(n):
        candidates=[i for i in range(j,n) if a[i][j].hi<0 or a[i][j].lo>0]
        if not candidates:raise ValueError('Regression pivot cannot be certified nonzero')
        pivot=max(candidates,key=lambda i:min(abs(a[i][j].lo),abs(a[i][j].hi)))
        a[j],a[pivot]=a[pivot],a[j]
        for i in range(j+1,n):
            factor=a[i][j]/a[j][j]
            for k in range(j+1,n+1):a[i][k]=a[i][k]-factor*a[j][k]
            # For each actual system this eliminated entry is exactly zero.
            a[i][j]=Interval.point(0)
    out=[Interval.point(0)]*n
    for j in reversed(range(n)):
        out[j]=(a[j][n]-sum((a[j][k]*out[k] for k in range(j+1,n)),Interval.point(0)))/a[j][j]
    return out


def crossfit_intervals(y,predictors):
    from .proof_intervals import trigonometric
    x=[list(row) for row in predictors];n=len(y)
    if n<16 or len(x)!=n or not x[0] or any(len(row)!=len(x[0]) for row in x):raise ValueError('Cross-fit needs N>=16 predictors')
    order=sorted(range(n),key=lambda i:tuple((v.lo+v.hi)/2 for v in x[i]))
    for a,b in zip(order,order[1:]):
        certified=False
        for va,vb in zip(x[a],x[b]):
            if va.hi<vb.lo:certified=True;break
            if va.lo==va.hi==vb.lo==vb.hi:continue
            raise ValueError('Cross-fit sorting boundary remains unresolved')
        if not certified and any(va.lo!=va.hi or vb.lo!=vb.hi for va,vb in zip(x[a],x[b])):
            raise ValueError('Cross-fit tie boundary remains unresolved')
    result=[None]*n;p=len(x[0])
    def basis(z):
        return [Interval.point(1)]+z+[v.square() for v in z]+[trigonometric(v) for v in z]+[
            hyperbolic_tangent(v) for v in z]+[z[i]*z[j] for i in range(p) for j in range(i)]
    for side in (0,1):
        hold,fit=order[side::2],order[1-side::2]
        means=[average([x[i][j] for i in fit]) for j in range(p)]
        sd=[variance([x[i][j] for i in fit]).sqrt() for j in range(p)]
        sd=[Interval(max(Q(1e-6),v.lo),max(Q(1e-6),v.hi)) for v in sd]
        rows={i:basis([((v-m)/sigma).clip(-10,10) for v,m,sigma in zip(x[i],means,sd)]) for i in order}
        width=len(rows[fit[0]])
        matrix=[[sum((rows[i][j]*rows[i][k] for i in fit),Interval.point(0))+
                 (Q(1e-8) if j==k==0 else Q(.01) if j==k else 0) for k in range(width)] for j in range(width)]
        rhs=[sum((rows[i][j]*y[i] for i in fit),Interval.point(0)) for j in range(width)]
        coefficients=interval_solve(matrix,rhs)
        for i in hold:result[i]=sum((a*b for a,b in zip(rows[i],coefficients)),Interval.point(0))
    return result

def raw_feature(data,index):
    # Historical Rule inputs come from statistics.extract_one. They do not use
    # CDIR crossfit_prediction or its different basis, floors and penalties.
    from .actual_statistics_intervals import raw_feature as actual_feature
    return actual_feature(data,index)


def rule_labels(program,data):
    from .statistics import FEATURES
    from .rules import Rule
    Rule.from_dict(program)
    if tuple(program['names'])!=FEATURES:raise ValueError('Rule is not bound to actual statistics.extract_one feature schema')
    cache={}
    def expression(node):
        if node['op']=='var':
            i=node['index']
            if i not in cache:cache[i]=raw_feature(data,i)
            return cache[i]
        if node['op']=='constant':return Interval.point(node['value'])
        # Replace feature variables by certified interval inputs in a compact AST.
        args=[expression(a) for a in node.get('args',[])]
        rewritten={'op':node['op'],'args':[{'op':'var','index':i} for i in range(len(args))]}
        return expression_interval(rewritten,args)
    def visit(t):
        if 'label' in t:return {t['label']}
        v=expression(t['expr']);threshold=Q(t['threshold'])
        if v.hi<threshold:return visit(t['left'])
        if v.lo>=threshold:return visit(t['right'])
        return visit(t['left'])|visit(t['right'])
    return visit(program['tree'])


def certify_discovery_box(network,program,domain):
    try:
        data=[[Interval.from_dict(x) for x in row] for row in domain]
        labels=rule_labels(program,data)
        logits=discoverer_logits(network,data)
        winners=[j for j in range(4) if all(logits[j].lo>logits[k].hi for k in range(4) if k!=j)]
        status='proved' if len(winners)==1 and labels=={winners[0]} else 'unresolved'
        # Different single labels over a nonempty whole box strictly refute agreement.
        if len(winners)==1 and len(labels)==1 and labels!={winners[0]}:status='refuted'
        return {'schema':'ncd.discovery-box-fidelity.v2','status':status,'domain':domain,
            'logit_enclosures':[x.to_dict() for x in logits],'rule_labels':sorted(labels),'neural_winners':winners,
            'scope':'this historical discovery checkpoint and historical actual-statistics Rule on the complete declared dataset box',
            'semantics':'exact saved coefficients and mathematical operators; additional inference rounding excluded',
            'does_not_close_full_discovery_domain':True}
    except ValueError as exc:
        return {'schema':'ncd.discovery-box-fidelity.v2','status':'unresolved','domain':domain,'reason':str(exc),
            'does_not_close_full_discovery_domain':True}


def verify_discovery_box(network,program,certificate):
    expected=certify_discovery_box(network,program,certificate['domain'])
    mathematical={k:v for k,v in certificate.items() if k not in ('network_sha256','program_sha256','source_program_sha256')}
    if expected!=mathematical:raise ValueError('Discovery proof mismatch')
    return {'status':'verified','conclusion':certificate['status'],'does_not_close_original_requirement':True}
