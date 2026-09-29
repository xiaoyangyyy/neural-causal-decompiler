"""Proof-carrying comparison of an actual frozen Tanh mechanism and CDIR AST.

The original checkpoint is never modified. The portable export is a mathematical
function with exact serialized coefficients, not a device rounding certificate.
"""
from fractions import Fraction as Q
from pathlib import Path
import time
from .io import digest,read_json,save_json
from .proof_intervals import (Interval,affine,hyperbolic_tangent,logarithm,
    trigonometric,protected_division)


def export_mechanism(checkpoint):
    from .mechanisms import load_mechanism
    model=load_mechanism(checkpoint)
    state=model.state_dict()
    def numbers(tensor):
        def walk(value):
            if isinstance(value,list):return [walk(x) for x in value]
            return str(Q(value))
        return walk(tensor.detach().cpu().tolist())
    layers=[]
    for index in (0,2,4):
        layers.append({'weights':numbers(state[f'network.{index}.weight']),
                       'bias':numbers(state[f'network.{index}.bias']),
                       'activation':'tanh' if index!=4 else 'identity'})
    return {'schema':'ncd.frozen-tanh-mechanism.v1','checkpoint_sha256':digest(checkpoint),
        'parents':list(model.parents),'mean':numbers(model.mean),'std':numbers(model.std),
        'output_mean':numbers(model.ymean),'output_scale':numbers(model.ystd),'layers':layers}


def neural_interval(network,box):
    if network.get('schema')!='ncd.frozen-tanh-mechanism.v1':raise ValueError('Unsupported network export')
    parents=network['parents']
    if len(set(parents))!=len(parents) or any(type(i)is not int or not 0<=i<len(box) for i in parents):
        raise ValueError('Invalid frozen parent support')
    inputs=[box[i] for i in parents] if parents else [Interval.point(0)]
    if len(inputs)!=len(network['mean']) or len(inputs)!=len(network['std']):raise ValueError('Normalization dimensions')
    if any(Q(s)<=0 for s in network['std']) or Q(network['output_scale'])<=0:raise ValueError('Invalid normalization scale')
    h=[(x-Q(m))/Q(s) for x,m,s in zip(inputs,network['mean'],network['std'])]
    for layer in network['layers']:
        h=affine(layer['weights'],layer['bias'],h)
        if layer['activation']=='tanh':h=[hyperbolic_tangent(x) for x in h]
        elif layer['activation']!='identity':raise ValueError('Unsupported activation')
    if len(h)!=1:raise ValueError('Expected scalar mechanism')
    return h[0]*Q(network['output_scale'])+Q(network['output_mean'])


def expression_interval(node,box):
    op=node['op'];args=node.get('args',[])
    if op=='var':return box[node['index']]
    if op=='constant':return Interval.point(node['value'])
    a=[expression_interval(x,box) for x in args]
    if op=='add':return a[0]+a[1]
    if op=='sub':return a[0]-a[1]
    if op=='mul':return a[0]*a[1]
    if op=='div':return protected_division(a[0],a[1],str(Q(1e-8)))
    if op=='square':return a[0].square()
    if op=='abs':return a[0].abs()
    if op=='tanh':return hyperbolic_tangent(a[0])
    if op=='sqrt':return Interval(max(Q(0),a[0].lo),max(Q(0),a[0].hi)).sqrt()
    if op=='log':
        v=a[0].abs();floor=Q(1e-12)
        return logarithm(Interval(max(floor,v.lo),max(floor,v.hi)))
    if op in ('sin','cos'):return trigonometric(a[0],op=='cos')
    raise ValueError('Unsupported CDIR proof operator: '+op)


def _validate(network,program,domain,epsilon,scale):
    from .cdir import Node
    Node.from_dict(program)
    box=[Interval.from_dict(x) for x in domain]
    if not box or Q(epsilon)<0 or Q(scale)<=0:raise ValueError('Invalid fidelity contract')
    neural_interval(network,[Interval.point((x.lo+x.hi)/2) for x in box])
    return box


def _error(network,program,box,scale):
    return (neural_interval(network,box)-expression_interval(program,box))/Q(scale)


def certify_mechanism(network,program,domain,epsilon='1/100',scale=None,max_boxes=256,seconds=43200):
    scale=network['output_scale'] if scale is None else str(Q(scale))
    box=_validate(network,program,domain,epsilon,scale)
    if type(max_boxes)is not int or max_boxes<1 or seconds<=0:raise ValueError('Invalid proof budget')
    deadline=time.monotonic()+seconds
    tree=[];pending=[(None,None,box)];witness=None
    while pending:
        parent,side,current=pending.pop()
        index=len(tree)
        node={'box':[x.to_dict() for x in current]}
        tree.append(node)
        if parent is not None:tree[parent][side]=index
        if len(tree)>max_boxes or time.monotonic()>deadline:
            node.update(kind='unresolved',reason='budget exhausted');continue
        try:
            error=_error(network,program,current,scale)
            if error.abs().hi<=Q(epsilon):
                node.update(kind='proved',error=error.to_dict());continue
            middle=[Interval.point((x.lo+x.hi)/2) for x in current]
            actual=_error(network,program,middle,scale)
            if actual.lo>Q(epsilon) or actual.hi<-Q(epsilon):
                witness={'point':[str(x.lo) for x in middle],'error':actual.to_dict()}
                node.update(kind='counterexample',error=actual.to_dict())
                # A verified existential witness suffices. Preserve the rest as unresolved.
                for p,s,b in pending:
                    idx=len(tree);tree.append({'box':[x.to_dict() for x in b],'kind':'unresolved','reason':'counterexample found'})
                    if p is not None:tree[p][s]=idx
                pending=[];break
            axis=max(range(len(current)),key=lambda i:current[i].hi-current[i].lo)
            if current[axis].lo==current[axis].hi:
                node.update(kind='unresolved',reason='precision insufficient at point');continue
            cut=(current[axis].lo+current[axis].hi)/2
            left=list(current);right=list(current)
            left[axis]=Interval(current[axis].lo,cut);right[axis]=Interval(cut,current[axis].hi)
            node.update(kind='split',axis=axis,cut=str(cut))
            pending.extend([(index,'right',right),(index,'left',left)])
        except ValueError as exc:node.update(kind='unresolved',reason=str(exc))
    status='refuted' if witness else 'proved' if all(n['kind']!='unresolved' for n in tree) else 'unresolved'
    return {'schema':'ncd.frozen-mechanism-fidelity.v1','status':status,'epsilon':str(Q(epsilon)),
        'normalizer':str(Q(scale)),'domain':domain,'tree':tree,'counterexample':witness,
        'scope':'this frozen scalar neural mechanism versus this candidate expression on this closed input box',
        'semantics':'exact serialized neural coefficients and mathematical transcendental functions; excludes additional inference rounding',
        'global_synthesis_or_true_SCM_recovery':False}


def verify_mechanism(network,program,certificate):
    if certificate.get('schema')!='ncd.frozen-mechanism-fidelity.v1':raise ValueError('Unsupported fidelity certificate')
    if certificate.get('global_synthesis_or_true_SCM_recovery') is not False:raise ValueError('Unsupported global recovery claim')
    box=_validate(network,program,certificate['domain'],certificate['epsilon'],certificate['normalizer'])
    tree=certificate['tree'];visited=set();unresolved=False;found=[]
    def visit(index,expected):
        nonlocal unresolved
        if type(index)is not int or not 0<=index<len(tree) or index in visited:raise ValueError('Invalid proof tree reference')
        visited.add(index);node=tree[index]
        if node['box']!=[x.to_dict() for x in expected]:raise ValueError('Incomplete or changed domain cover')
        kind=node['kind']
        if kind=='split':
            axis=node['axis'];cut=Q(node['cut'])
            if type(axis)is not int or not 0<=axis<len(expected) or not expected[axis].lo<cut<expected[axis].hi:raise ValueError('Invalid domain split')
            left=list(expected);right=list(expected)
            left[axis]=Interval(expected[axis].lo,cut);right[axis]=Interval(cut,expected[axis].hi)
            visit(node['left'],left);visit(node['right'],right)
        elif kind=='proved':
            error=_error(network,program,expected,certificate['normalizer'])
            if node['error']!=error.to_dict() or error.abs().hi>Q(certificate['epsilon']):raise ValueError('False interval proof')
        elif kind=='counterexample':
            point=[Interval.point((x.lo+x.hi)/2) for x in expected]
            error=_error(network,program,point,certificate['normalizer'])
            if node['error']!=error.to_dict() or not (error.lo>Q(certificate['epsilon']) or error.hi<-Q(certificate['epsilon'])):raise ValueError('Invalid strict counterexample')
            found.append({'point':[str(x.lo) for x in point],'error':error.to_dict()})
        elif kind=='unresolved':unresolved=True
        else:raise ValueError('Unknown leaf type')
    if not tree:raise ValueError('Empty proof cover')
    visit(0,box)
    if visited!=set(range(len(tree))):raise ValueError('Unreachable proof tree rows')
    status='refuted' if found else 'unresolved' if unresolved else 'proved'
    if certificate['status']!=status or certificate['counterexample']!=(found[0] if found else None):raise ValueError('Fidelity conclusion mismatch')
    return {'status':'verified','conclusion':status,'boxes_checked':len(visited),
        'scope':certificate['scope'],'does_not_close_original_requirement':True}
