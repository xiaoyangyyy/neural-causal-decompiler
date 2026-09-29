"""Neural-only affine-cell synthesis with mean-value fidelity certificates.

Search is heuristic; completeness and shortest-program claims are not made.
Every accepted cell includes its boundary. Device arithmetic is not certified.
"""
from fractions import Fraction as Q
import time,json
from pathlib import Path
from ncd.proof_intervals import Interval,hyperbolic_tangent,affine
from ncd.frozen_mechanism_proof import neural_interval,expression_interval
from ncd.cdir import Node


def neural_gradient(network,box):
    n=len(box);parents=network['parents']
    # The existing validator checks dimensions, scales, parent indices and layers.
    neural_interval(network,[Interval.point((v.lo+v.hi)/2) for v in box])
    inputs=[box[i] for i in parents] if parents else [Interval.point(0)]
    h=[(v-Q(m))/Q(s) for v,m,s in zip(inputs,network['mean'],network['std'])]
    g=[[Interval.point(Q(int(bool(parents) and parents[k]==j))/Q(network['std'][k])) for j in range(n)] for k in range(len(h))]
    for layer in network['layers']:
        weights=layer['weights'];hh=affine(weights,layer['bias'],h)
        gg=[[sum((Q(w)*g[k][j] for k,w in enumerate(row)),Interval.point(0)) for j in range(n)] for row in weights]
        if layer['activation']=='tanh':
            hh=[hyperbolic_tangent(v) for v in hh]
            derivative=[Interval(max(Q(0),1-v.square().hi),min(Q(1),1-v.square().lo)) for v in hh]
            gg=[[factor*v for v in row] for factor,row in zip(derivative,gg)]
        elif layer['activation']!='identity':raise ValueError('Unsupported derivative activation')
        h,g=hh,gg
    if len(h)!=1:raise ValueError('Scalar mechanism required')
    return h[0]*Q(network['output_scale'])+Q(network['output_mean']),[v*Q(network['output_scale']) for v in g[0]]


def affine_candidate(network,box):
    center=[Interval.point((v.lo+v.hi)/2) for v in box]
    y,grad=neural_gradient(network,center)
    coefficients=[Q(float((v.lo+v.hi)/2)) for v in grad]
    intercept=Q(float((y.lo+y.hi)/2-sum((a*v.lo for a,v in zip(coefficients,center)),Q(0))))
    expression={'op':'constant','value':float(intercept)}
    for i,a in enumerate(coefficients):
        if a:
            expression={'op':'add','args':[expression,{'op':'mul','args':[{'op':'constant','value':float(a)},{'op':'var','index':i}]}]}
    Node.from_dict(expression)
    return {'expression':expression,'coefficients':[str(v) for v in coefficients],'intercept':str(intercept)}


def cell_bound(network,candidate,box,scale):
    # Confirm the declared affine slope is exactly the serialized CDIR program.
    if len(candidate['coefficients'])!=len(box):raise ValueError('Affine gradient dimension mismatch')
    expression={'op':'constant','value':float(Q(candidate['intercept']))}
    for i,a in enumerate(candidate['coefficients']):
        if Q(a):expression={'op':'add','args':[expression,{'op':'mul','args':[{'op':'constant','value':float(Q(a))},{'op':'var','index':i}]}]}
    if expression!=candidate['expression'] or any(Q(float(Q(v)))!=Q(v) for v in candidate['coefficients']+[candidate['intercept']]):raise ValueError('Affine expression/slopes mismatch')
    center=[Interval.point((v.lo+v.hi)/2) for v in box]
    center_error=neural_interval(network,center)-expression_interval(expression,center)
    _,derivative=neural_gradient(network,box)
    error=center_error
    for interval,slope,value in zip(derivative,candidate['coefficients'],box):
        radius=(value.hi-value.lo)/2
        error=error+(interval-Q(slope))*Interval(-radius,radius)
    return error/Q(scale),derivative


def certify_piecewise(network,domain,epsilon='1/100',max_cells=512,seconds=300,checkpoint=None,checkpoint_path=None):
    if type(max_cells)is not int or max_cells<1 or seconds<=0 or Q(epsilon)<0:raise ValueError('Proof budget')
    box=[Interval.from_dict(v) for v in domain];scale=Q(network['output_scale'])
    if scale<=0 or not box:raise ValueError('Mechanism domain/scale')
    if checkpoint is None:
        nodes=[{'box':domain}]
    else:
        if checkpoint['network']!=network or checkpoint['domain']!=domain or checkpoint['epsilon']!=str(Q(epsilon)):raise ValueError('Changed frozen resume contract')
        verify_piecewise(checkpoint)
        nodes=[dict(row) for row in checkpoint['nodes']]
    def pack():
        unresolved=[i for i,row in enumerate(nodes) if row.get('kind','pending')=='pending']
        return {'schema':'ncd.piecewise-affine-mechanism.v1','status':'unresolved' if unresolved else 'proved',
        'network':network,'domain':domain,'epsilon':str(Q(epsilon)),'normalization_scale':str(scale),
        'nodes':nodes,'unresolved_cells':unresolved,'search_cells':len(nodes),
        'scope':'this frozen mathematical neural mechanism on the entire declared closed box',
        'parents_are_truth_correct':'not claimed','mechanism_truth_fidelity':'not claimed',
        'hardware_rounding_covered':False,'mdl_minimality':'unresolved; heuristic upper bound only'}
    def persist(value):
        if checkpoint_path is not None:
            path=Path(checkpoint_path);path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
    deadline=time.monotonic()+seconds
    pending=[i for i,row in enumerate(nodes) if row.get('kind','pending')=='pending']
    while pending and time.monotonic()<deadline:
        i=pending.pop();row=nodes[i];cell=[Interval.from_dict(v) for v in row['box']]
        candidate=affine_candidate(network,cell);bound,gradient=cell_bound(network,candidate,cell,scale)
        if max(abs(bound.lo),abs(bound.hi))<=Q(epsilon):
            row.update(kind='leaf',candidate=candidate,normalized_error=bound.to_dict());persist(pack());continue
        axes=[j for j in network['parents'] if cell[j].lo<cell[j].hi]
        if not axes or len(nodes)+2>max_cells:continue
        axis=max(axes,key=lambda j:(gradient[j].hi-gradient[j].lo)*(cell[j].hi-cell[j].lo))
        middle=Q(float((cell[axis].lo+cell[axis].hi)/2))
        if not cell[axis].lo<middle<cell[axis].hi:continue
        left=cell[:];right=cell[:];left[axis]=Interval(cell[axis].lo,middle);right[axis]=Interval(middle,cell[axis].hi)
        li=len(nodes);ri=li+1
        nodes.extend([{'box':[v.to_dict() for v in child]} for child in (left,right)])
        row.update(kind='split',axis=axis,threshold=str(middle),left=li,right=ri)
        pending.extend([ri,li]);persist(pack())
    result=pack()
    persist(result)
    return result


def verify_piecewise(certificate):
    if certificate['schema']!='ncd.piecewise-affine-mechanism.v1':raise ValueError('Piecewise proof schema')
    network=certificate['network'];nodes=certificate['nodes'];scale=Q(certificate['normalization_scale']);epsilon=Q(certificate['epsilon'])
    if scale!=Q(network['output_scale']) or not nodes or nodes[0]['box']!=certificate['domain'] or epsilon<0:raise ValueError('Piecewise proof contract')
    seen=set();leaves=[];pending=[]
    def visit(index,expected):
        if type(index)is not int or not 0<=index<len(nodes) or index in seen:raise ValueError('Cell tree cycle/sharing/index')
        seen.add(index);row=nodes[index]
        if row['box']!=expected:raise ValueError('Incomplete or changed child domain')
        cell=[Interval.from_dict(v) for v in expected];kind=row.get('kind','pending')
        if kind=='leaf':
            error,_=cell_bound(network,row['candidate'],cell,scale)
            if error.to_dict()!=row['normalized_error'] or max(abs(error.lo),abs(error.hi))>epsilon:raise ValueError('Unverified affine leaf')
            leaves.append(index)
        elif kind=='split':
            axis=row['axis'];middle=Q(row['threshold'])
            if type(axis)is not int or not 0<=axis<len(cell) or Q(float(middle))!=middle or not cell[axis].lo<middle<cell[axis].hi:raise ValueError('Invalid split boundary')
            left=cell[:];right=cell[:];left[axis]=Interval(cell[axis].lo,middle);right[axis]=Interval(middle,cell[axis].hi)
            visit(row['left'],[v.to_dict() for v in left]);visit(row['right'],[v.to_dict() for v in right])
        elif kind=='pending':pending.append(index)
        else:raise ValueError('Unknown cell kind')
    visit(0,certificate['domain'])
    if len(seen)!=len(nodes) or sorted(pending)!=certificate['unresolved_cells'] or certificate['search_cells']!=len(nodes):raise ValueError('Orphan or concealed proof cells')
    status='unresolved' if pending else 'proved'
    if certificate['status']!=status:raise ValueError('Invalid proof conclusion')
    return {'status':'verified','conclusion':status,'proved_leaves':len(leaves),'unresolved_cells':len(pending),
        'all_boundaries_included':True,'original_requirement_closed':False}


def program(certificate):
    if verify_piecewise(certificate)['conclusion']!='proved':raise ValueError('Incomplete proof has no certified total program')
    nodes=certificate['nodes']
    def emit(index):
        row=nodes[index]
        if row['kind']=='leaf':return {'expression':row['candidate']['expression']}
        return {'input_index':row['axis'],'threshold':float(Q(row['threshold'])),'left':emit(row['left']),'right':emit(row['right'])}
    # CDIR comparisons are scalar statistics comparisons. Mechanism routing
    # explicitly reads one scalar input per row; arithmetic leaves reuse CDIR.
    return {'schema':'ncd.piecewise-cdir-mechanism-program.v1','domain':certificate['domain'],'tree':emit(0)}


def execute_program(spec,data,return_trace=False):
    import numpy as np
    if spec['schema']!='ncd.piecewise-cdir-mechanism-program.v1':raise ValueError('Program schema')
    values=np.asarray(data,dtype=float);domain=[Interval.from_dict(v) for v in spec['domain']]
    if values.ndim!=2 or values.shape[1]!=len(domain) or not np.isfinite(values).all():raise ValueError('Program inputs')
    if any(not v.lo<=Q(float(x))<=v.hi for row in values for x,v in zip(row,domain)):raise ValueError('Inputs outside declared program domain')
    output=np.empty(len(values));trace=[]
    def validate(node,depth=0):
        if depth>24:raise ValueError('Program control nesting')
        if set(node)=={'expression'}:Node.from_dict(node['expression']);return
        if set(node)!={'input_index','threshold','left','right'} or type(node['input_index'])is not int or not 0<=node['input_index']<len(domain) or not np.isfinite(node['threshold']):raise ValueError('Program branch')
        validate(node['left'],depth+1);validate(node['right'],depth+1)
    validate(spec['tree'])
    def run(node,indices,path):
        if not len(indices):return
        if 'expression' in node:
            output[indices]=Node.from_dict(node['expression']).evaluate(values[indices])
            trace.append({'path':path,'executed_rows':indices.tolist()});return
        mask=values[indices,node['input_index']]<node['threshold']
        run(node['left'],indices[mask],path+['left']);run(node['right'],indices[~mask],path+['right'])
    run(spec['tree'],np.arange(len(values)),[])
    return (output,trace) if return_trace else output
