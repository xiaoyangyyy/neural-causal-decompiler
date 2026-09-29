"""Exact shared-control initial handoff for mixed state/control ReLU units."""
from fractions import Fraction as Q
from itertools import product
from math import prod
from .continuous_separation import _digest
from .continuous_compositional_realization import verify_weighted
from .phase_initial_handoff import _observation,_scalar_pieces,_fingerprint
from .nonlinear_action_closure import SQUARE
from .rational_polytope import box,clip,evaluate
from .reachable_two_stage import _reachable_indices


def _extract(system,coordinate):
    net=system.transition
    if system.action_dim!=2 or len(net.weights)!=2:
        raise ValueError('Two controls and one hidden ReLU layer required')
    relevant=tuple(j for j,w in enumerate(net.weights[1][coordinate]) if w)
    axes=tuple(sorted({k for j in relevant for k,w in enumerate(net.weights[0][j][:-2]) if w}))
    if len(axes)>2:
        raise ValueError('At most two state support axes per scalar output required')
    padded=axes+(None,)*(2-len(axes))
    terms=[]
    mixed=0
    for j in relevant:
        raw=net.weights[0][j]
        form=tuple(Q(raw[k]) if k is not None else Q(0) for k in padded)
        form+=tuple(map(Q,raw[-2:]))+(Q(net.biases[0][j]),)
        terms.append((Q(net.weights[1][coordinate][j]),form))
        mixed+=int(any(raw[:-2]) and any(raw[-2:]))
    return axes,padded,tuple(terms),mixed


def _difference_pieces(terms,low,high,center,rank_cache=None,max_regions=4096):
    combined={}
    for weight,form in terms:
        centered=(Q(0),Q(0),form[2],form[3],form[4]+form[0]*center[0]+form[1]*center[1])
        combined[form]=combined.get(form,Q(0))+weight
        combined[centered]=combined.get(centered,Q(0))-weight
    signed=tuple(sorted((form,w) for form,w in combined.items() if w))
    root=box(low,high)
    crossing=tuple(form for form,w in signed if
        min(evaluate(form,p) for p in root.vertices)<0<max(evaluate(form,p) for p in root.vertices))
    polygons=[root]
    cache={} if rank_cache is None else rank_cache
    splits=0
    for form in crossing:
        children=[]
        for poly in polygons:
            scores=tuple(evaluate(form,p) for p in poly.vertices)
            if min(scores)>=0 or max(scores)<=0:
                children.append(poly)
            else:
                first=clip(poly,form,cache)
                second=clip(poly,tuple(-x for x in form),cache)
                if not first.vertices or not second.vertices:
                    raise ValueError('Joint phase split lost a nonempty child')
                children.extend((first,second))
                splits+=1
        if len(children)>max_regions:
            raise ValueError('Joint phase region budget exceeded; unresolved')
        polygons=children
    pieces=[]
    for poly in polygons:
        midpoint=tuple(sum((p[k] for p in poly.vertices),Q(0))/len(poly.vertices) for k in range(4))
        affine=[Q(0)]*5
        for form,weight in signed:
            scores=tuple(evaluate(form,p) for p in poly.vertices)
            if evaluate(form,midpoint)>=0:
                if min(scores)<0:
                    raise ValueError('Positive joint phase not fixed on polytope')
                for k,x in enumerate(form):
                    affine[k]+=weight*x
            elif max(scores)>0:
                raise ValueError('Negative joint phase not fixed on polytope')
        pieces.append((poly,tuple(affine)))
    values=[(evaluate(form,p),p) for poly,form in pieces for p in poly.vertices]
    minimum,maximum=min(values),max(values)
    if not minimum[0]<=0<=maximum[0]:
        raise ValueError('Shared-control difference misses its zero at the state midpoint')
    return tuple(pieces),minimum,maximum,splits


def _cell(terms,indices,pair_bins,rank_cache,max_regions):
    low=tuple(Q(q,n) for q,n in zip(indices,pair_bins))+(Q(0),Q(0))
    high=tuple(Q(q+1,n) for q,n in zip(indices,pair_bins))+(Q(1),Q(1))
    center=tuple(Q(2*q+1,2*n) for q,n in zip(indices,pair_bins))
    pieces,minimum,maximum,splits=_difference_pieces(terms,low,high,center,rank_cache,max_regions)
    action_terms=tuple((w,(f[2],f[3],f[4]+f[0]*center[0]+f[1]*center[1])) for w,f in terms)
    action_pieces=_scalar_pieces(action_terms,SQUARE,Q(0),{})
    lipschitz=max(abs(a)+abs(b) for poly,(a,b,c) in action_pieces)
    fingerprint=_digest([{'vertices':[[str(x) for x in p] for p in poly.vertices],
        'affine_difference':[str(x) for x in form]} for poly,form in pieces])
    return {'indices':list(indices),'minimum_difference':str(minimum[0]),
        'maximum_difference':str(maximum[0]),'minimum_witness':[str(x) for x in minimum[1]],
        'maximum_witness':[str(x) for x in maximum[1]],
        'difference_error_upper':str(max(abs(minimum[0]),abs(maximum[0]))),
        'joint_phase_count':len(pieces),'joint_splits':splits,
        'joint_vertices_checked':sum(len(poly.vertices) for poly,form in pieces),
        'joint_partition_sha256':fingerprint,'center_action_lipschitz':str(lipschitz),
        'center_action_phase_count':len(action_pieces),
        'center_action_partition_sha256':_fingerprint(action_pieces)}


def _proof(system,recurrent,max_regions=4096):
    if not isinstance(max_regions,int) or isinstance(max_regions,bool) or max_regions<1:
        raise ValueError('Positive integer phase region budget required')
    _observation(system)
    if Q(recurrent['epsilon'])!=Q(17,100) or verify_weighted(system,recurrent)['status']!='certified':
        raise ValueError('Certified recurrent relation at epsilon=17/100 required')
    bins=tuple(3 if i<4 else 1 for i in range(system.state_dim))
    rows=[]
    for coordinate in range(system.state_dim):
        axes,padded,terms,mixed=_extract(system,coordinate)
        pair_bins=tuple(bins[k] if k is not None else 1 for k in padded)
        cache={}
        cells=[]
        half=Q(1,2*recurrent['coordinate_bins'][coordinate])
        radius=Q(recurrent['coordinate_radii'][coordinate])
        for indices in product(*(range(n) for n in pair_bins)):
            cell=_cell(terms,indices,pair_bins,cache,max_regions)
            bound=Q(cell['difference_error_upper'])+Q(cell['center_action_lipschitz'])/Q(2*recurrent['action_bins'])+half
            cell['initial_to_recurrent_error_upper']=str(bound)
            cell['handoff_slack']=str(radius-bound)
            cells.append(cell)
        bound=max(Q(c['initial_to_recurrent_error_upper']) for c in cells)
        rows.append({'coordinate':coordinate,'state_axes':list(axes),'mixed_hidden_units':mixed,
            'state_cells':cells,'initial_to_recurrent_error_upper':str(bound),
            'recurrent_relation_radius':str(radius),'handoff_slack':str(radius-bound)})
    ranges=_reachable_indices(recurrent)
    count=prod(b-a+1 for a,b in ranges)
    failed=[r['coordinate'] for r in rows if Q(r['handoff_slack'])<0]
    return {'schema':'ncd.joint-phase-initial-handoff.v1',
        'status':'unresolved' if failed else 'certified','system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent),'state_dim':system.state_dim,
        'action_dim':system.action_dim,'epsilon':recurrent['epsilon'],
        'action_bins':recurrent['action_bins'],'max_phase_regions':max_regions,
        'initial_coordinate_bins':list(bins),'recurrent_coordinate_bins':recurrent['coordinate_bins'],
        'initial_output_error_upper':['1/6']*4,
        'initial_to_recurrent_error_upper':[r['initial_to_recurrent_error_upper'] for r in rows],
        'recurrent_relation_radii':[str(Q(x)) for x in recurrent['coordinate_radii']],
        'recurrent_index_low':[a for a,b in ranges],'recurrent_index_high':[b for a,b in ranges],
        'initial_state_count':81,'recurrent_state_count':count,
        'upper_bound':None if failed else 81+count,
        'minimum_initial_label_count':None if failed else 81,
        'initial_packing':{'observed_axes':[0,1,2,3],'axis_values':['0','1/2','1'],
            'point_count':81,'minimum_output_separation':'1/2','twice_epsilon':'17/50'},
        'failed_coordinates':failed,'coordinate_proofs':rows,
        'mixed_hidden_units_checked':sum(r['mixed_hidden_units'] for r in rows),
        'state_pair_cells_checked':sum(len(r['state_cells']) for r in rows),
        'joint_vertices_checked':sum(c['joint_vertices_checked'] for r in rows for c in r['state_cells']),
        'minimum_handoff_slack':str(min(Q(r['handoff_slack']) for r in rows)),
        'scope':'full unit initial/control cubes; exact shared-control difference phases in four dimensions; all finite horizons through the verified recurrent relation',
        'boundary':'sparse local one-hidden-layer networks; arbitrary dense/deep networks, global minimum and original end-to-end causal requirements remain open'}


def certify_joint_handoff(system,recurrent,*,max_regions=4096):
    if max_regions<1:
        raise ValueError('Positive phase region budget required')
    return _proof(system,recurrent,max_regions)


def verify_joint_handoff(system,recurrent,certificate):
    if certificate.get('schema')!='ncd.joint-phase-initial-handoff.v1':
        raise ValueError('Unsupported joint initial certificate')
    expected=_proof(system,recurrent,certificate['max_phase_regions'])
    if expected!=certificate:
        raise ValueError('Joint phase initial handoff replay mismatch')
    return {'status':'unresolved' if expected['failed_coordinates'] else 'verified',
        'initial_state_count':81,'recurrent_state_count':expected['recurrent_state_count'],
        'upper_bound':expected['upper_bound'],'minimum_initial_label_count':expected['minimum_initial_label_count']}
