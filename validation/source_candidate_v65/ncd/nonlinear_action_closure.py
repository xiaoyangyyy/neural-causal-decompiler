"""Exact phase-polygon shared-action closure for frozen ReLU realizations.

At a fixed finite-state center, a one-hidden-layer network with two actions
is affine on exact action polygons. Joint cell feasibility retains action
correlations across all active coordinates, including boundary contacts.
"""
from __future__ import annotations
from collections import deque
from fractions import Fraction as Q
from itertools import product

from .continuous_compositional_realization import _quantize
from .continuous_separation import ContinuousReLUSystem, _digest
from .initial_grid_synthesis import (
    _state_bins, initial_grid_initial, initial_grid_output, initial_grid_step,
    verify_initial_grid)
from .reachable_two_stage import _center, _reachable_indices

SQUARE=((Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1)),(Q(0),Q(1)))


def _evaluate(form,point):
    a,b,c=form
    return a*point[0]+b*point[1]+c


def _clip(poly,form):
    """Closed convex clipping, preserving point and segment intersections."""
    if not poly:
        return ()
    result=[]
    previous=poly[-1]
    before=_evaluate(form,previous)
    for current in poly:
        after=_evaluate(form,current)
        if (before>=0)!=(after>=0):
            ratio=before/(before-after)
            result.append(tuple(x+ratio*(y-x) for x,y in zip(previous,current)))
        if after>=0:
            result.append(current)
        previous,before=current,after
    unique=[]
    for point in result:
        if point not in unique:
            unique.append(point)
    return tuple(unique)


def _compile(system,active):
    net=system.transition
    if system.action_dim!=2 or len(net.weights)!=2:
        raise ValueError('Two actions and one hidden ReLU layer required')
    relevant=tuple(j for j in range(len(net.weights[0]))
                   if any(net.weights[1][i][j] for i in active))
    hidden={j:(tuple((k,Q(w)) for k,w in enumerate(net.weights[0][j][:-2]) if w),
               Q(net.weights[0][j][-2]),Q(net.weights[0][j][-1]),
               Q(net.biases[0][j])) for j in relevant}
    output=tuple((Q(net.biases[1][i]),
                  tuple((j,Q(net.weights[1][i][j])) for j in relevant
                        if net.weights[1][i][j])) for i in active)
    return hidden,output


def _regions(compiled,center,cache):
    sparse,outputs=compiled
    hidden={j:(a,b,c+sum((w*center[k] for k,w in state),Q(0)))
            for j,(state,a,b,c) in sparse.items()}
    crossing=tuple(sorted({form for form in hidden.values()
        if min(_evaluate(form,p) for p in SQUARE)<0
        and max(_evaluate(form,p) for p in SQUARE)>0}))
    if crossing not in cache:
        polygons=[SQUARE]
        for form in crossing:
            split=[]
            for poly in polygons:
                scores=tuple(_evaluate(form,p) for p in poly)
                if min(scores)>=0 or max(scores)<=0:
                    split.append(poly)
                else:
                    first=_clip(poly,form)
                    second=_clip(poly,tuple(-x for x in form))
                    if not first or not second:
                        raise ValueError('Action-square split lost a region')
                    split.extend((first,second))
            polygons=split
        cache[crossing]=tuple(polygons)
    pieces=[]
    for poly in cache[crossing]:
        midpoint=tuple(sum((p[k] for p in poly),Q(0))/len(poly) for k in range(2))
        enabled=set()
        for j,form in hidden.items():
            scores=tuple(_evaluate(form,p) for p in poly)
            if _evaluate(form,midpoint)>=0:
                if min(scores)<0:
                    raise ValueError('Positive phase not fixed on polygon')
                enabled.add(j)
            elif max(scores)>0:
                raise ValueError('Negative phase not fixed on polygon')
        forms=[]
        for bias,weights in outputs:
            affine=[Q(0),Q(0),bias]
            for j,weight in weights:
                if j in enabled:
                    for k,x in enumerate(hidden[j]):
                        affine[k]+=weight*x
            forms.append(tuple(affine))
        pieces.append((poly,tuple(forms)))
    return crossing,tuple(pieces)


def _feasible(indices,bins,pieces):
    for poly,forms in pieces:
        remaining=poly
        for q,n,(a,b,c) in zip(indices,bins,forms):
            remaining=_clip(remaining,(a,b,c-Q(q,n)))
            remaining=_clip(remaining,(-a,-b,Q(q+1,n)-c))
            if not remaining:
                break
        if remaining:
            return True
    return False


def _expand(indices,active,dimension):
    full=[0]*dimension
    for i,q in zip(active,indices):
        full[i]=q
    return tuple(full)


def _successors(compiled,center,target_bins,image_ranges,cache):
    crossing,pieces=_regions(compiled,center,cache)
    ranges=[]
    for i,(n,(left,right)) in enumerate(zip(target_bins,image_ranges)):
        values=tuple(_evaluate(forms[i],p) for poly,forms in pieces for p in poly)
        low,high=min(values),max(values)
        if not 0<=low<=high<=1:
            raise ValueError('Action-region image outside certified unit cube')
        a,b=_quantize(low,n),_quantize(high,n)
        if not left<=a<=b<=right:
            raise ValueError('Action-region image outside recurrent enclosure')
        ranges.append(range(a,b+1))
    candidates=tuple(product(*ranges))
    successors=tuple(q for q in candidates if _feasible(q,target_bins,pieces))
    if not successors:
        raise ValueError('Nonempty action square has no successor cell')
    return successors,len(candidates),len(crossing),len(pieces)


def _verify_initial_certificate(system,recurrent,initial):
    schema=initial.get('schema')
    if schema=='ncd.optimal-initial-grid.v1':
        return verify_initial_grid(system,recurrent,initial)
    if schema=='ncd.phase-initial-handoff.v1':
        from .phase_initial_handoff import verify_phase_handoff
        return verify_phase_handoff(system,recurrent,initial)
    if schema=='ncd.joint-phase-initial-handoff.v1':
        from .joint_phase_initial_handoff import verify_joint_handoff
        result=verify_joint_handoff(system,recurrent,initial)
        if result['status']!='verified':
            raise ValueError('Unresolved joint initial handoff cannot support a closure')
        return result
    raise ValueError('Unsupported initial simulation certificate')


def _proof(system: ContinuousReLUSystem,recurrent: dict,initial: dict):
    prior=_verify_initial_certificate(system,recurrent,initial)
    if prior['status']!='verified':
        raise ValueError('Certified initial handoff required')
    initial_bins=tuple(initial['initial_coordinate_bins'])
    recurrent_bins=tuple(recurrent['coordinate_bins'])
    active=tuple(i for i,(a,b) in enumerate(zip(initial_bins,recurrent_bins)) if a>1 or b>1)
    target_bins=tuple(recurrent_bins[i] for i in active)
    full_ranges=_reachable_indices(recurrent)
    image_ranges=tuple(full_ranges[i] for i in active)
    compiled=_compile(system,active)
    cache={}
    initial_graph={}
    recurrent_graph={}
    initial_union=set()
    initial_candidates=recurrent_candidates=0
    phase_counts=set()
    region_counts=set()
    for indices in product(*(range(initial_bins[i]) for i in active)):
        center=_center(_expand(indices,active,system.state_dim),initial_bins)
        successors,count,phases,regions=_successors(compiled,center,target_bins,image_ranges,cache)
        initial_graph[indices]=successors
        initial_union.update(successors)
        initial_candidates+=count
        phase_counts.add(phases)
        region_counts.add(regions)
    if len(initial_graph)!=prior['initial_state_count']:
        raise ValueError('Initial grid enumeration incomplete')
    seen=set(initial_union)
    pending=deque(sorted(seen))
    while pending:
        indices=pending.popleft()
        center=_center(_expand(indices,active,system.state_dim),recurrent_bins)
        successors,count,phases,regions=_successors(compiled,center,target_bins,image_ranges,cache)
        recurrent_graph[indices]=successors
        recurrent_candidates+=count
        phase_counts.add(phases)
        region_counts.add(regions)
        for target in successors:
            if target not in seen:
                seen.add(target)
                pending.append(target)
    if set(recurrent_graph)!=seen or not all(set(v)<=seen for v in recurrent_graph.values()):
        raise ValueError('Recurrent action graph not closed')
    return {
        'schema':'ncd.nonlinear-action-closure.v1',
        'status':'certified',
        'system_sha256':_digest(system.to_dict()),
        'recurrent_certificate_sha256':_digest(recurrent),
        'initial_grid_certificate_sha256':_digest(initial),
        'state_dim':system.state_dim,'action_dim':system.action_dim,
        'epsilon':recurrent['epsilon'],'action_bins':recurrent['action_bins'],
        'active_coordinates':list(active),
        'initial_coordinate_bins':list(initial_bins),
        'recurrent_coordinate_bins':list(recurrent_bins),
        'initial_graph':[{'state':list(q),'successors':[list(v) for v in initial_graph[q]]}
                         for q in sorted(initial_graph)],
        'recurrent_graph':[{'state':list(q),'successors':[list(v) for v in recurrent_graph[q]]}
                           for q in sorted(recurrent_graph)],
        'recurrent_active_indices':[list(q) for q in sorted(seen)],
        'initial_state_count':len(initial_graph),
        'recurrent_state_count':len(seen),
        'upper_bound':len(initial_graph)+len(seen),
        'previous_upper_bound':prior['upper_bound'],
        'distinct_initial_successors':len(initial_union),
        'initial_candidate_edges':initial_candidates,
        'initial_retained_edges':sum(map(len,initial_graph.values())),
        'recurrent_candidate_edges':recurrent_candidates,
        'recurrent_retained_edges':sum(map(len,recurrent_graph.values())),
        'action_phase_boundary_counts':sorted(phase_counts),
        'action_region_counts':sorted(region_counts),
        'source_centers_checked':len(initial_graph)+len(recurrent_graph),
        'scope':'full unit initial/action cubes, all finite horizons',
        'boundary':'least closed conservative shared-action phase-polygon graph; not the exact concrete reachable set or global minimum',
    }


def certify_nonlinear_closure(system,recurrent,initial):
    return _proof(system,recurrent,initial)


def verify_nonlinear_closure(system,recurrent,initial,certificate):
    if certificate.get('schema')!='ncd.nonlinear-action-closure.v1':
        raise ValueError('Unsupported nonlinear closure certificate')
    expected=_proof(system,recurrent,initial)
    if certificate!=expected:
        raise ValueError('Nonlinear action-closure replay mismatch')
    return {'status':'verified','initial_state_count':expected['initial_state_count'],
            'recurrent_state_count':expected['recurrent_state_count'],
            'upper_bound':expected['upper_bound']}


def _validate(system,recurrent,initial,certificate,state):
    _state_bins(system,recurrent,initial,state)
    active=certificate['active_coordinates']
    indices=tuple(state[1][i] for i in active)
    graph=certificate['initial_graph'] if state[0]=='initial' else certificate['recurrent_graph']
    for row in graph:
        if tuple(row['state'])==indices:
            return row
    raise ValueError('State outside certified nonlinear closure')


def nonlinear_initial(system,recurrent,initial,certificate,point):
    state=initial_grid_initial(system,initial,point)
    _validate(system,recurrent,initial,certificate,state)
    return state


def nonlinear_output(system,recurrent,initial,certificate,state):
    _validate(system,recurrent,initial,certificate,state)
    return initial_grid_output(system,recurrent,initial,state)


def nonlinear_step(system,recurrent,initial,certificate,state,action):
    row=_validate(system,recurrent,initial,certificate,state)
    target=initial_grid_step(system,recurrent,initial,state,action)
    _validate(system,recurrent,initial,certificate,target)
    indices=[target[1][i] for i in certificate['active_coordinates']]
    if indices not in row['successors']:
        raise ValueError('Runtime successor outside certified nonlinear graph')
    return target



def _export_program(system,recurrent,initial,certificate):
    verify_nonlinear_closure(system,recurrent,initial,certificate)
    active=tuple(certificate['active_coordinates'])
    compiled=_compile(system,active)
    cache={}
    templates=[]
    template_ids={}
    graphs=[]
    for name,bins in (('initial_graph',initial['initial_coordinate_bins']),
                      ('recurrent_graph',recurrent['coordinate_bins'])):
        rows=[]
        for source in certificate[name]:
            center=_center(_expand(source['state'],active,system.state_dim),bins)
            _,pieces=_regions(compiled,center,cache)
            offsets=tuple(form[-1] for form in pieces[0][1])
            template={
                'vertices':[[[str(x) for x in point] for point in poly] for poly,forms in pieces],
                'affine_forms':[[[str(a),str(b),str(c-offsets[i])]
                    for i,(a,b,c) in enumerate(forms)] for poly,forms in pieces]}
            key=_digest(template)
            if key not in template_ids:
                template_ids[key]=len(templates)
                templates.append(template)
            rows.append({'state':source['state'],'successors':source['successors'],
                'output':[str(center[i]) for i in range(4)],
                'action_template':template_ids[key],
                'state_output_offsets':[str(x) for x in offsets]})
        graphs.append(rows)
    return {'schema':'ncd.nonlinear-action-program.v1','status':'certified',
        'system_sha256':_digest(system.to_dict()),
        'closure_certificate_sha256':_digest(certificate),
        'state_dim':system.state_dim,'action_dim':system.action_dim,
        'epsilon':recurrent['epsilon'],'action_bins':recurrent['action_bins'],
        'active_coordinates':list(active),
        'initial_coordinate_bins':initial['initial_coordinate_bins'],
        'recurrent_coordinate_bins':recurrent['coordinate_bins'],
        'action_templates':templates,'initial_graph':graphs[0],'recurrent_graph':graphs[1],
        'initial_state_count':certificate['initial_state_count'],
        'recurrent_state_count':certificate['recurrent_state_count'],
        'upper_bound':certificate['upper_bound'],
        'scope':'standalone finite-state program; full unit initial/action cubes; all finite horizons',
        'boundary':'certified realization upper, not a globally minimum causal program'}


def export_nonlinear_program(system,recurrent,initial,certificate):
    """Compile a proof-bound program whose execution needs no source network."""
    return _export_program(system,recurrent,initial,certificate)


def verify_nonlinear_program(system,recurrent,initial,certificate,program):
    if program.get('schema')!='ncd.nonlinear-action-program.v1':
        raise ValueError('Unsupported nonlinear program')
    expected=_export_program(system,recurrent,initial,certificate)
    if program!=expected:
        raise ValueError('Standalone nonlinear program replay mismatch')
    return {'status':'verified','upper_bound':expected['upper_bound'],
            'action_template_count':len(expected['action_templates'])}


def _program_row(program,state):
    if not isinstance(state,tuple) or len(state)!=2:
        raise ValueError('Malformed standalone program state')
    phase,indices=state
    if phase not in ('initial','recurrent'):
        raise ValueError('Unknown standalone program phase')
    bins=program['initial_coordinate_bins'] if phase=='initial' else program['recurrent_coordinate_bins']
    if (not isinstance(indices,(tuple,list)) or len(indices)!=program['state_dim']
            or not all(isinstance(q,int) and not isinstance(q,bool) and 0<=q<n
                       for q,n in zip(indices,bins))):
        raise ValueError('Malformed standalone program indices')
    active=[indices[i] for i in program['active_coordinates']]
    for row in program[phase+'_graph']:
        if row['state']==active:
            return row
    raise ValueError('State outside standalone nonlinear program')


def program_initial(program,point):
    if len(point)!=program['state_dim'] or any(not 0<=Q(x)<=1 for x in point):
        raise ValueError('Initial point outside program unit cube')
    state=('initial',tuple(_quantize(Q(x),n) for x,n in zip(point,program['initial_coordinate_bins'])))
    _program_row(program,state)
    return state


def program_output(program,state):
    return tuple(Q(x) for x in _program_row(program,state)['output'])


def _inside(poly,point):
    return all((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])>=0
               for a,b in zip(poly,poly[1:]+poly[:1]))


def program_step(program,state,action):
    row=_program_row(program,state)
    if len(action)!=program['action_dim'] or any(not 0<=Q(x)<=1 for x in action):
        raise ValueError('Action outside program unit cube')
    m=program['action_bins']
    controls=tuple(Q(2*_quantize(Q(a),m)+1,2*m) for a in action)
    template=program['action_templates'][row['action_template']]
    for raw_poly,raw_forms in zip(template['vertices'],template['affine_forms']):
        poly=tuple(tuple(Q(x) for x in p) for p in raw_poly)
        if _inside(poly,controls):
            image=tuple(_evaluate(tuple(Q(x) for x in form),controls)+Q(offset)
                        for form,offset in zip(raw_forms,row['state_output_offsets']))
            break
    else:
        raise ValueError('Action outside compiled phase regions')
    active=program['active_coordinates']
    bins=program['recurrent_coordinate_bins']
    if any(not 0<=x<=1 for x in image):
        raise ValueError('Compiled image outside unit cube')
    indices=tuple(_quantize(x,bins[i]) for i,x in zip(active,image))
    if list(indices) not in row['successors']:
        raise ValueError('Compiled successor outside certified graph')
    target=('recurrent',_expand(indices,active,program['state_dim']))
    _program_row(program,target)
    return target
