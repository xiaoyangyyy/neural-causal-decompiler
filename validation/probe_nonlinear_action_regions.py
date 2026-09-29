"""Exploratory exact action-region geometry; not an accepted machine proof."""
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from ncd.continuous_compositional_realization import _quantize,value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json,save_json,digest
from ncd.reachable_two_stage import _center,_reachable_indices

ROOT=Path(__file__).resolve().parents[1]


def evaluate(form,point):
    a,b,c=form
    return a*point[0]+b*point[1]+c


def clip(poly,form):
    if not poly:return ()
    result=[]
    previous=poly[-1]
    before=evaluate(form,previous)
    for current in poly:
        after=evaluate(form,current)
        if (before>=0)!=(after>=0):
            ratio=before/(before-after)
            result.append(tuple(x+ratio*(y-x) for x,y in zip(previous,current)))
        if after>=0:result.append(current)
        previous,before=current,after
    unique=[]
    for point in result:
        if point not in unique:unique.append(point)
    return tuple(unique)


def regions(system,center,active):
    net=system.transition
    if system.action_dim!=2 or len(net.weights)!=2:
        raise ValueError('Two actions and a one-hidden-layer network required')
    relevant=[j for j in range(len(net.weights[0]))
              if any(net.weights[1][i][j] for i in active)]
    hidden={j:(Q(net.weights[0][j][-2]),Q(net.weights[0][j][-1]),
        Q(net.biases[0][j])+sum((Q(w)*x for w,x in zip(net.weights[0][j][:-2],center)),Q(0)))
        for j in relevant}
    crossing=sorted({form for form in hidden.values()
        if min(evaluate(form,p) for p in ((Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1)),(Q(0),Q(1))))<0
        and max(evaluate(form,p) for p in ((Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1)),(Q(0),Q(1))))>0})
    polygons=[((Q(0),Q(0)),(Q(1),Q(0)),(Q(1),Q(1)),(Q(0),Q(1)))]
    for form in crossing:
        split=[]
        for poly in polygons:
            scores=[evaluate(form,p) for p in poly]
            if min(scores)>=0 or max(scores)<=0:
                split.append(poly)
            else:
                split.extend((clip(poly,form),clip(poly,tuple(-x for x in form))))
        polygons=split
    result=[]
    for poly in polygons:
        midpoint=tuple(sum((p[i] for p in poly),Q(0))/len(poly) for i in range(2))
        enabled={j for j,form in hidden.items() if evaluate(form,midpoint)>=0}
        forms=[]
        for i in active:
            affine=[Q(0),Q(0),Q(net.biases[1][i])]
            for j in enabled:
                weight=Q(net.weights[1][i][j])
                for k,x in enumerate(hidden[j]):affine[k]+=weight*x
            forms.append(tuple(affine))
        for point in poly:
            actual=value(net,list(center)+list(point))
            if any(evaluate(form,point)!=actual[i] for form,i in zip(forms,active)):
                raise ValueError('Exact action-region extraction mismatch')
        result.append((poly,tuple(forms)))
    return crossing,tuple(result)


def feasible(indices,bins,pieces):
    for poly,forms in pieces:
        remaining=poly
        for q,n,(a,b,c) in zip(indices,bins,forms):
            remaining=clip(remaining,(a,b,c-Q(q,n)))
            remaining=clip(remaining,(-a,-b,Q(q+1,n)-c))
            if not remaining:break
        if remaining:return True
    return False


def main():
    source=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/'d_128'
    model=source/'system.json'
    system=ContinuousReLUSystem.from_dict(read_json(model))
    recurrent=read_json(source/'certificate.json')
    initial=read_json(ROOT/'runs'/'optimal_initial_grid_v1'/'seed_6101'/'d_128'/'certificate.json')
    active=tuple(i for i,n in enumerate(recurrent['coordinate_bins']) if n>1)
    target_bins=tuple(recurrent['coordinate_bins'][i] for i in active)
    target_ranges=tuple(_reachable_indices(recurrent)[i] for i in active)
    bins=initial['initial_coordinate_bins']
    starts=(tuple(0 for n in bins),tuple(n-1 for n in bins),tuple((n-1)//2 for n in bins))
    rows=[]
    for indices in starts:
        center=_center(indices,bins)
        crossing,pieces=regions(system,center,active)
        bounds=tuple((min(evaluate(forms[i],p) for poly,forms in pieces for p in poly),
                      max(evaluate(forms[i],p) for poly,forms in pieces for p in poly))
                     for i in range(len(active)))
        coordinate_ranges=tuple((_quantize(low,n),_quantize(high,n)) for (low,high),n in zip(bounds,target_bins))
        cartesian=[];joint=[]
        for candidate in product(*(range(a,b+1) for a,b in target_ranges)):
            if all(a<=q<=b for q,(a,b) in zip(candidate,coordinate_ranges)):
                cartesian.append(candidate)
                if feasible(candidate,target_bins,pieces):joint.append(candidate)
        row={'initial_active_indices':[indices[i] for i in active],
             'action_phase_boundary_count':len(crossing),'action_region_count':len(pieces),
             'coordinate_box_successors':len(cartesian),'joint_action_successors':len(joint),
             'joint_successor_active_indices':[list(q) for q in joint]}
        rows.append(row)
        print(row,flush=True)
    report={'status':'exploratory','model_sha256':digest(model),'seed':6101,'state_dim':128,
            'active_coordinates':list(active),'cases':rows,
            'scope':'three fixed initial centers; continuous action cube; closed target cells',
            'boundary':'not a full closed recurrent graph or accepted finite realization certificate'}
    save_json(ROOT/'validation'/'nonlinear_action_region_probe.json',report)
    return report


if __name__=='__main__':main()
