"""Exact bounded interchange diagrams for a declared fixed linear read/write map.

This validates the numerical intervention interface. Neural continuation and
nonlinear intermediate computation are separate proof obligations.
"""
from fractions import Fraction as Q
from .proof_intervals import Interval


def certify_interchange(read,write,base_domain,source_domain,masks,epsilon='1/100',independent_sources=False):
    R=[[Q(x) for x in row] for row in read];W=[[Q(x) for x in row] for row in write]
    k=len(R);d=len(base_domain);base=[Interval.from_dict(x) for x in base_domain]
    if type(independent_sources)is not bool:raise ValueError('Source mode must be Boolean')
    sources=([[Interval.from_dict(x) for x in row] for row in source_domain] if independent_sources else
             [[Interval.from_dict(x) for x in source_domain]]*k)
    source=sources[0] if sources else []
    if not k or not d or len(sources)!=k or any(len(row)!=d for row in sources) or len(source)!=d or any(len(r)!=d for r in R) or len(W)!=d or any(len(w)!=k for w in W):raise ValueError('Read/write dimensions')
    if Q(epsilon)<0 or not masks or len({tuple(m) for m in masks})!=len(masks):raise ValueError('Invalid interchange contract')
    if any(len(m)!=k or any(type(x)is not bool for x in m) or not any(m) for m in masks):raise ValueError('Nonempty Boolean masks required')
    matrix=[[sum((R[i][a]*W[a][j] for a in range(d)),Q(0))-(1 if i==j else 0) for j in range(k)] for i in range(k)]
    rows=[]
    for mask in masks:
        bounds=[]
        for i in range(k):
            coefficients=[sum((matrix[i][j]*R[j][a] for j in range(k) if mask[j]),Q(0)) for a in range(d)]
            # Exact extrema of a linear form; preserve shared-source cancellations.
            lo=hi=Q(0)
            if independent_sources:
                # Only the base is shared. Different source variables cannot cancel.
                for c,b in zip(coefficients,base):
                    ends=[-c*b.lo,-c*b.hi];lo+=min(ends);hi+=max(ends)
                for j in range(k):
                    if mask[j]:
                        for a in range(d):
                            c=matrix[i][j]*R[j][a];v=sources[j][a]
                            ends=[c*v.lo,c*v.hi];lo+=min(ends);hi+=max(ends)
            else:
                for c,v,b in zip(coefficients,source,base):
                    ends=[c*(v.lo-b.hi),c*(v.hi-b.lo)];lo+=min(ends);hi+=max(ends)
            bounds.append([str(lo),str(hi)])
        rows.append({'mask':mask,'coordinate_error_intervals':bounds,
                     'within_tolerance':all(max(abs(Q(a)),abs(Q(b)))<=Q(epsilon) for a,b in bounds)})
    return {'schema':'ncd.linear-interchange.v1','status':'proved' if all(r['within_tolerance'] for r in rows) else 'refuted',
        'read':[[str(x) for x in r] for r in R],'write':[[str(x) for x in r] for r in W],
        'base_domain':base_domain,'source_domain':source_domain,'independent_sources':independent_sources,'masks':masks,'epsilon':str(Q(epsilon)),
        'read_write_residual':[[str(x) for x in r] for r in matrix],'rows':rows,
        'scope':('all independent source_j and base states in the declared boxes; tau(h)=R*h, s_j=(R*source_j)_j, h<-h+W*mask*(s-R*h)' if independent_sources else 'all base/source states in the declared boxes and declared masks for tau(h)=R*h and h<-h+W*mask*(R*source-R*h)'),
        'neural_continuation_verified':False,'mapping_family_impossibility_proved':False}


def verify_interchange(certificate):
    expected=certify_interchange(certificate['read'],certificate['write'],certificate['base_domain'],certificate['source_domain'],certificate['masks'],certificate['epsilon'],certificate.get('independent_sources',False))
    if 'independent_sources' not in certificate:expected.pop('independent_sources')
    if expected!=certificate:raise ValueError('Interchange certificate mismatch')
    return {'status':'verified','conclusion':certificate['status'],'neural_continuation_verified':False}
