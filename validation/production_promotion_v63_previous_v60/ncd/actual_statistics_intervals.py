"""Rational enclosures of the actual ncd.statistics and graph_model front ends.

This is intentionally distinct from CDIR crossfit_prediction. Rank/sort
ambiguity remains unresolved. Mathematical operations exclude device rounding.
"""
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations
from .proof_intervals import Interval,DEN,down,up,exponential,logarithm,hyperbolic_tangent,trigonometric
from .discovery_fidelity_proof import average,variance,interval_solve,dependence_interval


def floor_interval(x,minimum):
    m=Q(minimum);return Interval(max(m,x.lo),max(m,x.hi))


def maximum(xs):
    return Interval(max(x.lo for x in xs),max(x.hi for x in xs))


def minimum(xs):
    return Interval(min(x.lo for x in xs),min(x.hi for x in xs))


def certified_order(columns):
    n=len(columns[0])
    if any(len(c)!=n for c in columns):raise ValueError('Sort dimensions')
    order=sorted(range(n),key=lambda i:tuple((c[i].lo+c[i].hi)/2 for c in columns))
    for a,b in zip(order,order[1:]):
        for c in columns:
            u,v=c[a],c[b]
            if u.hi<v.lo:break
            if u.lo==u.hi==v.lo==v.hi:continue
            raise ValueError('Actual statistics lexicographic sorting boundary remains unresolved')
    return order


def quantile(values,q):
    ordered=certified_order([values]);position=Q(q)*(len(values)-1)
    k=position.numerator//position.denominator;fraction=position-k
    return values[ordered[k]] if not fraction else values[ordered[k]]*(1-fraction)+values[ordered[k+1]]*fraction


def standardize(values):
    m=average(values);s=floor_interval(variance(values).sqrt(),Q(1e-6))
    return [(v-m)/s for v in values]


def negative_exponential(x):
    if x.hi>0:raise ValueError('Negative exponential requires nonpositive domain')
    lo=Q(0) if x.lo<=-80 else exponential(Interval.point(x.lo)).lo
    hi=Q(1,DEN) if x.hi<=-80 else exponential(Interval.point(x.hi)).hi
    # e>2 implies exp(-80)<2^-80; this avoids enormous reciprocal intermediates.
    return Interval(lo,hi)


def residual_statistics(y,x):
    # ncd.statistics.residual: sort primary x, secondary y, not the CDIR sort.
    if len(x)<16 or len(y)!=len(x):raise ValueError('Actual residual requires at least 16 rows')
    order=certified_order([x,y]);result=[None]*len(x)
    for parity in (0,1):
        hold,fit=order[parity::2],order[1-parity::2]
        xm=average([x[i] for i in fit]);xs=floor_interval(variance([x[i] for i in fit]).sqrt(),Q(1e-6))
        ym=average([y[i] for i in fit]);ys=floor_interval(variance([y[i] for i in fit]).sqrt(),Q(1e-6))
        z=[(x[i]-xm)/xs for i in fit]
        centers=[quantile(z,Q(q)) for q in (.1,.3,.5,.7,.9)]
        def basis(value):
            v=value.clip(-8,8);v2=v.square()
            return [Interval.point(1),v,v2/4,v2*v/16,trigonometric(v),trigonometric(v,True),
                trigonometric(2*v),trigonometric(2*v,True),hyperbolic_tangent(v),
                *[negative_exponential(-(v-c).square()/2) for c in centers]]
        rows={i:basis((x[i]-xm)/xs) for i in order};width=len(rows[fit[0]])
        gram=[[sum((rows[i][j]*rows[i][k] for i in fit),Interval.point(0))+
            (Q(1e-6) if j==k==0 else Q(.1) if j==k else 0) for k in range(width)] for j in range(width)]
        rhs=[sum((rows[i][j]*((y[i]-ym)/ys) for i in fit),Interval.point(0)) for j in range(width)]
        coefficients=interval_solve(gram,rhs)
        for i in hold:
            prediction=sum((a*b for a,b in zip(rows[i],coefficients)),Interval.point(0))*ys+ym
            result[i]=y[i]-prediction
    return result


def raw_feature(data,index):
    if len(data)<16 or any(len(row)!=2 for row in data):raise ValueError('Actual features require finite N x 2, N>=16')
    raw_x,raw_y=list(zip(*data));x,y=standardize(raw_x),standardize(raw_y)
    if index in (0,1):
        c=average([a*b for a,b in zip(x,y)]);return c if index==0 else c.abs()
    if index==2:return logarithm(floor_interval(variance(raw_x),Q(1e-12))/floor_interval(variance(raw_y),Q(1e-12)))
    if index in (3,4):return average([v.square()*v for v in (x if index==3 else y)])
    if index in (5,6):return average([v.square().square() for v in (x if index==5 else y)])-3
    if index==7:return dependence_interval(x,y)
    if index in (8,9,10,11):
        cause,effect=(x,y) if index in (8,10) else (y,x)
        r=residual_statistics(effect,cause)
        return dependence_interval(cause,r) if index in (8,9) else average([v.square() for v in r])
    if index in (12,13):
        a,b=(x,y) if index==12 else (y,x);return average([u.square()*v for u,v in zip(a,b)])
    raise ValueError('Unknown actual statistical feature')


@lru_cache(None)
def pi_interval():
    def atan(t):
        n=0;term=t;total=Q(0)
        while True:
            total+=term/(2*n+1);n+=1;term*=-t*t;next_term=term/(2*n+1)
            if abs(next_term)<Q(1,DEN*4096):
                return Interval(down(min(total,total+next_term)),up(max(total,total+next_term)))
    return 16*atan(Q(1,5))-4*atan(Q(1,239))


def erf_point(x):
    x=Q(x)
    if x<0:return -erf_point(-x)
    if x>=8:
        # Mills bound erfc(x)<=exp(-x^2)/(x sqrt(pi)), e>5/2, sqrt(pi)>1.
        return Interval(1-Q(2,5)**64/8,1)
    n=0;term=total=x
    while True:
        next_term=-term*x*x*(2*n+1)/((n+1)*(2*n+3))
        if n+1>=x*x and abs(next_term)<Q(1,DEN*4096):
            lo=min(total,total+next_term);hi=max(total,total+next_term)
            return (Interval(lo,hi)*2/pi_interval().sqrt()).clip(-1,1)
        n+=1;term=next_term;total+=term


def erf_interval(x):
    return Interval(erf_point(x.lo).lo,erf_point(x.hi).hi)


def normal_two_sided_tail(z):
    return (1-erf_interval(z.abs()/Interval.point(2).sqrt())).clip(0,1)


def partial_correlation(data,x,y,condition=()):
    cols=list(zip(*data));n=len(data)
    if condition:
        rows=[[Interval.point(1)]+[cols[j][i] for j in condition] for i in range(n)];k=len(rows[0])
        gram=[[sum((r[a]*r[b] for r in rows),Interval.point(0)) for b in range(k)] for a in range(k)]
        # Check that NumPy's default relative SVD cutoff does not truncate the full-rank system.
        inverse=[interval_solve(gram,[Interval.point(int(i==j)) for i in range(k)]) for j in range(k)]
        inv_norm=sum((max(abs(v.lo),abs(v.hi))**2 for col in inverse for v in col),Q(0))
        gram_norm=sum((max(abs(v.lo),abs(v.hi))**2 for row in gram for v in row),Q(0))
        cutoff=Q(1,2**52)*max(n,k)
        if 1/(inv_norm*gram_norm)<=cutoff**4:raise ValueError('Least-squares numerical rank remains unresolved')
        pair=[]
        for col in (cols[x],cols[y]):
            rhs=[sum((r[a]*v for r,v in zip(rows,col)),Interval.point(0)) for a in range(k)]
            beta=interval_solve(gram,rhs)
            pair.append([v-sum((a*b for a,b in zip(row,beta)),Interval.point(0)) for row,v in zip(rows,col)])
    else:pair=[[v-average(cols[j]) for v in cols[j]] for j in (x,y)]
    a,b=pair;num=sum((u*v for u,v in zip(a,b)),Interval.point(0))
    scale=(sum((v.square() for v in a),Interval.point(0))*sum((v.square() for v in b),Interval.point(0))).sqrt()
    return (num/floor_interval(scale,Q(1e-15))).clip(Q(-.999999),Q(.999999))


def fisher_p(data,x,y,condition=()):
    count=len(data)-len(condition)-3
    if count<=0:raise ValueError('Insufficient Fisher rows')
    r=partial_correlation(data,x,y,condition)
    z=logarithm((1+r)/(1-r))/2*Interval.point(count).sqrt()
    return normal_two_sided_tail(z)


def graph_features(data):
    from .graph_model import SWAP_FEATURES
    nodes=len(data[0]);result=[[[Interval.point(0) for _ in range(20)] for _ in range(nodes)] for _ in range(nodes)]
    for i,j in combinations(range(nodes),2):
        pair=[[r[i],r[j]] for r in data];base=[raw_feature(pair,k) for k in range(14)]
        others=[k for k in range(nodes) if k not in (i,j)]
        one=[partial_correlation(data,i,j,(k,)).abs() for k in others]
        two=[partial_correlation(data,i,j,z).abs() for z in combinations(others,2)]
        ps=[fisher_p(data,i,j,z) for size in range(min(2,len(others))+1) for z in combinations(others,size)]
        lo=sum(p.lo>Q(.01) for p in ps);hi=sum(p.hi>Q(.01) for p in ps)
        extra=[minimum(one) if one else base[0].abs(),minimum(two) if two else minimum(one) if one else base[0].abs(),
            maximum(ps),Interval(Q(lo,len(ps)),Q(hi,len(ps))),
            average([partial_correlation(data,i,k).abs() for k in others]) if others else Interval.point(0),
            average([partial_correlation(data,j,k).abs() for k in others]) if others else Interval.point(0)]
        result[i][j]=base+extra;result[j][i]=[result[i][j][k] for k in SWAP_FEATURES];result[j][i][2]=-result[j][i][2]
    return result
