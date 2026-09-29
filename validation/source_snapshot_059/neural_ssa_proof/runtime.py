"""Torch-free numeric and rational-interval execution of the explicit SSA."""
from fractions import Fraction as Q
import numpy as np
from ncd.proof_intervals import Interval,hyperbolic_tangent,logarithm
from .compiler import validate_program

class NumericProgram:
    def __init__(self,program):
        validate_program(program);self.p=program;self.nodes=program['nodes'];self.constants={n['id']:float(Q(n['value'])) for n in self.nodes if n['op']=='constant'}
        self.site_by_ref={s['ref']:sid for sid,s in program['sites'].items()}
    def run(self,data,patches=None,trace=True):
        data=np.asarray(data,dtype=np.float64);patches=patches or {}
        if data.ndim!=2 or data.shape[1]!=2 or len(data)<16 or not np.isfinite(data).all():raise ValueError('Finite N>=16 two-column input required')
        if not set(patches)<=set(self.p['sites']):raise ValueError('Unknown intervention variable')
        overrides={}
        for sid,value in patches.items():
            spec=self.p['sites'][sid];value=np.asarray(value,dtype=float)
            if spec['kind']=='vector':
                if value.shape!=(len(data),):raise ValueError('Incompatible source vector length')
            elif value.size!=1:raise ValueError('Scalar intervention required')
            else:value=float(value.reshape(-1)[0])
            if not np.isfinite(value).all():raise ValueError('Non-finite intervention')
            if sid in self.p['positive_intervention_sites'] and np.any(value<float(Q(self.p['positive_intervention_sites'][sid]))):raise ValueError('Intervention outside protected source domain')
            overrides[spec['ref']]=value
        values={};guards=[]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            for n in self.nodes:
                i,op=n['id'],n['op']
                if i in overrides:
                    values[i]=overrides[i]
                    if op=='if':guards.append({'node':i,'intervened':True,'visited_child':None,'unvisited_children':[1,2]})
                    continue
                a=[values[r] for r in n['args']]
                if op=='var':v=data[:,n['index']]
                elif op=='constant':v=self.constants[i]
                elif op=='add':v=a[0]+a[1]
                elif op=='sub':v=a[0]-a[1]
                elif op=='mul':v=a[0]*a[1]
                elif op=='div':
                    cutoff=float(Q(self.p['protected_division_cutoff']));den=np.asarray(a[1]);v=a[0]/np.where(np.abs(den)<cutoff,np.where(den<0,-cutoff,cutoff),den)
                elif op=='tanh':v=np.tanh(a[0])
                elif op=='square':v=a[0]*a[0]
                elif op=='mean':v=float(np.mean(a[0]))
                elif op=='var_stat':v=float(np.var(a[0]))
                elif op=='sqrt':v=np.sqrt(max(float(a[0]),0))
                elif op=='log':v=np.log(max(abs(float(a[0])),float(Q(self.p['protected_log_cutoff']))))
                elif op=='lt':v=bool(a[0]<a[1])
                elif op=='if':
                    v=a[1] if a[0] else a[2];guards.append({'node':i,'visited_child':1 if a[0] else 2,'unvisited_child':2 if a[0] else 1})
                elif op=='clip':v=np.clip(a[0],float(Q(n['lower'])),float(Q(n['upper'])))
                elif op=='argmax':v=int(np.argmax(a))
                else:raise ValueError('Unsupported runtime instruction')
                if i in overrides:v=overrides[i]
                if not np.isfinite(v).all():raise ValueError('Floating backend overflow/undefined; no device guarantee issued')
                values[i]=v
        return {'logits':np.array([values[i] for i in self.p['logits']]),'label':values[self.p['label']],
            'states':{sid:values[s['ref']] for sid,s in self.p['sites'].items()} if trace else {},'guards':guards}


def interval_program(program,data,patches=None):
    validate_program(program);patches=patches or {}
    def enclosure(v):return v if isinstance(v,Interval) else Interval.from_dict(v) if isinstance(v,list) else Interval.point(Q(v))
    rows=[[enclosure(v) for v in row] for row in data]
    if len(rows)<16 or any(len(row)!=2 for row in rows):raise ValueError('Interval input shape')
    overrides={}
    for sid,value in patches.items():
        if sid not in program['sites']:raise ValueError('Unknown source site')
        spec=program['sites'][sid]
        if spec['kind']=='vector':
            if len(value)!=len(rows):raise ValueError('Source vector shape')
            value=[enclosure(v) for v in value]
        else:value=enclosure(value)
        if sid in program['positive_intervention_sites'] and value.lo<Q(program['positive_intervention_sites'][sid]):raise ValueError('Uncertified source protection')
        overrides[spec['ref']]=value
    def unary(value,fn):return [fn(v) for v in value] if isinstance(value,list) else fn(value)
    def binary(a,b,fn):
        if isinstance(a,list):return [fn(x,y) for x,y in zip(a,b if isinstance(b,list) else [b]*len(a))]
        if isinstance(b,list):return [fn(a,y) for y in b]
        return fn(a,b)
    def mean(values):return sum(values,Interval.point(0))/len(values)
    values={};guards=[]
    for n in program['nodes']:
        i,op=n['id'],n['op']
        if i in overrides:
            values[i]=overrides[i]
            if op=='if':guards.append({'node':i,'intervened':True,'possible_children':[],'unvisited_children':[1,2]})
            continue
        a=[values[r] for r in n['args']]
        if op=='var':v=[r[n['index']] for r in rows]
        elif op=='constant':v=Interval.point(Q(n['value']))
        elif op in ('add','sub','mul','div'):
            v=binary(a[0],a[1],{'add':lambda x,y:x+y,'sub':lambda x,y:x-y,'mul':lambda x,y:x*y,'div':lambda x,y:x/y}[op])
        elif op=='square':v=unary(a[0],lambda x:x.square())
        elif op=='tanh':v=unary(a[0],hyperbolic_tangent)
        elif op=='sqrt':v=a[0].sqrt()
        elif op=='log':v=logarithm(a[0])
        elif op=='mean':v=mean(a[0])
        elif op=='var_stat':
            m=mean(a[0]);v=mean([(x-m).square() for x in a[0]])
        elif op=='lt':v={True} if a[0].hi<a[1].lo else {False} if a[0].lo>=a[1].hi else {True,False}
        elif op=='if':
            # Guard refinement is necessary: joining the unrefined false child
            # would incorrectly allow zero denominators around the std floor.
            floor=a[1].lo;raw=a[2];v=Interval(max(floor,raw.lo),max(floor,raw.hi))
            guards.append({'node':i,'possible_children':[1 if z else 2 for z in sorted(a[0])],'branch_boundary_retained':len(a[0])==2})
        elif op=='clip':v=unary(a[0],lambda x:x.clip(Q(n['lower']),Q(n['upper'])))
        elif op=='argmax':
            v=[j for j,x in enumerate(a) if all(x.hi>other.lo if k<j else x.hi>=other.lo for k,other in enumerate(a) if k!=j)]
        else:raise ValueError('Unsupported interval instruction')
        values[i]=overrides.get(i,v)
    return {'logit_enclosures':[values[i].to_dict() for i in program['logits']],
        'possible_labels':values[program['label']],'states':{sid:values[s['ref']] for sid,s in program['sites'].items()},'guards':guards,
        'classification_resolved':len(values[program['label']])==1,'device_rounding_certified':False}
