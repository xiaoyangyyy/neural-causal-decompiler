"""Actual FX prefix extraction and exact normalization/intervention proof.

The learned encoder/head remain the frozen neural continuation. This is a
partial decompilation theorem, not full causal-algorithm recovery. Constants and
operators have explicit real mathematical semantics; device rounding is separate.
"""
from fractions import Fraction as Q
from pathlib import Path
from math import isqrt
import hashlib,json
from ncd.io import digest
from ncd.cdir import Node
from ncd.proof_intervals import Interval

MODEL_SOURCE='ee732b8f06c153951e6f3e559c9511591e8e6e632d815766c5192aff50bb3602'
PROTECTED_FLOOR=Q(1e-8)
INTERVENTION_ORDER=('mean_x','mean_y','std_x','std_y')


def key(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def export_graph(checkpoint):
    import torch
    from ncd.model import load_model
    if digest('ncd/model.py')!=MODEL_SOURCE:raise ValueError('Unreviewed model source')
    model=load_model(checkpoint);graph=torch.fx.symbolic_trace(model)
    def encode(value):
        if isinstance(value,torch.fx.Node):return {'node':value.name}
        if isinstance(value,(list,tuple)):return [encode(x) for x in value]
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,(str,int,float,bool)) or value is None:return value
        return {'repr':repr(value)}
    rows=[]
    for node in graph.graph.nodes:
        target=node.target if isinstance(node.target,str) else node.target.__module__+'.'+node.target.__name__
        rows.append({'name':node.name,'op':node.op,'target':target,'args':encode(node.args),'kwargs':encode(node.kwargs)})
    return {'schema':'ncd.frozen-discoverer-fx.v1','checkpoint_sha256':digest(checkpoint),
        'model_source_sha256':MODEL_SOURCE,'torch_version':str(torch.__version__),'width':model.width,
        'nodes':rows,'graph_sha256':hashlib.sha256(key(rows).encode()).hexdigest()}


def normalizer_branches(export):
    # Match the existing operations, not replacement modules or an inserted mean.
    rows=export['nodes'];by_name={r['name']:r for r in rows}
    if len(by_name)!=len(rows):raise ValueError('Duplicate FX node')
    def parent(row,index):
        value=row['args'][index]
        if not isinstance(value,dict) or set(value)!={'node'} or value['node'] not in by_name:raise ValueError('FX argument is not a graph edge')
        return by_name[value['node']]
    def require(row,op,target,kwargs=None):
        if row['op']!=op or row['target']!=target or (kwargs is not None and row['kwargs']!=kwargs):raise ValueError('Unsupported actual normalization operation')
    branches=[]
    for end in [r for r in rows if r['op']=='call_module' and r['target']=='encoder.0']:
        clipped=parent(end,0);require(clipped,'call_method','clamp',{})
        z=parent(clipped,0);require(z,'call_function','_operator.truediv',{})
        centered=parent(z,0);require(centered,'call_function','_operator.sub',{})
        denominator=parent(z,1);require(denominator,'call_method','clamp_min',{})
        mean=parent(centered,1);require(mean,'call_method','mean',{'dim':1,'keepdim':True})
        std=parent(denominator,0);require(std,'call_method','std',{'dim':1,'keepdim':True,'unbiased':False})
        raw=parent(centered,0)
        if parent(mean,0)!=raw or parent(std,0)!=raw:raise ValueError('Mean/std are not computed on the same raw input')
        if raw['op']=='placeholder' and raw['target']=='x':columns=[0,1]
        elif raw['op']=='call_method' and raw['target']=='flip' and raw['args'][1:]==[-1] and raw['kwargs']=={}:
            original=parent(raw,0)
            if original['op']!='placeholder' or original['target']!='x':raise ValueError('Unsupported swapped source')
            columns=[1,0]
        else:raise ValueError('Unsupported preprocessing source')
        if len(clipped['args'])!=3 or len(denominator['args'])!=2:raise ValueError('Normalization bounds arity')
        floor=Q(denominator['args'][1]);lower,upper=map(Q,clipped['args'][1:])
        if floor<=PROTECTED_FLOOR or lower>=upper:raise ValueError('Unresolved protection or clipping domain')
        branches.append({'columns':columns,'floor':str(floor),'clip':[str(lower),str(upper)],
            'nodes':{'mean':mean['name'],'std':std['name'],'denominator':denominator['name'],
                'centered':centered['name'],'normalized':z['name'],'clipped':clipped['name']}})
    if len(branches)!=2 or {tuple(b['columns']) for b in branches}!={(0,1),(1,0)}:raise ValueError('Both actual normalization branches required')
    if branches[0]['floor']!=branches[1]['floor'] or branches[0]['clip']!=branches[1]['clip']:raise ValueError('Incompatible swapped normalization contracts')
    return branches


def cdir_expression(column,floor):
    raw=Node('var',index=column);m=Node('mean',(raw,));s=Node('sqrt',(Node('var_stat',(raw,)),))
    f=Node('constant',value=float(Q(floor)))
    denominator=Node('if',(Node('lt',(s,f)),f,s))
    return Node('div',(Node('sub',(raw,m)),denominator)).to_dict()


def compile_program(export):
    branches=normalizer_branches(export);floor=branches[0]['floor'];bounds=branches[0]['clip']
    # Clip is an explicit typed intrinsic. Using an abs-difference identity in
    # floating execution can catastrophically cancel for huge normalized inputs.
    return {'schema':'ncd.normalizer-cdir-program.v1','columns':2,'minimum_samples':16,
        'floor':floor,'clip':bounds,'operators':['mean','var_stat','sqrt','lt','if','sub','div','clip'],
        'coordinates':[{'column':j,'normalized_expression':cdir_expression(j,floor)} for j in range(2)],
        'clip_intrinsic':'coordinatewise max(lower,min(upper,z)); three disjoint exact cases',
        'control_variables':list(INTERVENTION_ORDER),
        'neural_remainder':'the original unchanged frozen encoder/head continuation; not symbolically recovered'}


class Algebraic:
    """Exact value numerator/sqrt(denominator_squared), with rational coefficients."""
    def __init__(self,numerator=0,denominator_squared=1):
        n,d=Q(numerator),Q(denominator_squared)
        if d<=0:raise ValueError('Positive radical denominator')
        a,b=isqrt(d.numerator),isqrt(d.denominator)
        if a*a==d.numerator and b*b==d.denominator:n/=Q(a,b);d=Q(1)
        if not n:d=Q(1)
        self.n,self.d=n,d
    @classmethod
    def sqrt(cls,value):
        value=Q(value)
        if value<0:raise ValueError('Negative variance')
        return cls(value,value) if value else cls()
    def compare_rational(self,value):
        value=Q(value)
        if value==0:return (self.n>0)-(self.n<0)
        if value<0:return -Algebraic(-self.n,self.d).compare_rational(-value)
        if self.n<=0:return -1
        delta=self.n*self.n-value*value*self.d
        return (delta>0)-(delta<0)
    def clip(self,lower,upper):
        if self.compare_rational(lower)<0:return Algebraic(lower)
        if self.compare_rational(upper)>0:return Algebraic(upper)
        return self
    def bounds(self):return Interval.point(self.n)/Interval.point(self.d).sqrt()
    def to_dict(self):return {'numerator':str(self.n),'denominator_squared':str(self.d)}
    def approximate(self):
        v=self.bounds();return float((v.lo+v.hi)/2)


def rational_data(data):
    rows=[[Q(v) for v in row] for row in data]
    if len(rows)<16 or any(len(row)!=2 for row in rows):raise ValueError('At least 16 finite rational two-column rows required')
    return rows


def exact_statistics(rows,column,floor):
    values=[row[column] for row in rows];mean=sum(values,Q(0))/len(values)
    variance=sum(((v-mean)**2 for v in values),Q(0))/len(values);std=Algebraic.sqrt(variance)
    low=std.compare_rational(floor)<0;denominator=Algebraic(floor) if low else std
    return {'mean':Algebraic(mean),'std':std,'denominator':denominator,'guard':'floor' if low else 'std',
        'variance':variance}


def execute_exact(program,data,sources=None,mask=None):
    validate_program(program)
    rows=rational_data(data);floor=Q(program['floor']);lower,upper=map(Q,program['clip'])
    mask=[False]*4 if mask is None else list(mask)
    if len(mask)!=4 or any(type(v)is not bool for v in mask):raise ValueError('Four Boolean control variables required')
    if any(mask) and (sources is None or len(sources)!=4):raise ValueError('Independent source for each variable required')
    source_stats=[]
    for j in range(4):
        source_stats.append(exact_statistics(rational_data(sources[j]),j%2,floor) if mask[j] else None)
    states={};output=[];guards=[]
    for column in range(2):
        natural=exact_statistics(rows,column,floor)
        mean=source_stats[column]['mean'] if mask[column] else natural['mean']
        den=source_stats[column+2]['denominator'] if mask[column+2] else natural['denominator']
        if mean.d!=1 or den.compare_rational(floor)<0:raise ValueError('Invalid source-interface value')
        centered=[row[column]-mean.n for row in rows]
        normalized=[Algebraic(a*den.d/den.n,den.d) for a in centered]
        clipped=[v.clip(lower,upper) for v in normalized]
        suffix='xy'[column]
        states['mean_'+suffix]=mean.to_dict();states['std_raw_'+suffix]=natural['std'].to_dict();states['std_'+suffix]=den.to_dict()
        states['centered_'+suffix]=[Algebraic(v).to_dict() for v in centered]
        states['normalized_'+suffix]=[v.to_dict() for v in normalized]
        states['clipped_'+suffix]=[v.to_dict() for v in clipped]
        guards.append({'column':column,'natural_std_case':natural['guard'],
            'visited_conditional_child':1 if natural['guard']=='floor' else 2,
            'unvisited_conditional_child':2 if natural['guard']=='floor' else 1,
            'clipping_cases':['lower' if v.compare_rational(lower)<0 else 'upper' if v.compare_rational(upper)>0 else 'inside' for v in normalized]})
        output.append(clipped)
    return {'schema':'ncd.exact-normalizer-execution.v1','mask':mask,'states':states,'guards':guards,
        'outputs':[[output[j][i].to_dict() for j in range(2)] for i in range(len(rows))],
        'computed_mean_and_raw_std_visited':True,'unexecuted_conditional_children_not_covered':True}


def validate_program(program):
    if program.get('schema')!='ncd.normalizer-cdir-program.v1' or program.get('columns')!=2 or program.get('minimum_samples')!=16:raise ValueError('Program shape contract')
    floor=Q(program['floor']);lower,upper=map(Q,program['clip'])
    if floor<=PROTECTED_FLOOR or lower>=upper or program['control_variables']!=list(INTERVENTION_ORDER):raise ValueError('Program interface/protection contract')
    if program['coordinates']!=[{'column':j,'normalized_expression':cdir_expression(j,floor)} for j in range(2)]:raise ValueError('Unsupported or modified CDIR normalization expression')
    if program['operators']!=['mean','var_stat','sqrt','lt','if','sub','div','clip'] or program['clip_intrinsic']!='coordinatewise max(lower,min(upper,z)); three disjoint exact cases':raise ValueError('Unsupported intrinsic semantics')
    if program['neural_remainder']!='the original unchanged frozen encoder/head continuation; not symbolically recovered':raise ValueError('Unsupported full-decompilation claim')
    return True


def boundary_edges(export,branches):
    inside={name for branch in branches for name in branch['nodes'].values()}
    rows=export['nodes'];edges=[]
    def refs(value):
        if isinstance(value,dict):
            if set(value)=={'node'}:yield value['node']
            else:
                for v in value.values():yield from refs(v)
        elif isinstance(value,list):
            for v in value:yield from refs(v)
    for row in rows:
        if row['name'] not in inside:
            for source in refs([row['args'],row['kwargs']]):
                if source in inside:edges.append({'source':source,'consumer':row['name']})
    expected={branch['nodes'][name] for branch in branches for name in ('denominator','clipped')}
    if {e['source'] for e in edges}!=expected:raise ValueError('Unverified prefix state escapes into the continuation')
    return edges


def certify_normalizer(checkpoint):
    export=export_graph(checkpoint);branches=normalizer_branches(export);program=compile_program(export)
    edges=boundary_edges(export,branches);floor=Q(program['floor']);lower,upper=map(Q,program['clip'])
    masks=[[bool(i&(1<<j)) for j in range(4)] for i in range(16)]
    bindings={p:digest(p) for p in ['ncd/model.py','ncd/cdir.py','ncd/program_trace.py','ncd/proof_intervals.py','normalizer_proof/realization.py']}
    return {'schema':'ncd.actual-normalizer-composition.v1','status':'proved','checkpoint':str(checkpoint),
        'source_sha256':bindings,'fx_export':export,'branches':branches,'program':program,
        'read_map':'Read actual mean/clamp_min node coordinate; swap branch uses the corresponding reversed coordinate.',
        'write_map':'Replace only that coordinate in both corresponding actual FX occurrences; independent source_j for each selected logical variable.',
        'masks':masks,'source_family':'four independently selected valid source datasets; no shared-source cancellation assumption',
        'domain':'every finite real N x 2 dataset with N>=16, including zero variance, and every valid source tuple',
        'executable_exact_domain':'serialized finite rational inputs; includes every finite binary64 dataset',
        'primitive_relations':[
            'mean_j=sum(x_rj)/N, pop_variance_j=sum((x_rj-natural_mean_j)^2)/N>=0',
            'actual std(unbiased=False)=sqrt(pop_variance); CDIR sqrt(max(var,0)) has the same value',
            'if raw_std<floor then floor else raw_std equals clamp_min, including equality and zero variance',
            'source clamped std>=floor>protected-division cutoff, so CDIR div is ordinary division everywhere admitted',
            'center=raw-effective_mean; z=center/effective_std; clip is the same three-case function on both sides'
        ],
        'floor_cases':[{'guard':'raw_std<floor','output':'floor'},{'guard':'raw_std>=floor','output':'raw_std'}],
        'clip_cases':[{'guard':'z<lower','output':str(lower)},{'guard':'lower<=z<=upper','output':'z'},{'guard':'z>upper','output':str(upper)}],
        'protection_margin':str(floor-PROTECTED_FLOOR),'continued_boundary_edges':edges,
        'composition':'Local exact primitive relations imply equality of all 12 program state coordinates and both FX copies under every mask. All prefix-to-continuation edge values agree; unchanged deterministic suffix then gives identical final logits and labels.',
        'numeric_error':'0','classification_labels_equal':True,'collateral_states_checked':'all unpatched mean/std states and both centered/normalized/clipped columns, not only the patched coordinate',
        'branch_coverage':'both floor guards and all three clipping cases are included; runtime trace marks the actually visited conditional child',
        'partial_neural_decompilation':True,'learned_encoder_head_symbolically_recovered':False,
        'device_rounding_certified':False,'mdl_minimality':'unresolved','true_SCM_recovery':'unresolved','original_R5_closed':False}


def verify_normalizer(certificate,checkpoint=None):
    # Independently re-export the actual model; no proposer-supplied graph is
    # accepted as evidence. Primitive/guard/cut conditions are checked separately.
    checkpoint=checkpoint or certificate['checkpoint']
    if str(checkpoint)!=certificate['checkpoint']:raise ValueError('Changed checkpoint reference')
    actual=export_graph(checkpoint)
    if actual!=certificate['fx_export']:raise ValueError('FX graph or checkpoint mismatch')
    program=certificate['program'];validate_program(program)
    branches=normalizer_branches(actual)
    if branches!=certificate['branches'] or program['floor']!=branches[0]['floor'] or program['clip']!=branches[0]['clip']:raise ValueError('Program does not lower actual FX constants/operations')
    if certificate['continued_boundary_edges']!=boundary_edges(actual,branches):raise ValueError('Incomplete continuation cut')
    if certificate['masks']!=[[bool(i&(1<<j)) for j in range(4)] for i in range(16)]:raise ValueError('Omitted compatible intervention combinations')
    if certificate['protection_margin']!=str(Q(program['floor'])-PROTECTED_FLOOR):raise ValueError('Invalid division guard')
    # Remaining theorem fields are a fixed checked proof scheme, not freely
    # chosen metadata that could turn a local proof into full original closure.
    expected=certify_normalizer(checkpoint)
    if certificate!=expected:raise ValueError('Composition proof scheme or scope mismatch')
    return {'status':'verified','conclusion':'proved','compatible_masks':16,'independent_sources':True,
        'mathematical_final_logits_equal':True,'all_degenerate_and_boundary_inputs_included':True,
        'scope':'actual preprocessing prefix plus identical frozen neural continuation','original_R5_closed':False}
