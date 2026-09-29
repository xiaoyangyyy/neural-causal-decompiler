"""Lower the actual Discoverer FX graph to typed scalar/vector CDIR-style SSA."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json
from ncd.io import digest
from normalizer_proof.realization import export_graph,MODEL_SOURCE

def key(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def target(checkpoint):
    import torch
    from ncd.model import load_model
    fx=export_graph(checkpoint);model=load_model(checkpoint)
    if type(model.width)is not int or not 1<=model.width<=256:raise ValueError('Unsupported target width')
    modules={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            if module.bias is None or not bool(torch.isfinite(module.weight).all()) or not bool(torch.isfinite(module.bias).all()):raise ValueError('Non-real or unsupported affine parameters')
            modules[name]={'operator':'linear','weights':[[str(Q(float(v))) for v in row] for row in module.weight.detach().tolist()],
                'bias':[str(Q(float(v))) for v in module.bias.detach().tolist()]}
        elif isinstance(module,torch.nn.Tanh):modules[name]={'operator':'tanh'}
    return {'schema':'ncd.full-discoverer-target.v1','checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
        'fx':fx,'modules':modules,'parameter_semantics':'exact real values of the frozen parameters loaded by the source-bound ncd.model.load_model'}


def compile_target(t,max_nodes=100000):
    rows=t['fx']['nodes'];nodes=[];constants={};groups={};sites={};positive_sites={}
    def emit(op,args=(),kind=None,**fields):
        if len(nodes)>=max_nodes:raise RuntimeError('SSA instruction budget exhausted')
        if kind is None:kind='vector' if any(nodes[a]['kind']=='vector' for a in args) else 'scalar'
        i=len(nodes);nodes.append({'id':i,'op':op,'args':list(args),'kind':kind,**fields});return i
    def const(value):
        value=str(Q(value))
        if value not in constants:constants[value]=emit('constant',value=value,kind='scalar')
        return constants[value]
    def copy(ref):return emit('add',(ref,const(0)))
    def group(name,refs,layout):
        if name in groups:raise ValueError('Duplicate FX group')
        groups[name]={'refs':refs,'layout':layout,'kind':nodes[refs[0]]['kind']}
        for j,ref in enumerate(refs):
            sid=name+'/'+str(j);sites[sid]={'ref':ref,'fx_node':name,'column':j,'kind':nodes[ref]['kind'],'layout':layout}
        return groups[name]
    def source(argument):
        if not isinstance(argument,dict) or set(argument)!={'node'} or argument['node'] not in groups:raise ValueError('Unverified FX tensor dependency')
        return groups[argument['node']]
    def tensor_op(op,a,b):
        if len(a['refs'])!=len(b['refs']):raise ValueError('Unsupported channel broadcasting')
        layout=a['layout'] if a['kind']=='vector' else b['layout']
        return [emit(op,(x,y)) for x,y in zip(a['refs'],b['refs'])],layout
    floor=Q(1e-5)
    for row in rows:
        name,op,fn,args,kw=row['name'],row['op'],row['target'],row['args'],row['kwargs']
        if op=='placeholder':
            if fn!='x' or args or kw:raise ValueError('Unsupported raw input')
            group(name,[emit('var',kind='vector',index=j) for j in range(2)],'B_N_C');continue
        if op=='output':
            out=source(args[0])
            if len(out['refs'])!=4 or out['kind']!='scalar':raise ValueError('Four scalar logits required')
            logits=out['refs'];continue
        if op=='call_module':
            incoming=source(args[0]);module=t['modules'].get(fn)
            if module is None or len(args)!=1 or kw:raise ValueError('Unsupported learned module')
            if module['operator']=='linear':
                refs=[];weights,bias=module['weights'],module['bias']
                if len(weights)!=len(bias) or any(len(w)!=len(incoming['refs']) for w in weights):raise ValueError('Affine dimensions')
                for w,b in zip(weights,bias):
                    result=const(b)
                    for coefficient,x in zip(w,incoming['refs']):result=emit('add',(result,emit('mul',(const(coefficient),x))))
                    refs.append(result)
            else:refs=[emit('tanh',(x,)) for x in incoming['refs']]
            group(name,refs,'B_N_C' if incoming['kind']=='vector' else 'B_C');continue
        if op=='call_method':
            incoming=source(args[0]);refs=incoming['refs'];layout=incoming['layout']
            if fn in ('mean','std'):
                dim=args[1] if len(args)>1 else kw.get('dim')
                if incoming['kind']!='vector' or dim!=1:raise ValueError('Unsupported reduction axis')
                layout='B_1_C' if kw.get('keepdim',False) else 'B_C'
                if fn=='std':
                    if kw.get('unbiased') is not False:raise ValueError('Sample standard deviation unsupported')
                    refs=[emit('sqrt',(emit('var_stat',(x,),kind='scalar'),),kind='scalar') for x in refs]
                else:refs=[emit('mean',(x,),kind='scalar') for x in refs]
            elif fn=='clamp_min':
                if len(args)!=2 or Q(args[1])!=floor or incoming['kind']!='scalar':raise ValueError('Unsupported clamp floor')
                f=const(floor);refs=[emit('if',(emit('lt',(x,f),kind='bool'),f,x),kind='scalar') for x in refs]
                for j in range(len(refs)):positive_sites[name+'/'+str(j)]=str(floor)
            elif fn=='clamp':
                if len(args)!=3 or args[1:]!=[-20,20]:raise ValueError('Unsupported clipping bounds')
                refs=[emit('clip',(x,),lower='-20',upper='20') for x in refs]
            elif fn in ('square','log'):
                refs=[emit(fn,(x,)) for x in refs]
            elif fn=='flip':
                if args[1:]!=[-1] or kw or len(refs)!=2:raise ValueError('Unsupported variable flip')
                refs=[copy(x) for x in refs[::-1]]
            else:raise ValueError('Unsupported FX method '+fn)
            group(name,refs,layout);continue
        if op=='call_function':
            if fn in ('_operator.sub','_operator.add','_operator.truediv'):
                a,b=source(args[0]),source(args[1]);refs,layout=tensor_op({'_operator.sub':'sub','_operator.add':'add','_operator.truediv':'div'}[fn],a,b)
            elif fn=='_operator.mul':
                if len(args)!=2 or Q(args[0])!=Q(1,2):raise ValueError('Unsupported multiplier')
                incoming=source(args[1]);refs=[emit('mul',(const(args[0]),x)) for x in incoming['refs']];layout=incoming['layout']
            elif fn=='_operator.getitem':
                incoming=source(args[0]);index=args[1]
                if index[0]!={'repr':'slice(None, None, None)'}:raise ValueError('Unsupported batch slice')
                if index[1]==0 and incoming['layout']=='B_1_C':
                    refs=[copy(x) for x in incoming['refs']]
                    for j in range(len(refs)):positive_sites[name+'/'+str(j)]=str(floor)
                elif index[1]==[1,0,2,3] and len(incoming['refs'])==4:refs=[copy(incoming['refs'][j]) for j in index[1]]
                else:raise ValueError('Unsupported tensor indexing')
                layout='B_C'
            elif fn=='torch.cat':
                incoming=[source(a) for a in args[0]]
                if kw!={'dim':1} or any(a['layout']!='B_C' for a in incoming):raise ValueError('Unsupported concatenate shape')
                refs=[copy(x) for a in incoming for x in a['refs']];layout='B_C'
            else:raise ValueError('Unsupported FX function '+fn)
            group(name,refs,layout);continue
        raise ValueError('Unsupported FX node '+op)
    label=emit('argmax',logits,kind='label')
    return {'schema':'ncd.full-discoverer-cdir-ssa.v1','minimum_samples':16,'columns':2,
        'nodes':nodes,'groups':groups,'sites':sites,'positive_intervention_sites':positive_sites,
        'logits':logits,'label':label,'protected_division_cutoff':str(Q(1e-8)),'protected_log_cutoff':str(Q(1e-12)),
        'clip_intrinsic':'coordinatewise max(lower,min(upper,x))','label_tie_rule':'lowest index attaining the maximum',
        'opaque_neural_operations':0,'mdl_status':'upper bound only; no shorter-candidate search or semantic minimality proof'}


def validate_program(p):
    if p['schema']!='ncd.full-discoverer-cdir-ssa.v1' or p['minimum_samples']!=16 or p['columns']!=2:raise ValueError('Program input signature')
    if p['protected_division_cutoff']!=str(Q(1e-8)) or p['protected_log_cutoff']!=str(Q(1e-12)):raise ValueError('Changed protected semantics')
    if p['clip_intrinsic']!='coordinatewise max(lower,min(upper,x))':raise ValueError('Changed clip semantics')
    arities={'var':0,'constant':0,'add':2,'sub':2,'mul':2,'div':2,'tanh':1,'square':1,'mean':1,'var_stat':1,'sqrt':1,'log':1,'lt':2,'if':3,'clip':1,'argmax':4}
    nodes=p['nodes']
    for i,n in enumerate(nodes):
        if type(n['id'])is not int or n['id']!=i or n['op'] not in arities or len(n['args'])!=arities[n['op']] or any(type(a)is not int or not 0<=a<i for a in n['args']):raise ValueError('Cyclic, dangling or unsupported SSA instruction')
        op=n['op'];args=[nodes[a] for a in n['args']];numeric={'scalar','vector'}
        expected='vector' if any(a['kind']=='vector' for a in args) else 'scalar'
        if op=='var':
            if type(n['index'])is not int or n['index'] not in (0,1):raise ValueError('Raw variable index')
            expected='vector'
        elif op=='constant':
            q=Q(n['value'])
            if Q(float(q))!=q:raise ValueError('Non-binary64 frozen constant')
        elif op in ('mean','var_stat'):
            if args[0]['kind']!='vector':raise ValueError('Reduction requires vector')
            expected='scalar'
        elif op=='lt':
            if any(a['kind']!='scalar' for a in args):raise ValueError('Scalar guard required')
            expected='bool'
        elif op=='if':
            if args[0]['kind']!='bool' or args[1]['kind']!=args[2]['kind']:raise ValueError('Conditional type mismatch')
            guard=args[0]
            if guard['op']!='lt' or guard['args']!=[n['args'][2],n['args'][1]] or args[1]['op']!='constant':raise ValueError('Unsupported conditional refinement')
            expected=args[1]['kind']
        elif op=='argmax':
            if any(a['kind']!='scalar' for a in args):raise ValueError('Scalar logits required')
            expected='label'
        elif op=='clip':
            if Q(n['lower'])!=-20 or Q(n['upper'])!=20:raise ValueError('Unsupported clip constants')
        elif any(a['kind'] not in numeric for a in args):raise ValueError('Numeric operands required')
        if n['kind']!=expected:raise ValueError('SSA type mismatch')
    if len(p['logits'])!=4 or any(type(r)is not int or not 0<=r<len(nodes) for r in p['logits']) or type(p['label'])is not int or not 0<=p['label']<len(nodes):raise ValueError('Output reference')
    refs=[s['ref'] for s in p['sites'].values()]
    if len(refs)!=len(set(refs)) or any(not 0<=r<len(nodes) for r in refs):raise ValueError('Intervention site alias/dangling reference')
    for sid,s in p['sites'].items():
        if sid!=s['fx_node']+'/'+str(s['column']) or s['kind']!=nodes[s['ref']]['kind'] or p['groups'][s['fx_node']]['refs'][s['column']]!=s['ref']:raise ValueError('Site/variable relation mismatch')
    expected_sites=set()
    for name,g in p['groups'].items():
        if not g['refs'] or g['layout'] not in ('B_N_C','B_1_C','B_C'):raise ValueError('Tensor group signature')
        kind='vector' if g['layout']=='B_N_C' else 'scalar'
        if g['kind']!=kind or any(type(r)is not int or not 0<=r<len(nodes) or nodes[r]['kind']!=kind for r in g['refs']):raise ValueError('Group kind/reference mismatch')
        expected_sites.update(name+'/'+str(j) for j in range(len(g['refs'])))
    if set(p['sites'])!=expected_sites:raise ValueError('Incomplete writable coordinate coverage')
    if not set(p['positive_intervention_sites'])<=expected_sites or any(Q(v)<=0 for v in p['positive_intervention_sites'].values()):raise ValueError('Invalid source-domain premise')
    if p['opaque_neural_operations']!=0 or p['label_tie_rule']!='lowest index attaining the maximum':raise ValueError('Opaque or changed output semantics')
    if any(nodes[r]['kind']!='scalar' for r in p['logits']) or nodes[p['label']]['args']!=p['logits']:raise ValueError('Output identity')
    return True
