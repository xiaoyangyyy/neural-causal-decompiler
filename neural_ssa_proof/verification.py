"""Independent FX-to-program local relation checker and composition proof."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib,json
from ncd.io import digest
from .compiler import target,compile_target,validate_program,key


def hash_value(value):return hashlib.sha256(key(value).encode()).hexdigest()

def relations(t,p):
    validate_program(p);nodes=p['nodes'];groups=p['groups'];covered=set();records=[];positive={}
    fx=t['fx']['nodes'];by_name={n['name']:n for n in fx};constant=lambda r:Q(nodes[r]['value']) if nodes[r]['op']=='constant' else None
    def tensor(arg):
        if set(arg)!={'node'} or arg['node'] not in groups:raise ValueError('Missing actual FX tensor mapping')
        return groups[arg['node']]['refs']
    def check_unary(ref,op,arg):
        covered.add(ref)
        if nodes[ref]['op']!=op or nodes[ref]['args']!=[arg]:raise ValueError('Unary local relation mismatch')
    def check_binary(ref,op,a,b):
        covered.add(ref)
        if nodes[ref]['op']!=op or nodes[ref]['args']!=[a,b]:raise ValueError('Binary local relation mismatch')
    def affine(ref,inputs):
        symbols={v:i for i,v in enumerate(inputs)}
        def add(a,b):
            c=dict(a)
            for k,v in b.items():c[k]=c.get(k,Q(0))+v
            return {k:v for k,v in c.items() if v}
        def scale(a,value):return {k:v*value for k,v in a.items() if v*value}
        def walk(r):
            if r in symbols:return {symbols[r]:Q(1)}
            covered.add(r);n=nodes[r]
            if n['op']=='constant':return {} if Q(n['value'])==0 else {-1:Q(n['value'])}
            if n['op']=='add':return add(walk(n['args'][0]),walk(n['args'][1]))
            if n['op']=='sub':return add(walk(n['args'][0]),scale(walk(n['args'][1]),-1))
            if n['op']=='mul':
                a,b=n['args'];ca,cb=constant(a),constant(b)
                if ca is not None:covered.add(a);return scale(walk(b),ca)
                if cb is not None:covered.add(b);return scale(walk(a),cb)
            raise ValueError('Non-affine hidden computation in affine relation')
        return walk(ref)
    def copy_relation(ref,incoming):
        if affine(ref,[incoming])!={0:Q(1)}:raise ValueError('Copy/reindex local relation mismatch')
    expected_groups={n['name'] for n in fx if n['op']!='output'}
    if set(groups)!=expected_groups:raise ValueError('Actual tensor group omitted or invented')
    for row in fx:
        name,op,fn,args=row['name'],row['op'],row['target'],row['args']
        if op=='output':
            if p['logits']!=tensor(args[0]):raise ValueError('Program logits are not actual output')
            continue
        refs=groups[name]['refs'];identity=[]
        actual_count=2 if op=='placeholder' else None
        expected_layout='B_N_C' if op=='placeholder' else None
        if op=='placeholder':
            if len(refs)!=2:raise ValueError('Input dimensions')
            for j,r in enumerate(refs):
                covered.add(r)
                if nodes[r]['op']!='var' or nodes[r]['index']!=j:raise ValueError('Raw column read relation')
            identity=['raw inputs are the identical two sample vectors']
        elif op=='call_module':
            incoming=tensor(args[0]);module=t['modules'][fn]
            actual_count=len(module['bias']) if module['operator']=='linear' else len(incoming)
            expected_layout='B_N_C' if groups[args[0]['node']]['kind']=='vector' else 'B_C'
            if module['operator']=='linear':
                if len(refs)!=len(module['bias']):raise ValueError('Affine output width')
                for r,weights,bias in zip(refs,module['weights'],module['bias']):
                    expected={j:Q(v) for j,v in enumerate(weights) if Q(v)}
                    if Q(bias):expected[-1]=Q(bias)
                    if affine(r,incoming)!=expected:raise ValueError('Actual frozen affine coefficients differ')
                identity=['exact rational coefficient normal forms equal; arbitrary real input coordinates']
            else:
                for r,a in zip(refs,incoming):check_unary(r,'tanh',a)
                identity=['identical mathematical Tanh by congruence on equal arguments']
        elif op=='call_method':
            incoming=tensor(args[0]);actual_count=len(incoming);expected_layout=groups[args[0]['node']]['layout']
            if fn in ('std','mean'):expected_layout='B_1_C' if row['kwargs'].get('keepdim',False) else 'B_C'
            if fn=='std':
                for r,a in zip(refs,incoming):
                    if nodes[r]['op']!='sqrt':raise ValueError('Population std square root missing')
                    covered.add(r);variance=nodes[r]['args'][0];check_unary(variance,'var_stat',a)
                identity=['same population variance independent of a separately patched mean; nonnegative variance makes sqrt protection inactive']
            elif fn=='clamp_min':
                f=Q(args[1])
                for j,(r,a) in enumerate(zip(refs,incoming)):
                    n=nodes[r];covered.add(r)
                    if n['op']!='if' or n['args'][2]!=a or constant(n['args'][1])!=f:raise ValueError('Floor branch/value mismatch')
                    covered.add(n['args'][1]);check_binary(n['args'][0],'lt',a,n['args'][1]);positive[name+'/'+str(j)]=str(f)
                identity=['both <floor and >=floor cases; zero variance and equality retained']
            elif fn=='clamp':
                for r,a in zip(refs,incoming):
                    check_unary(r,'clip',a)
                    if [Q(nodes[r]['lower']),Q(nodes[r]['upper'])]!=list(map(Q,args[1:])):raise ValueError('Clipping boundaries changed')
                identity=['same three-case clip, including both boundaries']
            elif fn=='flip':
                for r,a in zip(refs,incoming[::-1]):copy_relation(r,a)
                identity=['coordinate reversal through independent copy variables']
            else:
                for r,a in zip(refs,incoming):check_unary(r,fn,a)
                if fn=='log':
                    for a in incoming:
                        sid=next((sid for sid,s in p['sites'].items() if s['ref']==a),None)
                        if sid not in positive or Q(positive[sid])<=Q(p['protected_log_cutoff']):raise ValueError('Ordinary log domain not certified')
                    identity=['positive std input exceeds protected-log cutoff, including compatible source writes']
                else:identity=['same '+fn+' function and reduction axes under real semantics']
        elif op=='call_function':
            if fn=='torch.cat':
                expected_layout='B_C'
                incoming=[a for tensor_arg in args[0] for a in tensor(tensor_arg)]
                actual_count=len(incoming)
                for r,a in zip(refs,incoming):copy_relation(r,a)
                identity=['all concatenated coordinates have distinct writable copy variables']
            elif fn=='_operator.getitem':
                incoming=tensor(args[0]);index=args[1][1]
                selected=incoming if index==0 else [incoming[j] for j in index];actual_count=len(selected);expected_layout='B_C'
                for j,(r,a) in enumerate(zip(refs,selected)):
                    copy_relation(r,a)
                    if index==0:
                        sid=next((sid for sid,s in p['sites'].items() if s['ref']==a),None)
                        if sid not in positive:raise ValueError('Scale alias lacks positive premise')
                        positive[name+'/'+str(j)]=positive[sid]
                identity=['same slice/reindex, with distinct intervention variables']
            elif fn=='_operator.mul':
                incoming=tensor(args[1]);factor=Q(args[0]);actual_count=len(incoming);expected_layout=groups[args[1]['node']]['layout']
                for r,a in zip(refs,incoming):
                    if affine(r,[a])!={0:factor}:raise ValueError('Final average coefficient mismatch')
                identity=['same exact averaging coefficient']
            else:
                incoming_a,incoming_b=tensor(args[0]),tensor(args[1]);actual_count=len(incoming_a)
                if len(incoming_a)!=len(incoming_b):raise ValueError('Actual broadcast relation changed')
                expected_layout=groups[args[0]['node']]['layout'] if groups[args[0]['node']]['kind']=='vector' else groups[args[1]['node']]['layout']
                operator={'_operator.add':'add','_operator.sub':'sub','_operator.truediv':'div'}[fn]
                for r,a,b in zip(refs,incoming_a,incoming_b):
                    check_binary(r,operator,a,b)
                    if operator=='div':
                        sid=next((sid for sid,s in p['sites'].items() if s['ref']==b),None)
                        if sid not in positive or Q(positive[sid])<=Q(p['protected_division_cutoff']):raise ValueError('Protected division can differ from neural division')
                identity=['same '+operator+'; division uses certified positive scale above protection cutoff' if operator=='div' else 'same '+operator+' on matched inputs']
        else:raise ValueError('Unknown FX relation')
        if len(refs)!=actual_count or groups[name]['layout']!=expected_layout or any(nodes[r]['kind']!=('vector' if expected_layout=='B_N_C' else 'scalar') for r in refs):raise ValueError('Actual tensor shape/kind mapping changed')
        records.append({'fx_node':name,'coordinates':len(refs),'local_error_interval':['0','0'],'identities':identity})
    if positive!=p['positive_intervention_sites']:raise ValueError('Unsupported source-domain premise')
    covered.add(p['label'])
    if covered!=set(range(len(nodes))):raise ValueError('SSA primitive lacks a checked local relation')
    return {'relations':records,'covered_instructions':len(covered),'covered_tensor_groups':len(groups),
        'covered_writable_coordinates':len(p['sites']),'positive_source_bounds':positive,
        'composition':'Topological induction: an unpatched variable preserves its checked local relation; a patched variable receives the same independently supplied compatible source value on both sides. All mapped intermediates, collateral states, final logits and the common tie-resolving label are equal.',
        'compatible_masks':'every subset of the frozen mapped-site set, without enumerating or assuming a common source',
        'numeric_error_interval':['0','0'],'classification_boundaries_and_ties_included':True}


def certify(checkpoint,program=None):
    t=target(checkpoint);p=compile_target(t) if program is None else program;r=relations(t,p)
    return {'schema':'ncd.full-discoverer-realization.v1','status':'proved','checkpoint':str(checkpoint),
        'target_sha256':hash_value(t),'program_sha256':hash_value(p),'relations':r,
        'domain':'every finite real N x 2 dataset, N>=16; finite compatible source values with matching vector lengths and the explicit positive scale bounds',
        'sources':'each selected coordinate may use an independently chosen source execution; scalar sources may have different sample counts',
        'entire_loaded_neural_computation_realized':True,'neural_continuation_retained':False,'opaque_neural_calls':0,
        'representation':'scalar/vector primitive SSA; the learned weights remain explicit numerical coefficients',
        'causal_algorithm_compression_proved':False,'true_SCM_correctness':'unresolved','mdl_minimality':'unresolved',
        'device_rounding_certified':False,'original_claims_closed':0,'original_objective_achieved':False}


def verify(certificate,program,checkpoint=None):
    checkpoint=checkpoint or certificate['checkpoint']
    if str(checkpoint)!=certificate['checkpoint']:raise ValueError('Changed target reference')
    t=target(checkpoint)
    if hash_value(t)!=certificate['target_sha256'] or hash_value(program)!=certificate['program_sha256']:raise ValueError('Frozen weight/graph or program mismatch')
    r=relations(t,program)
    # Check theorem scope as well as local coefficients; no proposer-written
    # completion flag or stronger source/domain statement is accepted.
    expected=certify(checkpoint,program)
    if certificate!=expected:raise ValueError('Local proof, composition or scope mismatch')
    return {'status':'verified','conclusion':'proved','instructions':r['covered_instructions'],
        'mapped_coordinates':r['covered_writable_coordinates'],'tensor_groups':r['covered_tensor_groups'],
        'full_mathematical_output_fidelity':True,'all_compatible_subsets_proved':True,
        'original_objective_achieved':False,'device_rounding_certified':False}
