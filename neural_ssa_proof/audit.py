"""Actual FX read/write diagnostics for a fixed map, separate from the theorem."""
from fractions import Fraction as Q
import hashlib,itertools
import numpy as np
from .compiler import target,compile_target,key
from .runtime import NumericProgram
from .verification import certify,verify
from ncd.io import digest

CONTROLS=('mean/0','encoder_1/0','mean_5/2','head_1/1')

def audit(checkpoint,program):
    import torch
    from ncd.model import load_model
    certificate=certify(checkpoint,program);verify(certificate,program);before=digest(checkpoint)
    torch.set_num_threads(2);graph=torch.fx.symbolic_trace(load_model(checkpoint).double())
    numerical=NumericProgram(program);site_groups={}
    for sid,s in program['sites'].items():site_groups.setdefault(s['fx_node'],[]).append((sid,s))
    class Recorder(torch.fx.Interpreter):
        def __init__(self,patches=None):super().__init__(graph);self.patches=patches or {};self.states={}
        def run_node(self,node):
            value=super().run_node(node)
            if node.name in site_groups:
                group=site_groups[node.name]
                selected=[(sid,s) for sid,s in group if sid in self.patches]
                if selected:
                    value=value.clone()
                    for sid,s in selected:value[...,s['column']]=torch.as_tensor(self.patches[sid],dtype=value.dtype)
                for sid,s in group:
                    v=value[0,...,s['column']].detach().numpy()
                    self.states[sid]=v.copy() if s['kind']=='vector' else float(v.reshape(-1)[0])
            return value
    rng=np.random.default_rng(8100);data=[[[0,1]]*16,[[10,-1]]*16,rng.normal(size=(32,2)).tolist()]
    records=[];changed=0;floor_children=set();clip_cases=set()
    with torch.no_grad():
        for case,base in enumerate(data):
            n=len(base);sources=[[[3 if case==0 else -3,2]]*16,rng.normal(-.7,2,size=(n,2)).tolist(),rng.normal(1,1.5,size=(24,2)).tolist(),rng.normal(-2,3,size=(40,2)).tolist()]
            source_runs=[]
            for source in sources:
                r=Recorder();r.run(torch.tensor(source,dtype=torch.float64).unsqueeze(0));source_runs.append(r)
            base_tensor=torch.tensor(base,dtype=torch.float64).unsqueeze(0)
            natural=Recorder();natural.run(base_tensor)
            for mask in itertools.product([False,True],repeat=4):
                patches={sid:source_runs[j].states[sid] for j,sid in enumerate(CONTROLS) if mask[j]}
                r=Recorder(patches);logits=r.run(base_tensor).detach().numpy()[0]
                actual=numerical.run(base,patches);maximum=0.0
                for sid,expected in r.states.items():
                    value=actual['states'][sid]
                    error=float(np.max(np.abs(np.asarray(value)-expected)/np.maximum(1,np.abs(expected))))
                    maximum=max(maximum,error)
                    if not np.allclose(value,expected,rtol=1e-10,atol=1e-9):raise ValueError('Actual intermediate mismatch '+sid)
                    if sid not in patches and not np.allclose(expected,natural.states[sid],rtol=1e-12,atol=1e-12):changed+=1
                output_error=float(np.max(np.abs(logits-actual['logits'])))
                if not np.allclose(logits,actual['logits'],rtol=1e-10,atol=1e-9) or int(np.argmax(logits))!=actual['label']:raise ValueError('Actual final output mismatch')
                floor_children.update(g['visited_child'] for g in actual['guards'] if g.get('visited_child') is not None)
                for sid,s in program['sites'].items():
                    if s['fx_node'] in ('clamp','clamp_1'):
                        z=actual['states'][('truediv' if s['fx_node']=='clamp' else 'truediv_1')+'/'+str(s['column'])]
                        clip_cases.update('lower' if v<-20 else 'upper' if v>20 else 'inside' for v in z)
                records.append({'fixture':case,'mask':list(mask),'checked_intermediates':len(r.states),'max_state_relative_error':maximum,
                    'max_output_absolute_error':output_error,'labels_equal':True,'actual_logits_sha256':hashlib.sha256(logits.tobytes()).hexdigest()})
    if digest(checkpoint)!=before or floor_children!={1,2} or clip_cases!={'lower','inside','upper'} or not changed:raise ValueError('Weights/branch/collateral coverage incomplete')
    return {'schema':'ncd.full-discoverer-ssa-diagnostic.v1','status':'passed','controls':list(CONTROLS),'independent_sources':True,
        'source_scalar_counts':[16,24,40],'vector_source_count_matches_base':True,'executions':len(records),
        'checked_sites_per_execution':len(program['sites']),'collateral_changes_observed':changed,
        'max_output_absolute_error':max(r['max_output_absolute_error'] for r in records),'max_state_relative_error':max(r['max_state_relative_error'] for r in records),
        'floor_children':sorted(floor_children),'clip_cases':sorted(clip_cases),'records':records,
        'statistical_world_guarantee_claimed':False,'device_rounding_globally_certified':False}
