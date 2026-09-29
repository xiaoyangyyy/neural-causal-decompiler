"""Diagnostic interventions on the actual FX graph, separate from real proof."""
import hashlib,json
from fractions import Fraction as Q
import numpy as np
from .realization import Algebraic,execute_exact,export_graph,normalizer_branches,verify_normalizer,key


def number(value):
    return Algebraic(value['numerator'],value['denominator_squared']).approximate()


def fixtures(program):
    f=Q(program['floor'])
    alternate=lambda a,b:[[a if i%2 else -a,b if i%2 else -b] for i in range(16)]
    sources=[[[3,-4]]*16,[[9,-2]]*16,alternate(Q(1,4),4),alternate(4,2)]
    zero=[[[0,0]]*16 for _ in range(4)]
    rng=np.random.default_rng(8100)
    return [
        ('zero_variance',[[0,1]]*16,sources),
        ('floor_equality',alternate(f,f),zero),
        ('clip_edges',[[(-20*f if i%3==0 else 20*f if i%3==1 else 0),i-8] for i in range(16)],zero),
        ('large_normalized',[[Q(10)**100,0]]*16,zero),
        ('independent_source_shifts',alternate(2,3),sources),
        ('development_8100',rng.normal(size=(96,2)).tolist(),sources),
    ]


def audit_actual_interventions(certificate,checkpoint=None):
    import torch
    from ncd.model import load_model
    from ncd.io import digest
    checkpoint=checkpoint or certificate['checkpoint'];verify_normalizer(certificate,checkpoint)
    before=digest(checkpoint);torch.set_num_threads(2)
    graph=torch.fx.symbolic_trace(load_model(checkpoint).double());branches=normalizer_branches(export_graph(checkpoint))
    program=certificate['program']
    class Recorder(torch.fx.Interpreter):
        def __init__(self,replacements=None):
            super().__init__(graph);self.values={};self.replacements=replacements or {}
        def run_node(self,node):
            value=super().run_node(node)
            if node.name in self.replacements:value=self.replacements[node.name](value)
            if isinstance(value,torch.Tensor):self.values[node.name]=value.detach().clone()
            return value
    def tensor(rows):return torch.tensor([[float(Q(v)) for v in row] for row in rows],dtype=torch.float64).unsqueeze(0)
    def replacement(position,value):
        def apply(original):
            changed=original.clone();changed[...,position]=value;return changed
        return apply
    errors=[];visited_floor=set();visited_clip=set()
    with torch.no_grad():
        for name,data,sources in fixtures(program):
            input_tensor=tensor(data);source_runs=[]
            for rows in sources:
                r=Recorder();r.run(tensor(rows));source_runs.append(r)
            for mask in certificate['masks']:
                exact=execute_exact(program,data,sources,mask);patches={}
                for branch in branches:
                    for kind,offset in [('mean',0),('denominator',2)]:
                        selected=[(pos,col) for pos,col in enumerate(branch['columns']) if mask[col+offset]]
                        if selected:
                            node_name=branch['nodes'][kind]
                            def apply(original,selected=selected,node_name=node_name,offset=offset):
                                changed=original.clone()
                                for pos,col in selected:changed[...,pos]=source_runs[col+offset].values[node_name][...,pos]
                                return changed
                            patches[node_name]=apply
                actual=Recorder(patches);actual_output=actual.run(input_tensor)
                hybrid_patches={}
                for branch in branches:
                    den=torch.tensor([number(exact['states']['std_'+'xy'[col]]) for col in branch['columns']],dtype=torch.float64).reshape(1,1,2)
                    z=torch.tensor([[number(row[col]) for col in branch['columns']] for row in exact['outputs']],dtype=torch.float64).unsqueeze(0)
                    hybrid_patches[branch['nodes']['denominator']]=lambda original,den=den:den
                    hybrid_patches[branch['nodes']['clipped']]=lambda original,z=z:z
                hybrid=Recorder(hybrid_patches);hybrid_output=hybrid.run(input_tensor)
                state_error=0.0
                for branch in branches:
                    for kind,state_name in [('mean','mean'),('std','std_raw'),('denominator','std'),('centered','centered'),('normalized','normalized'),('clipped','clipped')]:
                        values=actual.values[branch['nodes'][kind]].numpy()[0]
                        for pos,col in enumerate(branch['columns']):
                            ideal=exact['states'][state_name+'_'+'xy'[col]]
                            expected=np.array([number(v) for v in ideal] if isinstance(ideal,list) else [number(ideal)])
                            observed=values[:,pos]
                            relative=np.max(np.abs(observed-expected)/np.maximum(1.0,np.abs(expected)))
                            state_error=max(state_error,float(relative))
                            if not np.allclose(observed,expected,rtol=1e-10,atol=1e-9):raise ValueError('Actual FX state mismatch: '+name+'/'+kind)
                final_error=float(torch.max(torch.abs(actual_output-hybrid_output)))
                if not torch.allclose(actual_output,hybrid_output,rtol=1e-10,atol=1e-9):raise ValueError('Frozen continuation mismatch')
                if actual_output.argmax(-1).tolist()!=hybrid_output.argmax(-1).tolist():raise ValueError('Diagnostic label mismatch')
                for guard in exact['guards']:
                    visited_floor.add(guard['natural_std_case']);visited_clip.update(guard['clipping_cases'])
                errors.append({'fixture':name,'mask':mask,'state_relative_error':state_error,'final_absolute_error':final_error,
                    'output_sha256':hashlib.sha256(actual_output.numpy().tobytes()).hexdigest(),'labels_equal':True})
    if digest(checkpoint)!=before:raise ValueError('Diagnostic mutated frozen weights')
    if visited_floor!={'floor','std'} or visited_clip!={'lower','inside','upper'}:raise ValueError('Missing diagnostic branch visit')
    return {'schema':'ncd.actual-normalizer-intervention-audit.v1','status':'passed','checkpoint_sha256':before,
        'fixture_count':6,'masks_per_fixture':16,'executions':len(errors),'independent_sources':True,
        'checked_states_per_branch':12,'actual_branches':2,'floor_cases':sorted(visited_floor),'clip_cases':sorted(visited_clip),
        'max_state_relative_error':max(r['state_relative_error'] for r in errors),
        'max_final_absolute_error':max(r['final_absolute_error'] for r in errors),'executions_detail':errors,
        'dtype':'float64 diagnostic with original frozen parameter values','mathematical_theorem_is_separate':True,
        'device_rounding_globally_certified':False,'whole_learned_algorithm_recovered':False}
