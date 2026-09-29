from copy import deepcopy
from fractions import Fraction as Q
import json,subprocess,sys
import numpy as np
import pytest
from ncd.proof_intervals import Interval
from neural_ssa_proof.compiler import target,compile_target,validate_program
from neural_ssa_proof.verification import certify,verify,hash_value
from neural_ssa_proof.runtime import NumericProgram,interval_program
from neural_ssa_proof.audit import audit

TARGET='runs/oblique_seed1193/teacher.pt'
@pytest.fixture(scope='module')
def proof():
 p=compile_target(target(TARGET));return p,certify(TARGET,p)

def resign(c,p):
 c=deepcopy(c);c['program_sha256']=hash_value(p);return c

def test_entire_actual_network_local_composition(proof):
 p,c=proof;r=verify(c,p)
 assert r['instructions']==37460 and r['mapped_coordinates']==1116 and r['tensor_groups']==43
 assert r['all_compatible_subsets_proved'] and r['full_mathematical_output_fidelity']
 assert c['relations']['numeric_error_interval']==['0','0'] and not c['neural_continuation_retained']
 assert not c['causal_algorithm_compression_proved'] and c['original_claims_closed']==0

def test_coefficient_tamper_rejected_even_with_updated_hash(proof):
 p,c=proof;p=deepcopy(p)
 constant=next(n for n in p['nodes'] if n['op']=='constant' and Q(n['value']) not in (0,Q(1,2),Q(1e-5)))
 constant['value']=str(Q(constant['value'])+Q(1,8))
 with pytest.raises(ValueError):verify(resign(c,p),p)

def test_omitted_actual_group_and_duplicate_sites_rejected(proof):
 p,c=proof;q=deepcopy(p);q['groups'].pop('encoder_1')
 with pytest.raises((ValueError,KeyError)):verify(resign(c,q),q)
 q=deepcopy(p);q['sites']['mean/0']['ref']=q['sites']['mean/1']['ref']
 with pytest.raises(ValueError):verify(resign(c,q),q)

def test_forged_stronger_scope_and_wrong_program_digest(proof):
 p,c=proof;q=deepcopy(c);q['device_rounding_certified']=True
 with pytest.raises(ValueError):verify(q,p)
 q=deepcopy(c);q['program_sha256']='0'*64
 with pytest.raises(ValueError):verify(q,p)

def test_loaded_weight_mismatch(tmp_path):
 import torch
 path=tmp_path/'teacher.pt';path.write_bytes(__import__('pathlib').Path(TARGET).read_bytes())
 p=compile_target(target(str(path)));c=certify(str(path),p)
 state=torch.load(path,weights_only=True);state['state_dict']['head.2.bias'][0]+=.25;torch.save(state,path)
 with pytest.raises(ValueError,match='mismatch'):verify(c,p)

def test_actual_multistep_controls_collateral_and_all_masks(proof):
 p,c=proof;r=audit(TARGET,p)
 assert r['executions']==48 and r['checked_sites_per_execution']==1116 and r['collateral_changes_observed']>0
 assert r['floor_children']==[1,2] and r['clip_cases']==['inside','lower','upper']
 assert r['max_output_absolute_error']<1e-10 and not r['statistical_world_guarantee_claimed']

def test_protected_scale_source_domain_rejected(proof):
 p,c=proof;runtime=NumericProgram(p)
 with pytest.raises(ValueError,match='protected'):runtime.run([[0,0]]*16,{'clamp_min/0':0})
 with pytest.raises(ValueError,match='protection'):interval_program(p,[[0,0]]*16,{'getitem/0':['0','1']})

def test_incompatible_vector_and_unknown_sources_rejected(proof):
 runtime=NumericProgram(proof[0])
 with pytest.raises(ValueError,match='length'):runtime.run([[0,0]]*16,{'encoder_1/0':[0]*15})
 with pytest.raises(ValueError,match='Unknown'):runtime.run([[0,0]]*16,{'invented/0':0})

def test_intervened_branch_not_counted_as_visited(proof):
 p,c=proof;r=NumericProgram(p).run([[0,0]]*16,{'clamp_min/0':1})
 patched=[g for g in r['guards'] if g.get('intervened')]
 assert len(patched)==1 and patched[0]['visited_child'] is None and patched[0]['unvisited_children']==[1,2]

def test_rational_interval_actual_input_strict_label(proof):
 p,c=proof;data=[[Q((i%3)-1,4),Q((i%5)-2,8)] for i in range(16)]
 r=interval_program(p,data);actual=NumericProgram(p).run(data)
 assert r['possible_labels']==[actual['label']] and not r['device_rounding_certified']
 for v,(lo,hi) in zip(actual['logits'],r['logit_enclosures']):assert float(Q(lo))-1e-12<=v<=float(Q(hi))+1e-12

def zero_program():
 t=target(TARGET)
 for m in t['modules'].values():
  if m['operator']=='linear':m['weights']=[['0']*len(w) for w in m['weights']];m['bias']=['0']*len(m['bias'])
 return compile_target(t)

def test_interval_floor_boundary_and_lowest_label_tie():
 p=zero_program();f=Q(1e-5);r=interval_program(p,[[Interval(-f,f),Interval.point(0)]]*16)
 assert any(g.get('branch_boundary_retained') for g in r['guards'])
 assert r['possible_labels']==[0] and r['classification_resolved']
 assert NumericProgram(p).run([[0,0]]*16)['label']==0

def test_runtime_executes_without_torch_import(proof,tmp_path):
 path=tmp_path/'program.json';path.write_text(json.dumps(proof[0]),encoding='utf-8')
 script="import builtins,json; original=builtins.__import__; builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(RuntimeError('Torch used')) if name.startswith('torch') else original(name,*a,**k); from neural_ssa_proof.runtime import NumericProgram; print(NumericProgram(json.load(open(__import__('sys').argv[1],encoding='utf-8'))).run([[0,0]]*16,trace=False)['label'])"
 r=subprocess.run([sys.executable,*(['-I'] if sys.flags.isolated else []),'-c',script,str(path)],capture_output=True,text=True,timeout=30)
 assert r.returncode==0,r.stderr
