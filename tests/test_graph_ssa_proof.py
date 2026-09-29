"""Actual graph fidelity, physical interchange, and hostile certificate checks."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,shutil
import numpy as np
import pytest
import torch
from graph_ssa_proof.compiler import compile_graph
from graph_ssa_proof.verification import certify,verify,hash_value
from graph_ssa_proof.binding import verify_checkpoint,load_primary_model,APPROVED_PRIMARY_SOURCES
from graph_ssa_proof.runtime import ExplicitGraphProgram,validate_program
from graph_ssa_proof.mapping import SITES,run_primary,validate_primary_mapping
from graph_ssa_proof.decoder import graph_tail,PRIMARY_FUNCTION_TEXT
from graph_ssa_proof import __main__ as cli
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'validation/actual_graph_program_targets_v1/manifest.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@pytest.fixture(scope='module')
def targets():
 torch.set_num_threads(2)
 return read(TARGET)['records']
@pytest.fixture(scope='module')
def actual(targets):
 result=[]
 for row in targets:
  net=read(ROOT/row['export']);model=load_primary_model(ROOT/row['checkpoint'],ROOT/'ncd').double()
  result.append((row,net,model))
 return result
@pytest.fixture(scope='module')
def first(actual):return actual[0][1],compile_graph(actual[0][1],3)

@pytest.mark.parametrize('index',range(4))
def test_actual_checkpoint_coefficients_and_graph_source(index,targets):
 row=targets[index];r=verify_checkpoint(read(ROOT/row['export']),ROOT/row['checkpoint'],ROOT/'ncd')
 assert r['actual_state_dict_read'] and r['actual_tensors_bound']==22 and not r['neural_inference_used_to_prove_fidelity']
@pytest.mark.parametrize('index,n',[(i,n) for i in range(4) for n in (3,5,8)])
def test_full_graph_domains_and_physical_sites(index,n,actual):
 _,network,model=actual[index];program=compile_graph(network,n);certificate=certify(network,program)
 result=verify(network,program,certificate);physical=validate_primary_mapping(model,program)
 assert result['checked_instructions']==138 and result['all_finite_feature_coordinates_covered']
 assert result['mathematical_error']=='0' and result['all_boundaries_retained']
 assert len(physical['sites'])==14 and result['all_compatible_declared_site_subsets_proved_mathematically']
 assert not result['original_objective_achieved'] and not result['hardware_rounding_covered']

TAMPERS=[('normalization.std','constant'),('normalization.clipped','clip'),('context.outgoing.selected','take'),('context.incoming.selected','take'),('context.global.selected','take'),('context.concatenated','concat'),('context.source.outgoing.reshape','reshape'),('attention.query.selected','take'),('attention.key.selected','take'),('attention.value.allowed','take'),('attention.key.transpose','transpose'),('attention.head_width','constant'),('attention.scores','real_div'),('attention.peak','max'),('attention.denominator','sum'),('attention.heads_join','transpose'),('norm.epsilon','constant'),('norm.variance','mean'),('norm.standardized','real_div'),('head.orientation.peak','max'),('head.orientation.log_denominator','log'),('head.log_softmax','sub'),('head.reverse_classes','take'),('head.reverse_pairs','transpose'),('symmetry.two','constant'),('output.denominator','sum')]
@pytest.mark.parametrize('name,op',TAMPERS)
def test_critical_equation_tampering_is_rejected(name,op,first):
 network,original=first;program=deepcopy(original);r=next(r for r in program['instructions'] if r['id']==name)
 assert r['op']==op
 if op=='constant':
  v=r['attrs']['value'];r['attrs']['value']='0' if isinstance(v,str) else ['1']*len(v)
 elif op=='clip':r['attrs']['upper']='19'
 elif op=='take':
  v=r['attrs']['indices'];v[0],v[1]=v[1],v[0]
 elif op=='concat':r['args'][1],r['args'][2]=r['args'][2],r['args'][1]
 elif op=='reshape':r['attrs']['shape']=[*reversed(r['attrs']['shape'])]
 elif op=='transpose':r['attrs']['axes']=[*reversed(r['attrs']['axes'])]
 elif op in ('mean','sum','max'):r['attrs']['axis']=0
 elif op in ('real_div','sub'):r['args']=list(reversed(r['args']))
 elif op=='log':r['op']='tanh'
 with pytest.raises(ValueError):certify(network,program)

@pytest.mark.parametrize('change',['hardware','domain','original','mapping','error'])
def test_certificate_cannot_strengthen_scope(change,first):
 network,program=first;certificate=certify(network,program)
 if change=='hardware':certificate['hardware_rounding_covered']=True
 elif change=='domain':certificate['domain']['values']='All raw datasets'
 elif change=='original':certificate['original_claim_closed']=True
 elif change=='mapping':certificate['interchange']['sites'].pop()
 else:certificate['error']['mathematical_logits']='0.01'
 with pytest.raises(ValueError):verify(network,program,certificate)

@pytest.mark.parametrize('change',['weights','checkpoint','activation','std','epsilon','extra-state'])
def test_export_cannot_launder_wrong_weights(change,targets):
 row=targets[0];network=deepcopy(read(ROOT/row['export']))
 if change=='weights':network['encoder'][0]['weights'][0][0]='0'
 elif change=='checkpoint':network['checkpoint_sha256']='0'*64
 elif change=='activation':network['encoder'][0]['activation']='identity'
 elif change=='std':network['std'][0]='0'
 elif change=='epsilon':network['norm']['epsilon']='1/100'
 else:network['head']['trunk'].append(deepcopy(network['head']['trunk'][0]))
 with pytest.raises(ValueError):verify_checkpoint(network,ROOT/row['checkpoint'],ROOT/'ncd')

def test_unknown_operator_is_explicitly_rejected(first):
 _,program=first;p=deepcopy(program);p['instructions'][6]['op']='protected_div'
 with pytest.raises(ValueError,match='Unsupported primitive'):validate_program(p)
def test_unproved_extra_computation_rejected(first):
 network,program=first;p=deepcopy(program)
 p['instructions'].append({'id':'unused','op':'tanh','args':['logits'],'attrs':{}})
 with pytest.raises(ValueError,match='orphan'):certify(network,p)
def test_decoder_added_statement_rejected(first):
 network,p=first;p=deepcopy(p);p['decoder']['decode_graph']+='\nprint("hidden computation")'
 with pytest.raises(ValueError):certify(network,p)
def test_shortness_cannot_be_asserted(first):
 network,p=first;p=deepcopy(p);p['mdl_minimality']='proved'
 with pytest.raises(ValueError):certify(network,p)

@pytest.mark.parametrize('index,n',[(i,n) for i in range(4) for n in (3,5,8)])
def test_real_hooks_composed_interventions_and_collateral_states(index,n,actual,tmp_path):
 _,network,model=actual[index];p=compile_graph(network,n);ir=ExplicitGraphProgram(p)
 x=(np.arange(2*n*n*24).reshape(2,n,n,24)%23-11)/19
 base=run_primary(model,x,p);b=ir.run(x,trace=True)
 assert np.max(np.abs(base['logits']-b['logits']))<1e-11
 # Each source tensor differs. Intervene at all single sites and all 16
 # subsets of four separated stages, plus the full 14-site combination.
 assignments={s['state']:((np.arange(base['states'][s['state']].size).reshape(base['states'][s['state']].shape)+i*7)%31-15)/13 for i,s in enumerate(SITES)}
 subset_names=[SITES[i]['state'] for i in (1,7,9,13)]
 subsets=[[subset_names[j] for j in range(4) if mask>>j&1] for mask in range(16)]
 subsets+=[[s['state']] for s in SITES]+[list(assignments)]
 diagnostics=[]
 for subset in subsets:
  patches={name:assignments[name] for name in subset}
  primary=run_primary(model,x,p,patches);explicit=ir.run(x,trace=True,patches=patches)
  assert len(primary['access'])==14 and all(a['module_executed'] for a in primary['access'])
  assert np.max(np.abs(primary['logits']-explicit['logits']))<1e-11
  mismatches=primary['logits'].argmax(-1)!=explicit['labels']
  score_error=float(np.max(np.abs(primary['logits']-explicit['logits'])))
  observed_rounding_slack=4*abs(float(np.spacing(np.max(np.abs(primary['logits'])))))
  disagreements=[]
  for location in np.argwhere(mismatches):
   location=tuple(map(int,location));scores=primary['logits'][location];chosen=int(explicit['labels'][location])
   # A diagnostic check of these computed score vectors, not a hardware
   # error theorem and not an acceptance gate for label equality.
   deficit=float(scores.max()-scores[chosen])
   assert deficit<=2*score_error+observed_rounding_slack
   disagreements.append({'location':list(location),'primary_scores':scores.tolist(),'explicit_scores':explicit['logits'][location].tolist(),'primary_label':int(scores.argmax()),'explicit_label':chosen,'observed_score_deficit':deficit})
  diagnostics.append({'intervened':subset,'maximum_score_error':score_error,'label_disagreements':disagreements,'hardware_label_identity':'observed-failed' if disagreements else 'observed-only','strict_mathematical_refutation':False})
  for s in SITES:
   name=s['state'];assert np.max(np.abs(primary['states'][name]-explicit['states'][name]))<1e-11
   collateral=(primary['states'][name]-base['states'][name])-(explicit['states'][name]-b['states'][name])
   assert np.max(np.abs(collateral))<1e-11
  flags={r['state']:r for r in explicit['execution']}
  assert all(flags[name]['intervened'] and not flags[name]['primitive_evaluated'] for name in subset)
 report={'target':actual[index][0]['id'],'nodes':n,'diagnostic_cases':diagnostics,'input_recipe':'(arange(2*n*n*24).reshape(2,n,n,24)%23-11)/19','patch_recipe':'For site index i: ((arange(mapped_state.size).reshape(mapped_state.shape)+i*7)%31-15)/13','all_domains_have_device_label_guarantee':False,'actual_14_modules_executed_per_case':True,'statistical_independent_worlds':False,'original_claim_closed':False}
 (tmp_path/'graph_intervention_diagnostics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 assert not certify(network,p)['hardware_rounding_covered']
 if n==8:assert any(d['label_disagreements'] for d in diagnostics), 'Keep the reachable hardware disagreement in regression coverage'

@pytest.mark.parametrize('bad',['unknown','shape','nonfinite'])
def test_incompatible_physical_intervention_rejected(bad,actual):
 _,net,model=actual[0];p=compile_graph(net,3);x=np.zeros((1,3,3,24));key=SITES[0]['state'];v=np.zeros((1,9,48))
 if bad=='unknown':key='norm.scale'
 elif bad=='shape':v=np.zeros((1,9,47))
 else:v[0,0,0]=np.inf
 with pytest.raises(ValueError):run_primary(model,x,p,{key:v})
def test_ordinary_division_never_becomes_protected(first):
 _,p=first;ir=ExplicitGraphProgram(p)
 with pytest.raises(ValueError,match='Ordinary division'):ir.run(np.zeros((1,3,3,24)),patches={'norm.scale':np.zeros((1,9,1))})
def test_ties_cycles_and_completion_retained():
 p=np.full((3,3,4),0.25);r=graph_tail(p)
 assert not r['partial_graph'].any() and not r['inferred_dag'].any()
 p[:]=0;p[:,:,0]=1
 for i,j in ((0,1),(1,2),(2,0)):
  p[i,j]=[0,0.8,0.1,0.1];p[j,i]=[0,0.1,0.8,0.1]
 r=graph_tail(p);assert len(r['removed_cycle_edges'])==1 and r['completion_is_not_causal_identification']
 p[:]=0;p[:,:,0]=1;p[0,1]=p[1,0]=[0,0,0,1]
 r=graph_tail(p);assert r['completion_choices']==[[0,1]] and r['partial_graph'][1,0] and not r['inferred_dag'][1,0]

@pytest.fixture
def portable(tmp_path,targets):
 root=tmp_path/'task';(root/'validation').mkdir(parents=True);(root/'ncd').mkdir()
 for name in APPROVED_PRIMARY_SOURCES:shutil.copy2(ROOT/'ncd'/name,root/'ncd'/name)
 shutil.copy2(TARGET,root/'validation/targets.json')
 for row in targets:
  for name in ('checkpoint','export'):
   to=root/row[name];to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/row[name],to)
 config={'schema':'ncd.actual-graph-program-protocol.v1','output':'runs/proof','target_manifest':'validation/targets.json','target_manifest_sha256':digest(root/'validation/targets.json'),'source_sha256':cli.loaded_sources(),'primary_source_sha256':APPROVED_PRIMARY_SOURCES,'stage_seconds':120,'artifact_bytes':128*1024**2,'original_objective_achieved':False}
 path=root/'validation/protocol.json';cli.save(path,config);return root,path

def test_portable_replay_weight_mismatch_and_resigned_scope_forgery(portable):
 root,path=portable;r=cli.prove(path);assert r['proved_feature_domain_cases']==12 and r['physical_module_map_independently_checked']
 out=root/'runs/proof';manifest=out/'manifest.json';assert cli.verify_bundle(manifest)==r
 cert=out/'targets/seed8101_active_graph/n3/certificate.json';v=read(cert);v['hardware_rounding_covered']=True;cli.save(cert,v)
 m=read(manifest);m['files'][cert.relative_to(out).as_posix()]=digest(cert);cli.save(manifest,m)
 with pytest.raises(ValueError,match='tampered'):cli.verify_bundle(manifest)
def test_partial_budget_preserved_and_resumed(portable,monkeypatch):
 root,path=portable;clock=[0]
 def fake_time():clock[0]+=200;return clock[0]
 with monkeypatch.context() as patch:
  patch.setattr(cli,'monotonic',fake_time);r=cli.prove(path)
 assert r['status']=='verified-partial' and len(r['unresolved_cases'])==12
 r=cli.prove(path,resume=True);assert r['status']=='verified' and r['proved_feature_domain_cases']==12
 assert (root/'runs/proof/history/revision_0000/retained_manifest.json').is_file()
def test_missing_proof_domain_rejected_even_if_manifest_resigned(portable):
 root,path=portable;cli.prove(path);out=root/'runs/proof';idx=read(out/'cases.json');idx['cases'].pop();cli.save(out/'cases.json',idx)
 m=read(out/'manifest.json');m['files']['cases.json']=digest(out/'cases.json');cli.save(out/'manifest.json',m)
 with pytest.raises(ValueError,match='Omitted'):cli.verify_bundle(out/'manifest.json')
