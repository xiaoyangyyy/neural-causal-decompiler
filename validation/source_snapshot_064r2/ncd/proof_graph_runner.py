"""Isolated installed graph verifier; exports scoped contracts, never trains/searches."""
from pathlib import Path
from importlib.metadata import version
import hashlib,importlib,json,sys
APPROVED_SCOPE={'schema': 'ncd.registered-actual-graph-scope.v1', 'targets': ['seed8101_active_graph', 'seed8101_observational_graph', 'seed8102_active_graph', 'seed8102_observational_graph'], 'nodes': [3, 5, 8], 'features': 24, 'width': 48, 'input_values': 'All finite real feature coordinates; every positive finite batch size', 'mathematical_score_probability_error': '0', 'physical_sites': 14, 'compatible_site_subsets': 16384, 'raw_feature_frontend_included': False, 'hardware_rounding_covered': False, 'true_graph_identification_proved': False, 'mdl_minimality_proved': False, 'other_networks_or_maps_covered': False, 'original_claim_closed': False}
APPROVED_CHECKPOINTS={'seed8101_observational_graph': '3e1c5c4967858ba5035fd4f897dd5872c95ee70f70131e45fe72b9cfcc2345fc', 'seed8101_active_graph': '15a3989a1831d5daf09b8c712cd1399cc04c9dad6d48f61c90b97587cb1a9ddf', 'seed8102_observational_graph': '4a29397323b775543afc8ad33741589773b5c1969f71db86964e1c618f76e80f', 'seed8102_active_graph': '3cd093729cbfe097663b748a2c39018efbb44a57cc91038cebd97c973c03c5e1'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def domain(n):return {'batch':'Every positive finite integer B','features':['B',n,n,24],'values':'All finite real feature coordinates, including diagonal tokens, clipping boundaries and label ties'}
def unresolved_contracts(job,reason):
 return [{'id':'actual_graph_math_'+target+'_n'+str(n),'target':target,'nodes':n,'checkpoint_sha256':APPROVED_CHECKPOINTS[target],'domain':domain(n),'scope':APPROVED_SCOPE,'status':'unresolved','reason':reason,'original_claim_closed':False} for target in APPROVED_SCOPE['targets'] for n in APPROVED_SCOPE['nodes']]
def contracts(artifact,actual):
 base=Path(artifact).parent;index=read(base/'cases.json');result=[]
 for case in index['cases']:
  folder=base/case['directory'];certificate=read(folder/'certificate.json')
  if certificate['domain']!=domain(case['nodes']) or certificate['checkpoint_sha256']!=APPROVED_CHECKPOINTS[case['target']] or certificate['error']['mathematical_logits']!='0' or certificate['error']['mathematical_probabilities']!='0' or len(certificate['interchange']['sites'])!=14 or certificate['hardware_rounding_covered'] or certificate['original_claim_closed']:
   raise ValueError('Graph theorem scope strengthened or another target')
  result.append({'id':'actual_graph_math_'+case['target']+'_n'+str(case['nodes']),'target':case['target'],'nodes':case['nodes'],'checkpoint_sha256':certificate['checkpoint_sha256'],'network_sha256':certificate['network_sha256'],'program_sha256':certificate['program_sha256'],'domain':certificate['domain'],'scope':APPROVED_SCOPE,'status':'proved','error':certificate['error'],'fixed_map':certificate['interchange'],'certificate_sha256':digest(folder/'certificate.json'),'proof_method':'Typed local equations and exact stored coefficients; topological composition with ideal real module contracts; device rounding excluded','uncovered':certificate['unsupported_or_uncovered'],'original_claim_closed':False})
 for case in index['unresolved']:
  result.extend(c for c in unresolved_contracts({},case['reason']) if c['target']==case['target'] and c['nodes']==case['nodes'])
 if len(result)!=12 or {(r['target'],r['nodes']) for r in result}!={(t,n) for t in APPROVED_SCOPE['targets'] for n in APPROVED_SCOPE['nodes']}:raise ValueError('Missing actual graph proof domain')
 if actual['proved_feature_domain_cases']!=sum(c['status']=='proved' for c in result):raise ValueError('Graph proof coverage mismatch')
 return sorted(result,key=lambda c:c['id'])
def main():
 request=read(sys.argv[1]);snapshot=Path(request['snapshot']).resolve();environment=Path(request['environment']).resolve();job=request['job']
 if request['schema']!='ncd.graph-proof-request.v1' or Path.cwd().resolve()!=snapshot or not Path(sys.executable).resolve().is_relative_to(environment) or request['threads']!=2:raise ValueError('Graph verifier environment/resource mismatch')
 if job['kind']!='actual_graph_ssa' or job['graph_scope']!=APPROVED_SCOPE or job['original_claim_closed'] is not False:raise ValueError('Unregistered graph verification scope')
 folder=Path(importlib.import_module('graph_ssa_proof').__file__).resolve().parent
 expected={Path(f).name:sha for f,sha in request['source_sha256'].items() if f.startswith('graph_ssa_proof/')}
 if not folder.is_relative_to(environment) or version('ncd-actual-graph-proof')!='0.1.0' or len(expected)!=8 or set(expected)!={p.name for p in folder.glob('*.py')}:raise ValueError('Wrong installed graph proof package')
 for file,sha in expected.items():
  if digest(folder/file)!=sha or digest(snapshot/'graph_ssa_proof'/file)!=sha:raise ValueError('Graph kernel source mismatch')
 artifact=Path(job['artifact'])
 if digest(artifact)!=job['artifact_sha256']:raise ValueError('Registered graph artifact changed')
 import torch
 torch.set_num_threads(2)
 from graph_ssa_proof.__main__ import verify_bundle
 actual=verify_bundle(artifact)
 if actual['status'] not in ('verified','verified-partial') or actual['frozen_networks']!=4 or not actual['checkpoint_bindings_independently_verified'] or not actual['physical_module_map_independently_checked'] or any(actual[k] is not False for k in ('raw_frontend_included','hardware_rounding_covered','true_graph_identification_proved','mdl_minimality_proved','original_claim_closed','original_objective_achieved')):raise ValueError('Graph replay does not establish the registered scoped contract')
 records=contracts(artifact,actual)
 result={'status':'verified','conclusion':'proved' if all(c['status']=='proved' for c in records) else 'unresolved','kernel':'graph_ssa_proof','program_realization_contracts':records,'independent_bundle_verification':actual,'graph_truth_or_equations_read':False,'neural_inference_used_to_prove':False,'original_claim_closed':False,'original_objective_achieved':False}
 Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
