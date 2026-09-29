from pathlib import Path
import json,hashlib,os,importlib
import numpy as np,torch
from graph_ssa_proof.runtime import ExplicitGraphProgram
from graph_ssa_proof.binding import load_primary_model
from graph_ssa_proof.mapping import SITES,run_primary
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/actual_graph_controls_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_actual_graph_controls_v1';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=read(ROOT/'validation/actual_graph_protocol_v1.json');base=ROOT/config['output'];package=Path(importlib.import_module('graph_ssa_proof').__file__).resolve().parent
if not package.is_relative_to(ROOT/'validation/actual_graph_env_v1') or {'graph_ssa_proof/'+p.name:digest(p) for p in package.glob('*.py')}!=config['source_sha256']:raise ValueError('Wrong installed read/write interpreter')
if (OUT/'status.json').exists():raise FileExistsError('Retain prior diagnostic controls')
selected=[SITES[i]['state'] for i in (1,7,9)];records=[];rng=np.random.default_rng(8100)
for row in read(base/'target_manifest.json')['records']:
 folder=base/'targets'/row['id'];model=load_primary_model(folder/'checkpoint.pt',base/'primary_sources').double()
 for n in (3,5,8):
  program=read(folder/('n'+str(n))/'program.json');ir=ExplicitGraphProgram(program)
  features=(np.arange(2*n*n*24).reshape(2,n,n,24)%29-14)/17
  patches={}
  for j,name in enumerate(selected):
   source=(np.arange(2*n*n*24).reshape(2,n,n,24)%(31+j*4)-15)/(19+j*2)
   patches[name]=run_primary(model,source,program)['states'][name]
  primary=run_primary(model,features,program,patches);proper=ir.run(features,trace=True,patches=patches)
  proper_state=max(float(np.max(np.abs(primary['states'][s['state']]-proper['states'][s['state']]))) for s in SITES)
  if proper_state>1e-11:raise ValueError('Fixed mapped states failed diagnostic')
  controls={}
  controls['random_read']={name:patches[name]@rng.normal(size=(program['width'],program['width']))/np.sqrt(program['width']) for name in selected}
  controls['shuffled_source_rows']={name:patches[name][::-1].copy() for name in selected}
  controls['permuted_coordinates']={name:np.roll(patches[name],7,axis=-1) for name in selected}
  details=[]
  for kind,wrong in controls.items():
   bad=ir.run(features,trace=True,patches=wrong)
   target=max(float(np.max(np.abs(primary['states'][name]-bad['states'][name]))) for name in selected)
   output=float(np.max(np.abs(primary['logits']-bad['logits'])))
   if target<=1e-6 or output<=1e-6:raise ValueError('Control did not detect this wrong mapping')
   details.append({'kind':kind,'mapped_target_state_error':target,'final_score_error':output,'this_control_mapping_matches':False,'all_mapping_families_refuted':False,'strict_refutation_certificate':False})
  records.append({'target':row['id'],'nodes':n,'proper_maximum_state_error':proper_state,'source_sites':selected,'independent_site_sources':True,'controls':details})
status={'schema':'ncd.actual-graph-controls.v1','status':'diagnostics-complete','cases':records,'graph_domain_cases':12,'negative_controls':36,'control_seed':8100,'new_worlds_or_training_runs':0,'rng_used_only_for_development_read_controls':True,'actual_neural_hooks_used':True,'graph_truth_or_equations_read':False,'correct_program_math_guarantee_from_this_diagnostic':False,'device_label_guarantee':False,'all_mapping_families_refuted':False,'source_sha256':config['source_sha256'],'script_sha256':digest(__file__),'accepted_bundle_sha256':digest(base/'manifest.json'),'original_claim_closed':False}
(OUT/'status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('Thirty-six wrong read controls detected; fixed physical-map diagnostics retained, no family refutation claimed')
