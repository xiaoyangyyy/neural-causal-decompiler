from pathlib import Path
import json,hashlib,importlib,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/actual_graph_program_targets_v1';OUT.mkdir()
import ncd,proof_extensions,torch
if ncd.__version__!='0.59.0' or not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/proof_extensions_env_v1') or not Path(proof_extensions.__file__).resolve().is_relative_to(ROOT/'validation/proof_extensions_env_v1'):raise ValueError('Wrong frozen source environment')
torch.set_num_threads(2)
from proof_extensions.frozen_graph import export_graph
from ncd.active_intervention_graph import load_active_factorized_graph

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
records=[]
for seed in (8101,8102):
 training=ROOT/f'runs/original_confirmation_v1/training_seed_{seed}'
 complete=json.loads((training/'complete.json').read_text(encoding='utf-8'));attempt=training/complete['attempt']
 for mode,folder,expected in (('observational_graph','observational_padded','active_input_observational_control_v1'),('active_graph','active_intervention','active_input_intervention_v1')):
  checkpoint=attempt/'models'/folder/'graph_teacher.pt'
  if digest(checkpoint)!=complete['teacher_sha256'][mode]:raise ValueError('Confirmation checkpoint changed')
  network=export_graph(checkpoint);model=load_active_factorized_graph(checkpoint)
  if network['architecture']!=expected or model.training or model.attention.dropout!=0 or model.attention.add_zero_attn or model.attention.bias_k is not None or model.attention.bias_v is not None or model.norm.eps<=0:raise ValueError('Unsupported primary model semantics')
  for name,sha in network['source_sha256'].items():
   if digest(ROOT/'ncd'/name)!=sha:raise ValueError('Current and frozen actual model source differ')
  name=f'seed{seed}_{mode}';file=OUT/(name+'.json')
  file.write_text(json.dumps(network,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
  records.append({'id':name,'seed':seed,'mode':mode,'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':digest(checkpoint),'export':file.relative_to(ROOT).as_posix(),'export_sha256':digest(file),'nodes':[3,5,8],'features':len(network['mean']),'width':network['width'],'source_sha256':network['source_sha256'],'actual_torch_version':torch.__version__,'dropout':'0','masked_key_contract':'All diagonal keys excluded, all n*n queries retained; n>=2 guarantees n*(n-1)>0 allowed keys','normalization_epsilon':network['norm']['epsilon'],'semantics':'Mathematical frozen-parameter computation; device rounding is not certified'})
(OUT/'manifest.json').write_text(json.dumps({'schema':'ncd.actual-graph-program-targets.v1','records':records,'exporter_sha256':digest(__file__),'ground_truth_read':False,'neural_network_suffix_retained':False,'original_claim_closed':False,'whole_project_complete':False},indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('Four actual frozen confirmation graph targets exported; 12 node-size bindings; Torch '+torch.__version__)
