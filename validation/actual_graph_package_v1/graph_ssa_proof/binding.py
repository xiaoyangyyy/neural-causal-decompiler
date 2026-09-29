"""Independently bind coefficients to the actual checkpoint and primary sources."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib
APPROVED_PRIMARY_SOURCES={'active_intervention_graph.py': 'b132c5be445e85293a8b410b9303b6a30b4e9454736ba6d149cd83313d0b0d35', 'factorized_node_context_graph.py': '41fecea62fc71caaa911663dc3461a3580111f888aa96c15245868488e7459b8', 'graph_model.py': 'de044407e023a4e575cdb7781bd1e893d277fd87a1058e1f6c866fc3805b0873', 'model.py': 'ee732b8f06c153951e6f3e559c9511591e8e6e632d815766c5192aff50bb3602', 'node_context_graph.py': '0ab3ff815813ae543f052866390223979e8b57dfc94ce254317e4b8fbde12298', 'graphs.py': '119ef2757b9200c05b3f2fddebf898bf189e4844e6010d29abf61480764085fc'}

APPROVED_CHECKPOINTS={'seed8101_observational_graph': '3e1c5c4967858ba5035fd4f897dd5872c95ee70f70131e45fe72b9cfcc2345fc', 'seed8101_active_graph': '15a3989a1831d5daf09b8c712cd1399cc04c9dad6d48f61c90b97587cb1a9ddf', 'seed8102_observational_graph': '4a29397323b775543afc8ad33741589773b5c1969f71db86964e1c618f76e80f', 'seed8102_active_graph': '3cd093729cbfe097663b748a2c39018efbb44a57cc91038cebd97c973c03c5e1'}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numbers(tensor):
 def walk(v):return [walk(x) for x in v] if isinstance(v,list) else str(Q(v))
 return walk(tensor.detach().cpu().tolist())

def verify_checkpoint(network,checkpoint,primary_sources):
 checkpoint=Path(checkpoint);primary_sources=Path(primary_sources)
 if network['checkpoint_sha256'] not in APPROVED_CHECKPOINTS.values():raise ValueError('Checkpoint outside the four frozen actual targets')
 if digest(checkpoint)!=network['checkpoint_sha256']:raise ValueError('Frozen weight identity mismatch')
 for name,expected in APPROVED_PRIMARY_SOURCES.items():
  if digest(primary_sources/name)!=expected:raise ValueError('Unknown actual model/graph source')
 if network['source_sha256']!={name:sha for name,sha in APPROVED_PRIMARY_SOURCES.items() if name!='graphs.py'}:raise ValueError('Primary neural source binding changed')
 from .decoder import APPROVED_AST_HASHES
 import ast
 found={}
 for filename in ('graph_model.py','graphs.py'):
  for item in ast.parse((primary_sources/filename).read_text(encoding='utf-8-sig')).body:
   if isinstance(item,ast.FunctionDef) and item.name in APPROVED_AST_HASHES:
    if item.name in found:raise ValueError('Duplicate primary graph-tail definition')
    found[item.name]=hashlib.sha256(ast.dump(item,include_attributes=False).encode()).hexdigest()
 if found!=APPROVED_AST_HASHES:raise ValueError('Graph tail is not the actual primary algorithm')
 import torch
 if torch.__version__!='2.12.1+cpu':raise ValueError('Actual primary library version outside frozen specification')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if saved['width']!=network['width'] or saved['architecture']!=network['architecture'] or saved['architecture'] not in ('active_input_observational_control_v1','active_input_intervention_v1'):raise ValueError('Wrong primary architecture')
 state=saved['state_dict'];expected=set()
 def tensor(key,value):
  expected.add(key)
  if key not in state or not bool(torch.isfinite(state[key]).all()) or numbers(state[key])!=value:raise ValueError('Actual frozen tensor differs: '+key)
 def affine(layer,prefix,activation):
  if set(layer)!={'weights','bias','activation'} or layer['activation']!=activation:raise ValueError('Primary affine/Tanh mapping changed')
  tensor(prefix+'.weight',layer['weights']);tensor(prefix+'.bias',layer['bias'])
 for field in ('mean','std'):tensor(field,network[field])
 if len(network['encoder'])!=2 or len(network['context'])!=2 or len(network['head']['trunk'])!=1:raise ValueError('Primary sequential layers changed')
 for name in ('encoder','context'):
  for i,layer in zip((0,2),network[name]):affine(layer,name+'.'+str(i),'tanh')
 tensor('attention.in_proj_weight',network['attention']['in_weights']);tensor('attention.in_proj_bias',network['attention']['in_bias'])
 affine(network['attention']['out'],'attention.out_proj','identity')
 if network['attention']['heads']!=4:raise ValueError('Primary head count changed')
 tensor('norm.weight',network['norm']['weights']);tensor('norm.bias',network['norm']['bias'])
 if network['norm']['epsilon']!=str(Q(torch.nn.LayerNorm(network['width']).eps)):raise ValueError('Actual norm epsilon differs')
 affine(network['head']['trunk'][0],'head_trunk.0','tanh')
 affine(network['head']['skeleton'],'skeleton_head','identity');affine(network['head']['orientation'],'orientation_head','identity')
 if set(state)!=expected:raise ValueError('Unmapped actual neural parameters')
 if network['schema']!='ncd.frozen-graph-math.v1' or network['semantics']!='exact serialized coefficients and mathematical operators; device rounding excluded':raise ValueError('Changed mathematical/device scope')
 return {'status':'verified','checkpoint_sha256':digest(checkpoint),'actual_tensors_bound':len(expected),'primary_source_sha256':APPROVED_PRIMARY_SOURCES,'torch_version':torch.__version__,'actual_state_dict_read':True,'neural_inference_used_to_prove_fidelity':False}


def load_primary_model(checkpoint,primary_sources):
 """Instantiate the exact approved source class ASTs, without checkout imports."""
 import ast,torch
 primary_sources=Path(primary_sources)
 for name,expected in APPROVED_PRIMARY_SOURCES.items():
  if digest(primary_sources/name)!=expected:raise ValueError('Primary source changed')
 specification=(('graph_model.py','GraphDiscoverer'),('node_context_graph.py','NodeContextGraphDiscoverer'),('factorized_node_context_graph.py','FactorizedNodeContextGraphDiscoverer'),('active_intervention_graph.py','ActiveFactorizedGraphDiscoverer'))
 body=[]
 for filename,class_name in specification:
  module=ast.parse((primary_sources/filename).read_text(encoding='utf-8-sig'))
  body.append(next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name==class_name))
 namespace={'__name__':'frozen_actual_graph_primary','torch':torch,'nn':torch.nn,'GRAPH_FEATURES':tuple(range(20)),'ACTIVE_FEATURE_COUNT':24,'SWAP_LABELS':[0,2,1,3]}
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<approved-actual-graph-classes>','exec'),namespace)
 if torch.__version__!='2.12.1+cpu':raise ValueError('Actual primary library version outside frozen specification')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if digest(checkpoint) not in APPROVED_CHECKPOINTS.values():raise ValueError('Unregistered primary checkpoint')
 model=namespace['ActiveFactorizedGraphDiscoverer'](saved['width']);model.load_state_dict(saved['state_dict']);model.eval()
 return model
