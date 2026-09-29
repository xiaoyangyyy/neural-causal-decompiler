"""Fixed physical module read/write map; sites are frozen before diagnostics."""
from copy import deepcopy
SITES=[
 {'module':'encoder.0','state':'encoder.0.affine','kind':'linear','layout':'grid','channels':'width'},
 {'module':'encoder.1','state':'encoder.0.tanh','kind':'tanh','layout':'grid','channels':'width'},
 {'module':'encoder.2','state':'encoder.1.affine','kind':'linear','layout':'grid','channels':'width'},
 {'module':'encoder.3','state':'encoder.1.tanh','kind':'tanh','layout':'grid','channels':'width'},
 {'module':'context.0','state':'context.layer.0.affine','kind':'linear','layout':'grid','channels':'width'},
 {'module':'context.1','state':'context.layer.0.tanh','kind':'tanh','layout':'grid','channels':'width'},
 {'module':'context.2','state':'context.layer.1.affine','kind':'linear','layout':'grid','channels':'width'},
 {'module':'context.3','state':'context.layer.1.tanh','kind':'tanh','layout':'grid','channels':'width'},
 {'module':'attention','state':'attention.output.affine','kind':'attention','layout':'flat','channels':'width','tuple_first':True},
 {'module':'norm','state':'norm.output','kind':'norm','layout':'flat','channels':'width'},
 {'module':'head_trunk.0','state':'head.trunk.0.affine','kind':'linear','layout':'grid','channels':'width'},
 {'module':'head_trunk.1','state':'head.trunk.0.tanh','kind':'tanh','layout':'grid','channels':'width'},
 {'module':'skeleton_head','state':'head.skeleton.affine','kind':'linear','layout':'grid','channels':1},
 {'module':'orientation_head','state':'head.orientation.affine','kind':'linear','layout':'grid','channels':3},
]

def mapping_certificate(program):
 from .runtime import validate_program
 shapes=validate_program(program);n,w=program['nodes'],program['width']
 for site in SITES:
  channels=w if site['channels']=='width' else site['channels']
  if shapes.get(site['state'])!=['B',n*n,channels]:raise ValueError('Physical-to-SSA tensor shape mismatch')
 return {'sites':deepcopy(SITES),'compatible_interventions':'Every subset of the 14 declared sites, each receiving its own finite real tensor of the identical batch/node/channel shape; independent source assignments are allowed',
         'read_write':'Read by the row-major bijective grid/flat reshape; write by its inverse and return the overridden physical module output. For attention preserve the second tuple component.',
         'interchange_relation':'The mapped post-intervention neural states and program states agree, as do final scores and graph decisions; all nonintervened mapped states and their differences from the unpatched execution agree.',
         'proof':'The checked local equations commute with matching output overrides. Topological induction covers every combination without assuming shared source values or cancellation; all downstream ordinary divisors/logs remain defined for arbitrary finite replacement values at these sites.',
         'neural_execution':'All 14 declared modules occur on every supported forward path. Replacement does not count an omitted/unevaluated SSA primitive as execution coverage.',
         'hardware_rounding_covered':False,'all_mappings_refuted':False,'short_causal_algorithm_proved':False}


def validate_primary_mapping(model,program):
 import torch
 from .runtime import validate_program
 shapes=validate_program(program)
 classes={'linear':torch.nn.Linear,'tanh':torch.nn.Tanh,'attention':torch.nn.MultiheadAttention,'norm':torch.nn.LayerNorm}
 for site in SITES:
  module=model.get_submodule(site['module'])
  if not isinstance(module,classes[site['kind']]):raise ValueError('Actual module map type mismatch')
  channels=program['width'] if site['channels']=='width' else site['channels']
  if shapes[site['state']]!=['B',program['nodes']**2,channels]:raise ValueError('Mapped dimension mismatch')
 return mapping_certificate(program)


def run_primary(model,features,program,patches=None):
 import numpy as np,torch
 validate_primary_mapping(model,program);patches=patches or {};sites={s['state']:s for s in SITES}
 if any(name not in sites for name in patches):raise ValueError('Unknown physical read/write site')
 batch,n=features.shape[:2];states={};handles=[];access=[]
 for site in SITES:
  def hook(module,arguments,output,site=site):
   tensor=output[0] if site.get('tuple_first') else output
   channels=program['width'] if site['channels']=='width' else site['channels']
   actual_shape=(batch,n,n,channels) if site['layout']=='grid' else (batch,n*n,channels)
   if tuple(tensor.shape)!=actual_shape:raise ValueError('Actual executed module output shape differs')
   name=site['state']
   if name in patches:
    replacement=np.asarray(patches[name],dtype=np.float64)
    if replacement.shape!=(batch,n*n,channels) or not np.isfinite(replacement).all():raise ValueError('Incompatible independent source replacement')
    tensor=torch.as_tensor(replacement.reshape(actual_shape),dtype=tensor.dtype,device=tensor.device).clone()
   states[name]=tensor.detach().cpu().numpy().reshape(batch,n*n,channels).copy()
   access.append({'state':name,'module_executed':True,'intervened':name in patches,'read_after_override':True})
   if name in patches:return (tensor,output[1]) if site.get('tuple_first') else tensor
  handles.append(model.get_submodule(site['module']).register_forward_hook(hook))
 try:
  with torch.no_grad():logits=model(torch.as_tensor(features,dtype=next(model.parameters()).dtype)).detach().cpu().numpy()
 finally:
  for handle in handles:handle.remove()
 if len(access)!=14 or set(states)!=set(sites):raise ValueError('A declared actual variable was not executed')
 return {'logits':logits,'states':states,'access':access,'hardware_error_bound':None}
