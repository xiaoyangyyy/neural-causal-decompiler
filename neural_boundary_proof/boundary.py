from pathlib import Path
from fractions import Fraction as Q
from collections import Counter
from ncd.io import digest
from neural_ssa_proof.compiler import target,validate_program,key
from neural_ssa_proof.verification import verify as verify_realization,hash_value
from neural_ssa_proof.runtime import interval_program

PAIRS=(('x','flip'),('mean','mean_3'),('std','std_1'),('clamp_min','clamp_min_1'),('sub','sub_1'),('truediv','truediv_1'),('clamp','clamp_1'),('encoder_0','encoder_4'),('encoder_1','encoder_5'),('encoder_2','encoder_6'),('encoder_3','encoder_7'),('mean_1','mean_4'),('square','square_1'),('mean_2','mean_5'),('getitem','getitem_1'),('log','log_1'),('cat','cat_1'),('head_0','head_3'),('head_1','head_4'),('head_2','head_5'))
POINTWISE={'add','sub','mul','div','tanh','square','sqrt','log','lt','if','clip'}

def sample_invariance(p):
 validate_program(p);allowed=POINTWISE|{'var','constant','mean','var_stat','argmax'}
 counts=Counter(n['op'] for n in p['nodes'])
 if not set(counts)<=allowed:raise ValueError('Operator lacks a row-permutation relation')
 for n in p['nodes']:
  if n['op'] in ('mean','var_stat') and n['kind']!='scalar':raise ValueError('Invalid invariant reduction')
  if n['op'] in ('sqrt','log','lt','if','argmax') and n['kind']=='vector':raise ValueError('Unproved vector guard/reduction')
 return {'domain':'all finite real N x 2 inputs, N >= 16, without internal interventions',
 'operators':dict(sorted(counts.items())),'primitive_relations':{'var':'common row permutation on raw vectors','constant':'scalar unchanged','pointwise':'common row permutation for vectors; scalar arguments unchanged','mean':'exact finite-sum invariance under bijective row reindexing','var_stat':'mean invariant; centered squares reindexed; exact mean invariant','argmax':'scalar logits unchanged, including ties'},
 'conclusion':'Every scalar SSA coordinate is unchanged; every vector coordinate is reindexed by the common permutation. Induction is in SSA order.',
 'device_arithmetic':'not claimed'}

def branch_identity(t,p):
 rows={r['name']:r for r in t['fx']['nodes']};renames=dict(PAIRS)
 def rename(value):
  if isinstance(value,dict):return {'node':renames[value['node']]} if set(value)=={'node'} else {k:rename(v) for k,v in value.items()}
  if isinstance(value,list):return [rename(v) for v in value]
  return value
 for a,b in PAIRS[1:]:
  ra,rb=rows[a],rows[b]
  if any(ra[k]!=rb[k] for k in ('op','target','kwargs')) or rename(ra['args'])!=rb['args']:raise ValueError('Swapped branch relation mismatch')
 flip=rows['flip']
 if flip['target']!='flip' or flip['args']!=[{'node':'x'},-1] or flip['kwargs']:raise ValueError('Input swap differs')
 output=p['logits'];h=p['groups']['head_2']['refs'];hs=p['groups']['head_5']['refs'];perm=p['groups']['getitem_2']['refs'];nodes=p['nodes']
 for j,r in enumerate(perm):
  node=nodes[r]
  if node['op']!='add' or node['args'][0]!=hs[[1,0,2,3][j]] or nodes[node['args'][1]]['op']!='constant' or Q(nodes[node['args'][1]]['value'])!=0:raise ValueError('Final class reindex mismatch')
 for j,r in enumerate(output):
  n=nodes[r]
  if n['op']!='mul' or nodes[n['args'][0]]['op']!='constant' or Q(nodes[n['args'][0]]['value'])!=Q(1,2):raise ValueError('Final average mismatch')
  a=nodes[n['args'][1]]
  if a['op']!='add' or a['args']!=[h[j],perm[j]]:raise ValueError('Final sum mismatch')
 return {'paired_tensor_groups':[list(pair) for pair in PAIRS],'class_swap':[1,0,2,3],
 'identity':'Both branches are the same row-invariant scalar head H on x and swap(x); logits = (H(x) + class_swap(H(swap(x))))/2.'}

def certify(realization,program):
 verify_realization(realization,program);t=target(realization['checkpoint']);row_proof=sample_invariance(program);branches=branch_identity(t,program)
 f=Q(1e-5);data=[[str(-f if i%2 else f),str(f if i%2 else -f)] for i in range(16)];permutation=[i^1 for i in range(16)];swapped=[r[::-1] for r in data]
 if swapped!=[data[i] for i in permutation] or sorted(permutation)!=list(range(16)):raise ValueError('Input symmetry witness invalid')
 result=interval_program(program,data);enclosures=result['logit_enclosures'];margin=min(Q(v[0]) for v in enclosures[:2])-max(Q(v[1]) for v in enclosures[2:])
 if margin<=0:raise ValueError('No strictly certified directional dominance')
 return {'schema':'ncd.actual-discoverer-label-swap-boundary.v1','status':'refuted','target':realization['checkpoint'],'checkpoint_sha256':digest(realization['checkpoint']),
 'realization_certificate_sha256':hash_value(realization),'program_sha256':hash_value(program),'sample_permutation_theorem':row_proof,'swapped_branch_relation':branches,
 'data':data,'row_permutation':permutation,'logit_enclosures':enclosures,'strict_direction_margin_lower':str(margin),
 'exact_tie_reason':'swap(data) is a row permutation; branch head H is row invariant; the two heads are identical. Thus logits[0]=logits[1] exactly.',
 'label_on_input':0,'label_on_swapped_input':0,'required_swapped_label':1,
 'claim_refuted':'This actual frozen classifier returns variable-swap-equivariant single labels on every finite real N x 2 input, N >= 16.',
 'quantifier':'one admissible input disproves this universal single-label claim; this is not a failed candidate search',
 'input_admissibility':'finite N=16, both marginal variances positive; collinear and exact tie boundaries are included in the declared real-input contract',
 'scope':'mathematical loaded network and lowest-index single-label decision; score equivariance is preserved',
 'not_refuted':['almost-sure distributional claims that explicitly exclude these measure-zero boundaries','set-valued or CPDAG outputs','other classifiers or redesigned tie rules','sample invariance'],
 'original_claim_id':'R2.variable_equivariance','original_claim_scope_assessment':'applicable to the universal single-label finite-input contract; original ledger closure requires trusted audit integration',
 'original_claim_closed_by_this_bundle':False,'device_rounding_certified':False,'original_objective_achieved':False}

def verify(certificate,realization,program):
 expected=certify(realization,program)
 if certificate!=expected:raise ValueError('Counterexample, inequality, scope or relation mismatch')
 return {'status':'verified','conclusion':'refuted','strict_real_input_counterexample':True,'target_reachability_proved':True,'universal_single_label_equivariance_refuted':True,
 'score_equivariance_refuted':False,'all_classifiers_refuted':False,'original_objective_achieved':False}
