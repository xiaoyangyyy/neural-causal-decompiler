"""Independent local-equation checker; never calls the program compiler."""
from fractions import Fraction as Q
from copy import deepcopy
import hashlib,json
from .runtime import validate_program
from .decoder import validate_decoder
from .mapping import mapping_certificate

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def hash_value(v):return hashlib.sha256(canonical(v).encode()).hexdigest()


def local_relations(network,program):
 shapes=validate_program(program);rows={r['id']:r for r in program['instructions']};checked=set();proof=[]
 n=program['nodes'];w=network['width'];heads=network['attention']['heads'];d=w//heads
 if network['architecture'] not in ('active_input_observational_control_v1','active_input_intervention_v1') or network['schema']!='ncd.frozen-graph-math.v1' or heads!=4 or w%heads or len(network['mean'])!=24 or len(network['std'])!=24 or any(Q(x)<=0 for x in network['std']) or Q(network['norm']['epsilon'])<=0:raise ValueError('Primary real domain/normalizer conditions')
 if (program['network_sha256'],program['checkpoint_sha256'],program['architecture'],program['width'],program['features'])!=(hash_value(network),network['checkpoint_sha256'],network['architecture'],w,24):raise ValueError('Program bound to a different network')
 if program['semantics']!='Exact real elementary tensor operations and exact frozen coefficients; floating runtime is diagnostic only' or program['mdl_minimality']!='unresolved; parameter-bearing explicit computation is not a shortest causal algorithm':raise ValueError('Strengthened semantics/minimality')
 def node(name,operator,args,attrs=None):
  expected={'id':name,'op':operator,'args':args,'attrs':attrs or {}}
  if name not in rows or canonical(rows[name])!=canonical(expected):raise ValueError('Local primitive equation differs: '+name)
  checked.add(name);return name
 def constant(name,value):return node(name,'constant',[],{'value':value})
 def affine(prefix,incoming,layer,activation):
  weight=constant(prefix+'.weight',layer['weights']);bias=constant(prefix+'.bias',layer['bias'])
  transpose=node(prefix+'.weight_transpose','transpose',[weight],{'axes':[1,0]})
  product=node(prefix+'.product','matmul',[incoming,transpose])
  output=node(prefix+'.affine','add',[product,bias])
  if layer['activation']!=('tanh' if activation else 'identity'):raise ValueError('Actual activation changed')
  return node(prefix+'.tanh','tanh',[output]) if activation else output
 input_ref=node('features','input',[],{'shape':['B',n,n,24]})
 token_ref=node('feature_tokens','reshape',[input_ref],{'shape':['B',n*n,24]})
 mean_ref=constant('normalization.mean',network['mean']);std_ref=constant('normalization.std',network['std'])
 centered=node('normalization.centered','sub',[token_ref,mean_ref]);scaled=node('normalization.scaled','real_div',[centered,std_ref])
 encoded=node('normalization.clipped','clip',[scaled],{'lower':'-20','upper':'20'})
 for i in range(2):encoded=affine('encoder.'+str(i),encoded,network['encoder'][i],True)
 proof.append({'relation':'shared encoder','argument':'Each serialized coefficient and bias is checked against the independently bound actual tensor; finite product-sums and equal-argument Tanh preserve equality for every real feature coordinate. Ordinary divisions use the positive frozen std, not protected CDIR.'})
 # Independently derive mask sets by integer quotient/remainder, rather than
 # the nested-loop construction used by the compiler.
 allowed=[p for p in range(n*n) if p//n!=p%n]
 outs=[[p for p in allowed if p//n==i] for i in range(n)]
 ins=[[p for p in allowed if p%n==i] for i in range(n)]
 reductions={}
 for role,indices in (('outgoing',outs),('incoming',ins)):
  selected=node('context.'+role+'.selected','take',[encoded],{'axis':1,'indices':indices})
  reductions[role]=node('context.'+role,'mean',[selected],{'axis':2,'keepdim':False})
 selected=node('context.global.selected','take',[encoded],{'axis':1,'indices':allowed})
 global_ref=node('context.global','mean',[selected],{'axis':1,'keepdim':False})
 concat=[encoded]
 for role in ('source','target'):
  shape=['B',n,1,w] if role=='source' else ['B',1,n,w]
  for direction in ('outgoing','incoming'):
   prefix='context.'+role+'.'+direction
   reshaped=node(prefix+'.reshape','reshape',[reductions[direction]],{'shape':shape})
   expanded=node(prefix+'.broadcast','broadcast',[reshaped],{'shape':['B',n,n,w]})
   concat.append(node(prefix+'.tokens','reshape',[expanded],{'shape':['B',n*n,w]}))
 global_ref=node('context.global.reshape','reshape',[global_ref],{'shape':['B',1,w]})
 concat.append(node('context.global.broadcast','broadcast',[global_ref],{'shape':['B',n*n,w]}))
 context=node('context.concatenated','concat',concat,{'axis':2})
 for i in range(2):context=affine('context.layer.'+str(i),context,network['context'][i],True)
 enriched=node('context.enriched','add',[encoded,context])
 proof.append({'relation':'node/global contexts','argument':'Deleting zero diagonal terms gives exactly n-1 incoming/outgoing terms and n*(n-1) global terms. Source/target broadcasts preserve each indexed coordinate and concatenate in the actual six-block order.'})
 attention=network['attention'];projection=affine('attention.projection',enriched,{'weights':attention['in_weights'],'bias':attention['in_bias'],'activation':'identity'},False)
 parts={}
 for index,role in enumerate(('query','key','value')):
  selected=node('attention.'+role+'.selected','take',[projection],{'axis':2,'indices':[index*w+j for j in range(w)]})
  split=node('attention.'+role+'.split','reshape',[selected],{'shape':['B',n*n,heads,d]})
  ref=node('attention.'+role+'.heads','transpose',[split],{'axes':[0,2,1,3]})
  if role!='query':ref=node('attention.'+role+'.allowed','take',[ref],{'axis':2,'indices':allowed})
  parts[role]=ref
 transpose=node('attention.key.transpose','transpose',[parts['key']],{'axes':[0,1,3,2]})
 scores=node('attention.dot','matmul',[parts['query'],transpose])
 dimension=constant('attention.head_width',str(d));scale=node('attention.scale','sqrt',[dimension])
 scores=node('attention.scores','real_div',[scores,scale])
 peak=node('attention.peak','max',[scores],{'axis':3,'keepdim':True})
 shifted=node('attention.centered','sub',[scores,peak]);exponents=node('attention.exponentials','exp',[shifted])
 total=node('attention.denominator','sum',[exponents],{'axis':3,'keepdim':True})
 probabilities=node('attention.probabilities','real_div',[exponents,total])
 weighted=node('attention.weighted_values','matmul',[probabilities,parts['value']])
 joined=node('attention.heads_join','transpose',[weighted],{'axes':[0,2,1,3]})
 joined=node('attention.join','reshape',[joined],{'shape':['B',n*n,w]})
 attended=affine('attention.output',joined,attention['out'],False)
 proof.append({'relation':'complete masked multihead attention','argument':'All n*n query tokens remain; only the n*(n-1) non-diagonal keys/values enter every head. Their nonempty score set has an attained maximum, hence at least one exp(score-max)=1, sum>=1. The checked Q/K/V slicing and head transpose exactly implement scaled finite dot-product softmax followed by value aggregation and frozen output projection.'})
 residual=node('norm.input','add',[enriched,attended])
 mean=node('norm.mean','mean',[residual],{'axis':2,'keepdim':True})
 centered=node('norm.centered','sub',[residual,mean]);square=node('norm.squares','mul',[centered,centered])
 variance=node('norm.variance','mean',[square],{'axis':2,'keepdim':True})
 epsilon=constant('norm.epsilon',network['norm']['epsilon'])
 radicand=node('norm.radicand','add',[variance,epsilon]);scale=node('norm.scale','sqrt',[radicand])
 standard=node('norm.standardized','real_div',[centered,scale])
 weight=constant('norm.weight',network['norm']['weights']);bias=constant('norm.bias',network['norm']['bias'])
 weighted=node('norm.weighted','mul',[standard,weight]);hidden=node('norm.output','add',[weighted,bias])
 proof.append({'relation':'learned LayerNorm','argument':'The population variance is mean of squared deviations, so variance+eps>=eps>0 and sqrt is positive. Every centered coordinate, learned scale and learned bias agrees with the actual LayerNorm definition. No additional clipping is inserted.'})
 trunk=affine('head.trunk.0',hidden,network['head']['trunk'][0],True)
 skeleton=affine('head.skeleton',trunk,network['head']['skeleton'],False)
 orientation=affine('head.orientation',trunk,network['head']['orientation'],False)
 peak=node('head.orientation.peak','max',[orientation],{'axis':2,'keepdim':True})
 shifted=node('head.orientation.shifted','sub',[orientation,peak]);exponents=node('head.orientation.exponentials','exp',[shifted])
 total=node('head.orientation.denominator','sum',[exponents],{'axis':2,'keepdim':True});logtotal=node('head.orientation.log_denominator','log',[total])
 conditional=node('head.log_softmax','sub',[shifted,logtotal])
 peak=node('head.conditional.peak','max',[conditional],{'axis':2,'keepdim':True});centered=node('head.conditional.centered','sub',[conditional,peak])
 directed=node('head.directed_scores','add',[skeleton,centered]);zero=node('head.absence_scores','zeros_like',[skeleton])
 raw=node('head.raw_tokens','concat',[zero,directed],{'axis':2});raw=node('head.raw_scores','reshape',[raw],{'shape':['B',n,n,4]})
 reversed_pairs=node('head.reverse_pairs','transpose',[raw],{'axes':[0,2,1,3]})
 reversed_classes=node('head.reverse_classes','take',[reversed_pairs],{'axis':3,'indices':[0,2,1,3]})
 summed=node('head.symmetric_sum','add',[raw,reversed_classes]);two=constant('symmetry.two','2')
 logits=node('logits','real_div',[summed,two])
 peak=node('output.peak','max',[logits],{'axis':3,'keepdim':True});shifted=node('output.shifted','sub',[logits,peak])
 exponents=node('output.exponentials','exp',[shifted]);total=node('output.denominator','sum',[exponents],{'axis':3,'keepdim':True})
 node('probabilities','real_div',[exponents,total])
 proof.append({'relation':'factorized head, symmetry and graph probabilities','argument':'The full positive-denominator log-softmax computation is retained. exp(o-max(o))=exp(o)/exp(max(o)) proves its log-normalization equals the actual definition. Reverse-pair indices and the [0,2,1,3] class involution are checked exactly, as is the final real probability normalization. Equality of the full score/probability tensors preserves every label comparison, including ties.'})
 validate_decoder(program['decoder'])
 proof.append({'relation':'complete graph tail','argument':'The five serialized function ASTs are independently matched to the approved actual source: classify, greedy cycle projection, DFS reachability, lexicographic topological order and one explicit rank-based DAG completion. The same finite loops and comparison branches receive equal probabilities. Equality covers tie, threshold and cycle-rejection branches; it does not identify ambiguous causal edges.'})
 if checked!=set(rows):raise ValueError('Unproved/orphan program computation')
 if len(network['encoder'])!=2 or len(network['context'])!=2 or len(network['head']['trunk'])!=1 or shapes['head.orientation.affine']!=['B',n*n,3] or shapes['head.skeleton.affine']!=['B',n*n,1]:raise ValueError('Incomplete actual blocks')
 return {'local_relations':proof,'checked_instructions':len(checked),'allowed_attention_keys':allowed,'positive_conditions':{'frozen_std_minimum':str(min(Q(x) for x in network['std'])),'attention_head_width':str(d),'attention_denominator_lower':'1','norm_radicand_lower':network['norm']['epsilon'],'orientation_log_denominator_lower':'1','output_denominator_lower':'1','symmetry_divisor':'2'}}


def certify(network,program):
 relations=local_relations(network,program)
 return {'schema':'ncd.actual-graph-program-fidelity.v1','status':'proved-scoped','network_sha256':hash_value(network),'program_sha256':hash_value(program),'checkpoint_sha256':network['checkpoint_sha256'],'nodes':program['nodes'],'domain':{'batch':'Every positive finite integer B','features':['B',program['nodes'],program['nodes'],24],'values':'All finite real feature coordinates, including diagonal tokens, clipping boundaries and label ties'},
 'checkpoint_binding_contract':'The local theorem concerns the supplied mathematical export; verify_checkpoint must independently bind it to the actual checkpoint before reporting actual-network fidelity',
 'error':{'mathematical_logits':'0','mathematical_probabilities':'0','pair_labels':'identical under the same lowest-index argmax rule','decoded_graph':'identical partially-directed prediction, rejection trace and explicit DAG completion'},
 'proof':relations,'interchange':mapping_certificate(program),'composition':'Topological induction over every typed SSA instruction, followed by structural equality of the complete finite graph tail',
 'unsupported_or_uncovered':['Raw statistical and do-response feature construction','Other actual graph architectures/checkpoints','Device rounding or bit-exact equivalence','True graph identification or an identified CPDAG','Shortest causal algorithm/MDL minimality','Unregistered physical graph sites or other mapping families'],
 'neural_network_suffix_retained':False,'all_boundaries_retained':True,'hardware_rounding_covered':False,'raw_frontend_included':False,'original_claim_closed':False,'original_objective_achieved':False}


def verify(network,program,certificate):
 # Re-check critical local coefficients, static masking, shapes, all ordinary
 # division/log domains and complete output/tail coverage independently.
 relations=local_relations(network,program)
 expected=certify(network,program)
 if certificate!=expected or certificate['proof']!=relations:raise ValueError('Graph-program proof was tampered or scope strengthened')
 return {'status':'verified','conclusion':'proved-scoped','nodes':program['nodes'],'checked_instructions':relations['checked_instructions'],'mathematical_error':'0','actual_checkpoint_binding_required':True,'all_finite_feature_coordinates_covered':True,'all_boundaries_retained':True,'raw_frontend_included':False,'hardware_rounding_covered':False,'true_graph_identification_proved':False,'declared_interchange_sites':14,'all_compatible_declared_site_subsets_proved_mathematically':True,'original_claim_closed':False,'original_objective_achieved':False}
