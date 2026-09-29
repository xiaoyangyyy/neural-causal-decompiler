"""Lower the complete known frozen graph computation to elementary tensor SSA."""
from fractions import Fraction as Q
from copy import deepcopy
import hashlib,json
from .decoder import PRIMARY_FUNCTION_TEXT


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest_value(v):return hashlib.sha256(canonical(v).encode()).hexdigest()

class Builder:
 def __init__(self):self.nodes=[];self.names=set()
 def emit(self,name,op,*args,**attrs):
  if name in self.names:raise ValueError('Duplicate SSA state')
  self.nodes.append({'id':name,'op':op,'args':list(args),'attrs':attrs});self.names.add(name);return name
 def constant(self,name,value):return self.emit(name,'constant',value=deepcopy(value))
 def affine(self,name,x,layer,activation=True):
  weights=self.constant(name+'.weight',layer['weights']);bias=self.constant(name+'.bias',layer['bias'])
  transpose=self.emit(name+'.weight_transpose','transpose',weights,axes=[1,0])
  product=self.emit(name+'.product','matmul',x,transpose)
  result=self.emit(name+'.affine','add',product,bias)
  return self.emit(name+'.tanh','tanh',result) if activation else result


def compile_graph(network,nodes):
 if type(nodes)is not int or nodes not in (3,5,8):raise ValueError('Frozen 3/5/8-node protocol')
 if network['schema']!='ncd.frozen-graph-math.v1' or network['architecture'] not in ('active_input_observational_control_v1','active_input_intervention_v1'):raise ValueError('Unsupported actual architecture')
 w=network['width'];f=len(network['mean']);h=network['attention']['heads'];n=nodes;k=n*(n-1)
 if type(w)is not int or not 4<=w<=256 or h!=4 or w%h or f!=24 or any(Q(v)<=0 for v in network['std']) or Q(network['norm']['epsilon'])<=0:raise ValueError('Frozen dimension/positive normalization contract')
 b=Builder();x=b.emit('features','input',shape=['B',n,n,f])
 x=b.emit('feature_tokens','reshape',x,shape=['B',n*n,f])
 mean=b.constant('normalization.mean',network['mean']);std=b.constant('normalization.std',network['std'])
 centered=b.emit('normalization.centered','sub',x,mean);normalized=b.emit('normalization.scaled','real_div',centered,std)
 z=b.emit('normalization.clipped','clip',normalized,lower='-20',upper='20')
 for index,layer in enumerate(network['encoder']):z=b.affine('encoder.'+str(index),z,layer)
 encoded=z
 active=[i*n+j for i in range(n) for j in range(n) if i!=j]
 outgoing=[[i*n+j for j in range(n) if j!=i] for i in range(n)]
 incoming=[[j*n+i for j in range(n) if j!=i] for i in range(n)]
 contexts=[]
 for name,indices in (('outgoing',outgoing),('incoming',incoming)):
  selected=b.emit('context.'+name+'.selected','take',encoded,axis=1,indices=indices)
  reduced=b.emit('context.'+name,'mean',selected,axis=2,keepdim=False)
  contexts.append(reduced)
 global_tokens=b.emit('context.global.selected','take',encoded,axis=1,indices=active)
 global_mean=b.emit('context.global','mean',global_tokens,axis=1,keepdim=False)
 pieces=[encoded]
 for role,shape in (('source',['B',n,1,w]),('target',['B',1,n,w])):
  for name,ref in zip(('outgoing','incoming'),contexts):
   ref=b.emit('context.'+role+'.'+name+'.reshape','reshape',ref,shape=shape)
   ref=b.emit('context.'+role+'.'+name+'.broadcast','broadcast',ref,shape=['B',n,n,w])
   pieces.append(b.emit('context.'+role+'.'+name+'.tokens','reshape',ref,shape=['B',n*n,w]))
 ref=b.emit('context.global.reshape','reshape',global_mean,shape=['B',1,w])
 pieces.append(b.emit('context.global.broadcast','broadcast',ref,shape=['B',n*n,w]))
 z=b.emit('context.concatenated','concat',*pieces,axis=2)
 for index,layer in enumerate(network['context']):z=b.affine('context.layer.'+str(index),z,layer)
 enriched=b.emit('context.enriched','add',encoded,z)
 a=network['attention'];projected=b.affine('attention.projection',enriched,{'weights':a['in_weights'],'bias':a['in_bias']},False)
 parts={};d=w//h
 for role,index in (('query',0),('key',1),('value',2)):
  v=b.emit('attention.'+role+'.selected','take',projected,axis=2,indices=list(range(index*w,(index+1)*w)))
  v=b.emit('attention.'+role+'.split','reshape',v,shape=['B',n*n,h,d])
  v=b.emit('attention.'+role+'.heads','transpose',v,axes=[0,2,1,3])
  if role!='query':v=b.emit('attention.'+role+'.allowed','take',v,axis=2,indices=active)
  parts[role]=v
 keys=b.emit('attention.key.transpose','transpose',parts['key'],axes=[0,1,3,2])
 scores=b.emit('attention.dot','matmul',parts['query'],keys)
 dimension=b.constant('attention.head_width',str(d));scale=b.emit('attention.scale','sqrt',dimension)
 scores=b.emit('attention.scores','real_div',scores,scale)
 peak=b.emit('attention.peak','max',scores,axis=3,keepdim=True)
 shifted=b.emit('attention.centered','sub',scores,peak)
 exps=b.emit('attention.exponentials','exp',shifted)
 total=b.emit('attention.denominator','sum',exps,axis=3,keepdim=True)
 probabilities=b.emit('attention.probabilities','real_div',exps,total)
 attended=b.emit('attention.weighted_values','matmul',probabilities,parts['value'])
 attended=b.emit('attention.heads_join','transpose',attended,axes=[0,2,1,3])
 attended=b.emit('attention.join','reshape',attended,shape=['B',n*n,w])
 attended=b.affine('attention.output',attended,a['out'],False)
 residual=b.emit('norm.input','add',enriched,attended)
 mean=b.emit('norm.mean','mean',residual,axis=2,keepdim=True)
 centered=b.emit('norm.centered','sub',residual,mean)
 squares=b.emit('norm.squares','mul',centered,centered)
 variance=b.emit('norm.variance','mean',squares,axis=2,keepdim=True)
 epsilon=b.constant('norm.epsilon',network['norm']['epsilon'])
 radicand=b.emit('norm.radicand','add',variance,epsilon)
 scale=b.emit('norm.scale','sqrt',radicand)
 normalized=b.emit('norm.standardized','real_div',centered,scale)
 weight=b.constant('norm.weight',network['norm']['weights']);bias=b.constant('norm.bias',network['norm']['bias'])
 weighted=b.emit('norm.weighted','mul',normalized,weight)
 hidden=b.emit('norm.output','add',weighted,bias)
 z=hidden
 for index,layer in enumerate(network['head']['trunk']):z=b.affine('head.trunk.'+str(index),z,layer)
 skeleton=b.affine('head.skeleton',z,network['head']['skeleton'],False)
 orientation=b.affine('head.orientation',z,network['head']['orientation'],False)
 peak=b.emit('head.orientation.peak','max',orientation,axis=2,keepdim=True)
 shifted=b.emit('head.orientation.shifted','sub',orientation,peak)
 exps=b.emit('head.orientation.exponentials','exp',shifted)
 total=b.emit('head.orientation.denominator','sum',exps,axis=2,keepdim=True)
 logtotal=b.emit('head.orientation.log_denominator','log',total)
 conditional=b.emit('head.log_softmax','sub',shifted,logtotal)
 peak=b.emit('head.conditional.peak','max',conditional,axis=2,keepdim=True)
 centered=b.emit('head.conditional.centered','sub',conditional,peak)
 directed=b.emit('head.directed_scores','add',skeleton,centered)
 zero=b.emit('head.absence_scores','zeros_like',skeleton)
 raw=b.emit('head.raw_tokens','concat',zero,directed,axis=2)
 raw=b.emit('head.raw_scores','reshape',raw,shape=['B',n,n,4])
 reverse=b.emit('head.reverse_pairs','transpose',raw,axes=[0,2,1,3])
 reverse=b.emit('head.reverse_classes','take',reverse,axis=3,indices=[0,2,1,3])
 total=b.emit('head.symmetric_sum','add',raw,reverse)
 two=b.constant('symmetry.two','2');logits=b.emit('logits','real_div',total,two)
 peak=b.emit('output.peak','max',logits,axis=3,keepdim=True)
 shifted=b.emit('output.shifted','sub',logits,peak)
 exps=b.emit('output.exponentials','exp',shifted)
 total=b.emit('output.denominator','sum',exps,axis=3,keepdim=True)
 probabilities=b.emit('probabilities','real_div',exps,total)
 return {'schema':'ncd.explicit-graph-tensor-ssa.v1','network_sha256':digest_value(network),'checkpoint_sha256':network['checkpoint_sha256'],'architecture':network['architecture'],'nodes':n,'width':w,'features':f,'instructions':b.nodes,'outputs':{'logits':logits,'probabilities':probabilities},'decoder':deepcopy(PRIMARY_FUNCTION_TEXT),'semantics':'Exact real elementary tensor operations and exact frozen coefficients; floating runtime is diagnostic only','raw_frontend_included':False,'neural_network_suffix_retained':False,'causal_identification_claimed':False,'mdl_minimality':'unresolved; parameter-bearing explicit computation is not a shortest causal algorithm','original_claim_closed':False}
