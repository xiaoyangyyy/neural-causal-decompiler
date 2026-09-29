"""Torch-free elementary graph-program interpreter and independent type checker."""
from fractions import Fraction as Q
from .decoder import validate_decoder,graph_tail

def array_shape(value):
 if isinstance(value,list):
  if not value:raise ValueError('Empty tensor constant/index list')
  shapes=[array_shape(v) for v in value]
  if any(s!=shapes[0] for s in shapes):raise ValueError('Ragged tensor')
  return [len(value)]+shapes[0]
 if not isinstance(value,(str,int)) or isinstance(value,bool):raise ValueError('Exact scalar syntax')
 q=Q(value)
 if max(q.numerator.bit_length(),q.denominator.bit_length())>4096:raise ValueError('Exact coefficient budget')
 return []


def broadcast(a,b):
 result=[]
 for i in range(1,max(len(a),len(b))+1):
  x=a[-i] if i<=len(a) else 1;y=b[-i] if i<=len(b) else 1
  if x==y:result.append(x)
  elif x==1:result.append(y)
  elif y==1:result.append(x)
  else:raise ValueError('Tensor broadcast mismatch')
 return result[::-1]


def shape_product(shape):
 b=shape.count('B');product=1
 for x in shape:
  if x!='B':product*=x
 return b,product


def validate_program(program):
 required={'schema','network_sha256','checkpoint_sha256','architecture','nodes','width','features','instructions','outputs','decoder','semantics','raw_frontend_included','neural_network_suffix_retained','causal_identification_claimed','mdl_minimality','original_claim_closed'}
 if set(program)!=required or program['schema']!='ncd.explicit-graph-tensor-ssa.v1' or program['nodes'] not in (3,5,8) or type(program['nodes'])is not int or program['raw_frontend_included'] is not False or program['neural_network_suffix_retained'] is not False or program['causal_identification_claimed'] is not False or program['original_claim_closed'] is not False:raise ValueError('Changed program scope')
 if type(program['features'])is not int or program['features']!=24 or type(program['width'])is not int or not 4<=program['width']<=256 or program['width']%4:raise ValueError('Frozen feature/head dimensions')
 rows=program['instructions'];shapes={}
 if not isinstance(rows,list) or not 1<=len(rows)<=512:raise ValueError('SSA instruction budget')
 for row in rows:
  if set(row)!={'id','op','args','attrs'} or not isinstance(row['id'],str) or row['id'] in shapes or not isinstance(row['args'],list) or any(ref not in shapes for ref in row['args']) or not isinstance(row['attrs'],dict):raise ValueError('Invalid SSA dependency or shape')
  op,args,attrs=row['op'],[shapes[r] for r in row['args']],row['attrs']
  if op=='input':
   if row['id']!='features' or args or attrs!={'shape':['B',program['nodes'],program['nodes'],program['features']]}:raise ValueError('Feature input declaration')
   shape=attrs['shape']
  elif op=='constant':
   if args or set(attrs)!={'value'}:raise ValueError('Constant declaration')
   shape=array_shape(attrs['value'])
  elif op in ('add','sub','mul','real_div'):
   if len(args)!=2 or attrs:raise ValueError('Binary operator')
   shape=broadcast(*args)
  elif op in ('sqrt','tanh','exp','log','zeros_like'):
   if len(args)!=1 or attrs:raise ValueError('Unary operator')
   shape=args[0]
  elif op=='clip':
   if len(args)!=1 or set(attrs)!={'lower','upper'} or Q(attrs['lower'])>Q(attrs['upper']):raise ValueError('Clipping bounds')
   shape=args[0]
  elif op in ('mean','sum','max'):
   if len(args)!=1 or set(attrs)!={'axis','keepdim'} or type(attrs['axis'])is not int or not 0<=attrs['axis']<len(args[0]) or type(attrs['keepdim'])is not bool:raise ValueError('Reduction axis')
   shape=list(args[0]);axis=attrs['axis']
   if shape[axis]=='B':raise ValueError('Batch reduction outside this graph proof')
   if attrs['keepdim']:shape[axis]=1
   else:shape.pop(axis)
  elif op in ('reshape','broadcast'):
   if len(args)!=1 or set(attrs)!={'shape'}:raise ValueError('Shape operation')
   shape=attrs['shape']
   if not isinstance(shape,list) or not shape or any(x!='B' and (type(x)is not int or x<=0) for x in shape) or shape.count('B')!=1:raise ValueError('Symbolic batch shape')
   if op=='reshape' and shape_product(shape)!=shape_product(args[0]):raise ValueError('Reshape element mismatch')
   if op=='broadcast' and broadcast(args[0],shape)!=shape:raise ValueError('Broadcast target')
  elif op=='transpose':
   if len(args)!=1 or set(attrs)!={'axes'} or any(type(x)is not int for x in attrs['axes']) or sorted(attrs['axes'])!=list(range(len(args[0]))):raise ValueError('Tensor transpose')
   shape=[args[0][axis] for axis in attrs['axes']]
  elif op=='take':
   if len(args)!=1 or set(attrs)!={'axis','indices'} or type(attrs['axis'])is not int or not 0<=attrs['axis']<len(args[0]):raise ValueError('Tensor take axis')
   axis=attrs['axis'];dim=args[0][axis];indices=attrs['indices'];index_shape=array_shape(indices)
   def check(v):
    if isinstance(v,list):
     for x in v:check(x)
    elif type(v)is not int or type(dim)is not int or not 0<=v<dim:raise ValueError('Out-of-range static tensor index')
   check(indices);shape=args[0][:axis]+index_shape+args[0][axis+1:]
  elif op=='concat':
   if not args or set(attrs)!={'axis'} or type(attrs['axis'])is not int or not 0<=attrs['axis']<len(args[0]):raise ValueError('Tensor concatenation axis')
   axis=attrs['axis'];shape=list(args[0]);shape[axis]=0
   for x in args:
    if len(x)!=len(shape) or any(x[i]!=shape[i] for i in range(len(x)) if i!=axis) or type(x[axis])is not int:raise ValueError('Tensor concatenation dimensions')
    shape[axis]+=x[axis]
  elif op=='matmul':
   if len(args)!=2 or attrs or min(map(len,args))<2 or args[0][-1]!=args[1][-2]:raise ValueError('Finite product-sum dimensions')
   shape=broadcast(args[0][:-2],args[1][:-2])+[args[0][-2],args[1][-1]]
  else:raise ValueError('Unsupported primitive retained as unresolved: '+str(op))
  shapes[row['id']]=list(shape)
 expected=['B',program['nodes'],program['nodes'],4]
 if program['outputs']!={'logits':'logits','probabilities':'probabilities'} or any(shapes.get(ref)!=expected for ref in program['outputs'].values()):raise ValueError('Missing actual graph output')
 validate_decoder(program['decoder']);return shapes

class ExplicitGraphProgram:
 def __init__(self,program):
  self.program=program;self.shapes=validate_program(program)
  import numpy as np
  def values(v):return [values(x) for x in v] if isinstance(v,list) else float(Q(v))
  self.constants={r['id']:np.array(values(r['attrs']['value']),dtype=np.float64) for r in program['instructions'] if r['op']=='constant'}
 def run(self,features,trace=False,patches=None,decode=True):
  import numpy as np
  x=np.asarray(features,dtype=np.float64);n=self.program['nodes'];f=self.program['features']
  if x.ndim!=4 or x.shape[0]<1 or x.shape[1:]!=(n,n,f) or not np.isfinite(x).all():raise ValueError('Finite graph feature input')
  batch=x.shape[0];values={};execution=[];patches=patches or {}
  if any(name not in self.shapes or name in self.constants or name=='features' for name in patches):raise ValueError('Unknown or non-writable state')
  def shape(name):return tuple(batch if d=='B' else d for d in self.shapes[name])
  with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
   for row in self.program['instructions']:
    name,op,attrs=row['id'],row['op'],row['attrs'];a=[values[r] for r in row['args']]
    if name in patches:v=np.asarray(patches[name],dtype=np.float64).copy()
    elif op=='input':v=x.copy()
    elif op=='constant':v=self.constants[name]
    elif op=='add':v=a[0]+a[1]
    elif op=='sub':v=a[0]-a[1]
    elif op=='mul':v=a[0]*a[1]
    elif op=='real_div':
     if np.any(a[1]==0):raise ValueError('Ordinary division is undefined; no protected-CDIR substitution')
     v=a[0]/a[1]
    elif op=='sqrt':
     if np.any(a[0]<0):raise ValueError('Negative real square root')
     v=np.sqrt(a[0])
    elif op=='log':
     if np.any(a[0]<=0):raise ValueError('Nonpositive real logarithm')
     v=np.log(a[0])
    elif op=='exp':v=np.exp(a[0])
    elif op=='tanh':v=np.tanh(a[0])
    elif op=='zeros_like':v=np.zeros_like(a[0])
    elif op=='clip':v=np.clip(a[0],float(Q(attrs['lower'])),float(Q(attrs['upper'])))
    elif op in ('mean','sum','max'):v=getattr(np,op)(a[0],axis=attrs['axis'],keepdims=attrs['keepdim'])
    elif op=='reshape':v=a[0].reshape(shape(name))
    elif op=='broadcast':v=np.broadcast_to(a[0],shape(name))
    elif op=='transpose':v=a[0].transpose(attrs['axes'])
    elif op=='take':v=np.take(a[0],attrs['indices'],axis=attrs['axis'])
    elif op=='concat':v=np.concatenate(a,axis=attrs['axis'])
    elif op=='matmul':v=np.matmul(*a)
    else:raise ValueError('Unsupported operator')
    if v.shape!=shape(name) or not np.isfinite(v).all():raise ValueError('Floating evaluator capacity/patch shape failed; mathematical/device guarantee not issued')
    values[name]=v
    execution.append({'state':name,'primitive_evaluated':name not in patches,'intervened':name in patches,'value_written':True})
  probabilities=values['probabilities']
  return {'logits':values['logits'],'probabilities':probabilities,'labels':np.argmax(probabilities,axis=-1),
          'graphs':[graph_tail(p,self.program['decoder']) for p in probabilities] if decode else [],
          'states':values if trace else {},'execution':execution,'hardware_error_bound':None,'no_neural_framework_used':True}
