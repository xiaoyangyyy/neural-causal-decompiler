"""Source-bound proof bundles and standalone primitive execution."""
import argparse,importlib,json,time
from pathlib import Path
from fractions import Fraction as Q
from ncd.io import digest
from .compiler import compile_target,target,key
from .verification import certify,verify,hash_value
from .runtime import NumericProgram,interval_program
from .audit import audit

FILES={'certificate.json','program.json','verification.json','intervention_audit.json','interval_check.json'}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def check_config(path):
 c=read(path)
 if c['schema']!='ncd.neural-ssa-protocol.v1' or c['original_objective_achieved'] is not False:raise ValueError('Invalid scope')
 if c['limits']!={'instructions':100000,'seconds':600,'artifact_bytes':104857600,'training_threads':2}:raise ValueError('Changed resource limits')
 if len(c['jobs'])!=2 or len({j['id'] for j in c['jobs']})!=2:raise ValueError('Target enumeration changed')
 required={'neural_ssa_proof/'+p.name for p in Path(importlib.import_module('neural_ssa_proof').__file__).parent.glob('*.py')}
 required|={'ncd/model.py','ncd/proof_intervals.py','normalizer_proof/realization.py'}
 if set(c['source_sha256'])!=required:raise ValueError('Incomplete source binding')
 for p,h in c['source_sha256'].items():
  if digest(p)!=h:raise ValueError('Frozen source mismatch '+p)
  name=p[:-3].replace('/','.');name=name.removesuffix('.__init__')
  if digest(importlib.import_module(name).__file__)!=h:raise ValueError('Loaded source mismatch '+name)
 for j in c['jobs']:
  if digest(j['checkpoint'])!=j['checkpoint_sha256']:raise ValueError('Frozen weights mismatch')
 return c

def rational_fixture():return [[str(Q((i%3)-1,4)),str(Q((i%5)-2,8))] for i in range(16)]
def interval_check(p):
 r=interval_program(p,rational_fixture());r.pop('states')
 return {'input':rational_fixture(),'result':r,'scope':'one exact rational input under real semantics; unresolved labels retained'}

def prove(config,resume=False):
 c=check_config(config);out=Path(c['output']);start=time.monotonic()
 if out.exists() and any(out.iterdir()):
  if not resume:raise FileExistsError('Existing evidence requires --resume')
  return verify_bundle(out/'manifest.json')
 out.mkdir(parents=True,exist_ok=True);write(out/'protocol.json',c);jobs=[]
 for j in c['jobs']:
  if time.monotonic()-start>=c['limits']['seconds']:raise TimeoutError('Proof budget exhausted; existing artifacts retained')
  d=out/j['id'];d.mkdir();p=compile_target(target(j['checkpoint']),c['limits']['instructions']);cert=certify(j['checkpoint'],p)
  v=verify(cert,p);a=audit(j['checkpoint'],p);r=interval_check(p)
  for name,value in zip(('certificate.json','program.json','verification.json','intervention_audit.json','interval_check.json'),(cert,p,v,a,r)):write(d/name,value)
  jobs.append({'id':j['id'],'files':{name:digest(d/name) for name in sorted(FILES)}})
  write(out/'checkpoint.json',{'completed_jobs':jobs,'original_objective_achieved':False})
  if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>c['limits']['artifact_bytes']:raise RuntimeError('Artifact budget exhausted; retained evidence')
 manifest={'schema':'ncd.neural-ssa-bundle.v1','protocol':str(config),'protocol_sha256':digest(config),'jobs':jobs,'original_objective_achieved':False}
 write(out/'manifest.json',manifest)
 return {'status':'proved-awaiting-independent-bundle-replay','jobs':len(jobs),'original_objective_achieved':False}

def verify_bundle(path):
 path=Path(path);m=read(path)
 if m['schema']!='ncd.neural-ssa-bundle.v1' or m['original_objective_achieved'] is not False:raise ValueError('Changed manifest scope')
 if digest(m['protocol'])!=m['protocol_sha256']:raise ValueError('Changed protocol')
 c=check_config(m['protocol'])
 if c!=read(path.parent/'protocol.json') or [j['id'] for j in m['jobs']]!=[j['id'] for j in c['jobs']]:raise ValueError('Wrong frozen jobs')
 results=[]
 for j,expected in zip(m['jobs'],c['jobs']):
  d=path.parent/j['id']
  if set(j['files'])!=FILES or any(digest(d/name)!=h for name,h in j['files'].items()):raise ValueError('Changed proof artifact')
  cert,p=read(d/'certificate.json'),read(d/'program.json')
  if cert['checkpoint']!=expected['checkpoint']:raise ValueError('Target mismatch')
  v=verify(cert,p)
  if v!=read(d/'verification.json') or audit(expected['checkpoint'],p)!=read(d/'intervention_audit.json') or interval_check(p)!=read(d/'interval_check.json'):raise ValueError('Independent replay differs')
  results.append({'id':j['id'],**v})
 return {'status':'verified','jobs':results,'original_objective_achieved':False}

def main():
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('prove');p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true')
 p=sub.add_parser('verify-bundle');p.add_argument('manifest')
 p=sub.add_parser('verify-proof');p.add_argument('certificate');p.add_argument('--program',required=True)
 p=sub.add_parser('execute');p.add_argument('--program',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--interval',action='store_true')
 a=parser.parse_args()
 if a.command=='prove':r=prove(a.config,a.resume)
 elif a.command=='verify-bundle':r=verify_bundle(a.manifest)
 elif a.command=='verify-proof':r=verify(read(a.certificate),read(a.program))
 else:
  x=read(a.input);p=read(a.program)
  if a.interval:r=interval_program(p,x['data'],x.get('patches'));r.pop('states')
  else:
   r=NumericProgram(p).run(x['data'],x.get('patches'),trace=False);r['logits']=r['logits'].tolist()
  r['scope']='programme execution; source-bound fidelity requires a verified certificate';write(a.output,r)
 print(json.dumps(r,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
