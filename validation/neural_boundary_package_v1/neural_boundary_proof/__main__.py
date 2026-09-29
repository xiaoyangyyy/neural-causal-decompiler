import argparse,importlib,json
from pathlib import Path
from ncd.io import read_json,save_json,digest
from .boundary import certify,verify

def check_config(path):
 c=read_json(path)
 if c['schema']!='ncd.neural-boundary-protocol.v1' or c['original_objective_achieved'] is not False:raise ValueError('Invalid protocol scope')
 dependency=read_json(c['realization_protocol'])
 if digest(c['realization_protocol'])!=c['realization_protocol_sha256']:raise ValueError('Realization protocol changed')
 required={'neural_boundary_proof/'+p.name for p in Path(importlib.import_module('neural_boundary_proof').__file__).parent.glob('*.py')}|set(dependency['source_sha256'])
 if set(c['source_sha256'])!=required:raise ValueError('Incomplete source binding')
 for f,h in c['source_sha256'].items():
  name=f[:-3].replace('/','.').removesuffix('.__init__')
  if digest(f)!=h or digest(importlib.import_module(name).__file__)!=h:raise ValueError('Source binding differs')
 for field in ('realization_certificate','program'):
  if digest(c[field])!=c[field+'_sha256']:raise ValueError('Dependency artifact changed')
 return c

def prove(config,resume=False):
 c=check_config(config);out=Path(c['output'])
 if out.exists() and any(out.iterdir()):
  if not resume:raise FileExistsError('Use --resume to independently verify preserved evidence')
  return verify_bundle(out/'manifest.json')
 out.mkdir(parents=True,exist_ok=True);save_json(out/'protocol.json',c)
 r,p=read_json(c['realization_certificate']),read_json(c['program']);cert=certify(r,p);verification=verify(cert,r,p)
 save_json(out/'certificate.json',cert);save_json(out/'verification.json',verification)
 m={'schema':'ncd.neural-boundary-bundle.v1','protocol':str(config),'protocol_sha256':digest(config),'files':{f:digest(out/f) for f in ('certificate.json','verification.json')},'original_objective_achieved':False}
 save_json(out/'manifest.json',m);return verification

def verify_bundle(path):
 path=Path(path);m=read_json(path)
 if m['schema']!='ncd.neural-boundary-bundle.v1' or m['original_objective_achieved'] is not False or digest(m['protocol'])!=m['protocol_sha256']:raise ValueError('Bundle scope or protocol changed')
 c=check_config(m['protocol'])
 if c!=read_json(path.parent/'protocol.json') or set(m['files'])!={'certificate.json','verification.json'} or any(digest(path.parent/f)!=h for f,h in m['files'].items()):raise ValueError('Bundle artifact changed')
 r=verify(read_json(path.parent/'certificate.json'),read_json(c['realization_certificate']),read_json(c['program']))
 if r!=read_json(path.parent/'verification.json'):raise ValueError('Verification differs')
 return r

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 a=s.add_parser('prove');a.add_argument('--config',required=True);a.add_argument('--resume',action='store_true')
 a=s.add_parser('verify-proof');a.add_argument('manifest')
 a=p.parse_args();r=prove(a.config,a.resume) if a.command=='prove' else verify_bundle(a.manifest)
 print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
