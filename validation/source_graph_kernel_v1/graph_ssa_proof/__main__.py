"""Portable proof generation and independent replay of actual graph programs."""
from pathlib import Path
from time import monotonic
import argparse,hashlib,json,importlib,shutil
from .binding import verify_checkpoint,load_primary_model,APPROVED_PRIMARY_SOURCES,APPROVED_CHECKPOINTS
from .mapping import validate_primary_mapping
from .compiler import compile_graph
from .verification import certify,verify

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_suffix(p.suffix+'.tmp')
 temporary.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(p)
def loaded_sources():
 folder=Path(importlib.import_module('graph_ssa_proof').__file__).resolve().parent
 return {'graph_ssa_proof/'+p.name:digest(p) for p in sorted(folder.glob('*.py'))}
def contained(root,name):
 p=(root/name).resolve()
 if not p.is_relative_to(root.resolve()) or p==root.resolve():raise ValueError('Unsafe proof artifact path')
 return p


def check_config(path):
 p=read(path)
 fields={'schema','output','target_manifest','target_manifest_sha256','source_sha256','primary_source_sha256','stage_seconds','artifact_bytes','original_objective_achieved'}
 if set(p)!=fields or p['schema']!='ncd.actual-graph-program-protocol.v1' or p['source_sha256']!=loaded_sources() or p['primary_source_sha256']!=APPROVED_PRIMARY_SOURCES or p['original_objective_achieved'] is not False:raise ValueError('Protocol/source/scope mismatch')
 if type(p['stage_seconds'])is not int or not 0<p['stage_seconds']<=43200 or type(p['artifact_bytes'])is not int or not 0<p['artifact_bytes']<=8*1024**3:raise ValueError('Frozen proof resource contract')
 root=Path(path).resolve().parent.parent;manifest=contained(root,p['target_manifest'])
 if digest(manifest)!=p['target_manifest_sha256']:raise ValueError('Target manifest changed')
 target=read(manifest);rows=target['records']
 if target['ground_truth_read'] is not False or target['original_claim_closed'] is not False or target['whole_project_complete'] is not False or len(rows)!=4 or {r['id']:r['checkpoint_sha256'] for r in rows}!=APPROVED_CHECKPOINTS:raise ValueError('Actual checkpoint family changed')
 for row in rows:
  if row['nodes']!=[3,5,8] or row['features']!=24 or row['width']!=48 or row['actual_torch_version']!='2.12.1+cpu':raise ValueError('Declared actual model/node scope changed')
  if digest(contained(root,row['checkpoint']))!=row['checkpoint_sha256'] or digest(contained(root,row['export']))!=row['export_sha256']:raise ValueError('Actual checkpoint/export changed')
 return p,target,root


def prove(config_path,resume=False):
 config,target,root=check_config(config_path);out=contained(root,config['output']);out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()) and not resume:raise FileExistsError('Retain evidence; use --resume for frozen-contract replay')
 protocol=out/'protocol.json'
 if protocol.is_file() and read(protocol)!=config:raise ValueError('Changed checkpointed proof protocol')
 if resume and (out/'manifest.json').is_file():
  verify_bundle(out/'manifest.json')
  history=out/'history';history.mkdir(exist_ok=True)
  archive=history/('revision_%04d'%len(list(history.iterdir())));archive.mkdir()
  for old in list(out.iterdir()):
   if old==history:continue
   if old.is_dir():shutil.copytree(old,archive/old.name)
   else:shutil.copy2(old,archive/('retained_manifest.json' if old.name=='manifest.json' else old.name))
 save(protocol,config)
 original_target=contained(root,config['target_manifest']);copied_target=out/'target_manifest.json'
 if copied_target.is_file() and digest(copied_target)!=config['target_manifest_sha256']:raise ValueError('Retained target registry changed')
 if not copied_target.is_file():shutil.copy2(original_target,copied_target)
 for name,sha in APPROVED_PRIMARY_SOURCES.items():
  source=root/'ncd'/name
  if digest(source)!=sha:raise ValueError('Actual primary source changed')
  destination=out/'primary_sources'/name;destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)
 started=monotonic();cases=[];pending=[]
 for row in target['records']:
  directory=out/'targets'/row['id'];directory.mkdir(parents=True,exist_ok=True)
  for old,new in ((row['checkpoint'],'checkpoint.pt'),(row['export'],'network.json')):
   destination=directory/new
   if destination.exists() and digest(destination)!=digest(root/old):raise ValueError('Retained actual target changed')
   if not destination.exists():shutil.copy2(root/old,destination)
  network=read(directory/'network.json');binding=verify_checkpoint(network,directory/'checkpoint.pt',out/'primary_sources');save(directory/'binding.json',binding)
  model=load_primary_model(directory/'checkpoint.pt',out/'primary_sources')
  for n in row['nodes']:
   case=directory/('n'+str(n));case.mkdir(exist_ok=True)
   if all((case/name).is_file() for name in ('program.json','certificate.json','verification.json')):
    retained_program=read(case/'program.json')
    validate_primary_mapping(model,retained_program)
    if retained_program['nodes']!=n or verify(network,retained_program,read(case/'certificate.json'))!=read(case/'verification.json'):raise ValueError('Retained proved case changed')
    cases.append({'target':row['id'],'nodes':n,'directory':case.relative_to(out).as_posix()})
    continue
   if monotonic()-started>config['stage_seconds'] or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>config['artifact_bytes']:
    pending.append({'target':row['id'],'nodes':n,'status':'unresolved','reason':'Frozen stage/artifact budget exhausted'});continue
   program=compile_graph(network,n);validate_primary_mapping(model,program);certificate=certify(network,program);verification=verify(network,program,certificate)
   if (case/'program.json').exists() and read(case/'program.json')!=program:raise ValueError('Resumed program differs')
   save(case/'program.json',program);save(case/'certificate.json',certificate);save(case/'verification.json',verification)
   cases.append({'target':row['id'],'nodes':n,'directory':case.relative_to(out).as_posix()})
   save(out/'progress.json',{'proved_feature_cases':len(cases),'declared_cases':12,'unresolved':pending,'original_objective_achieved':False})
 save(out/'cases.json',{'cases':cases,'unresolved':pending})
 files={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
 save(out/'manifest.json',{'schema':'ncd.actual-graph-program-bundle.v1','files':files,'source_sha256':loaded_sources(),'original_objective_achieved':False})
 return verify_bundle(out/'manifest.json')


def verify_bundle(path):
 path=Path(path).resolve();out=path.parent;m=read(path)
 if set(m)!={'schema','files','source_sha256','original_objective_achieved'} or m['schema']!='ncd.actual-graph-program-bundle.v1' or m['source_sha256']!=loaded_sources() or m['original_objective_achieved'] is not False:raise ValueError('Changed proof bundle scope/source')
 actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
 if set(m['files'])!=actual or any(digest(contained(out,name))!=sha for name,sha in m['files'].items()):raise ValueError('Incomplete or altered portable proof')
 config=read(out/'protocol.json');target=read(out/'target_manifest.json')
 fields={'schema','output','target_manifest','target_manifest_sha256','source_sha256','primary_source_sha256','stage_seconds','artifact_bytes','original_objective_achieved'}
 if set(config)!=fields or type(config['stage_seconds'])is not int or not 0<config['stage_seconds']<=43200 or type(config['artifact_bytes'])is not int or not 0<config['artifact_bytes']<=8*1024**3:raise ValueError('Altered portable resource/scope protocol')
 declared={(r['id'],n) for r in target['records'] for n in r['nodes']};index=read(out/'cases.json')
 if config['source_sha256']!=loaded_sources() or config['primary_source_sha256']!=APPROVED_PRIMARY_SOURCES or config['original_objective_achieved'] is not False or {r['id']:r['checkpoint_sha256'] for r in target['records']}!=APPROVED_CHECKPOINTS or declared!={(t,n) for t in APPROVED_CHECKPOINTS for n in (3,5,8)}:raise ValueError('Frozen portable family changed')
 if digest(out/'target_manifest.json')!=config['target_manifest_sha256']:raise ValueError('Copied frozen target manifest changed')
 if config['schema']!='ncd.actual-graph-program-protocol.v1' or target['schema']!='ncd.actual-graph-program-targets.v1' or target['ground_truth_read'] is not False or target['original_claim_closed'] is not False or target['whole_project_complete'] is not False:
  raise ValueError('Changed target metadata or information access')
 if len(target['records'])!=4 or any(r['nodes']!=[3,5,8] or r['features']!=24 or r['width']!=48 or r['actual_torch_version']!='2.12.1+cpu' for r in target['records']):raise ValueError('Changed actual node/feature scope')
 cases=index['cases'];pending=index['unresolved']
 if {(c['target'],c['nodes']) for c in cases}|{(c['target'],c['nodes']) for c in pending}!=declared or len(cases)+len(pending)!=12:raise ValueError('Omitted/duplicated declared graph domain')
 results=[]
 for row in target['records']:
  folder=contained(out,'targets/'+row['id']);network=read(folder/'network.json')
  if digest(folder/'checkpoint.pt')!=row['checkpoint_sha256'] or digest(folder/'network.json')!=row['export_sha256']:raise ValueError('Portable actual target differs')
  binding=verify_checkpoint(network,folder/'checkpoint.pt',out/'primary_sources')
  model=load_primary_model(folder/'checkpoint.pt',out/'primary_sources')
  if binding!=read(folder/'binding.json'):raise ValueError('Independent actual parameter binding differs')
  for c in [c for c in cases if c['target']==row['id']]:
   expected='targets/'+row['id']+'/n'+str(c['nodes'])
   if c['directory']!=expected:raise ValueError('Case points to another target/domain')
   directory=contained(out,c['directory']);program=read(directory/'program.json')
   if program['nodes']!=c['nodes']:raise ValueError('Case node scope differs')
   validate_primary_mapping(model,program)
   result=verify(network,program,read(directory/'certificate.json'))
   if result!=read(directory/'verification.json'):raise ValueError('Independent local proof differs')
   results.append({'target':row['id'],'nodes':c['nodes'],**result})
 for c in pending:
  if c.get('status')!='unresolved' or c.get('reason')!='Frozen stage/artifact budget exhausted':raise ValueError('Invalid retained unresolved proof case')
 return {'status':'verified' if not pending else 'verified-partial','proved_feature_domain_cases':len(results),'unresolved_cases':pending,'frozen_networks':4,'checkpoint_bindings_independently_verified':True,'physical_module_map_independently_checked':True,'results':results,'math_score_probability_error':'0' if not pending else 'unresolved-for-pending-domains','raw_frontend_included':False,'hardware_rounding_covered':False,'true_graph_identification_proved':False,'mdl_minimality_proved':False,'original_claim_closed':False,'original_objective_achieved':False}


def main():
 parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
 p=sub.add_parser('prove');p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true')
 p=sub.add_parser('verify-proof');p.add_argument('manifest')
 args=parser.parse_args();result=prove(args.config,args.resume) if args.command=='prove' else verify_bundle(args.manifest)
 print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
