from pathlib import Path
from datetime import datetime,timezone
from xml.etree import ElementTree as ET
import importlib,json,os,subprocess,sys,time
from ncd.io import digest,read_json,save_json
from ncd.proof_process import run_isolated
ROOT=Path(__file__).resolve().parents[1];C=read_json(ROOT/'validation/registry_v61r2_stage_protocol_v1.json');OUT=ROOT/C['output'];OUT.mkdir(exist_ok=True)
def check():
 import ncd
 if ncd.__version__!='0.61.0.dev2' or not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/wheel_v61r2_env'):raise ValueError('Wrong installed candidate')
 for f,h in C['source_sha256'].items():
  if digest(ROOT/f)!=h:raise ValueError('Changed stage source '+f)
 from ncd.proof_registry import validate_config
 if digest(ROOT/C['proof_protocol'])!=C['proof_protocol_sha256']:raise ValueError('Changed proof protocol')
 validate_config(read_json(ROOT/C['proof_protocol']))
def command(name,args,code=0):
 with (OUT/(name+'.stdout.log')).open('wb') as out,(OUT/(name+'.stderr.log')).open('wb') as err:
  child=subprocess.Popen([sys.executable,'-I','-m','ncd',*args],cwd=ROOT,stdout=out,stderr=err,creationflags=0x08000000)
  while child.poll() is None:
   directories=(OUT,ROOT/'runs/original_proof_registry_v3r2')
   if any(not d.resolve().is_relative_to(ROOT) for d in directories):raise ValueError('Unsafe artifact monitor target')
   if sum(f.stat().st_size for d in directories for f in d.rglob('*') if f.is_file())>C['artifact_bytes']:
    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait();raise RuntimeError('Artifact budget exhausted; partial evidence retained')
   time.sleep(.5)
  result=child

 if result.returncode!=code:raise ValueError(name+' CLI failed with '+str(result.returncode))
 text=(OUT/(name+'.stdout.log')).read_text(encoding='utf-8');index=text.find('{');return json.loads(text[index:])
def active():
 query="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
 p=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,timeout=30,creationflags=0x08000000)
 if p.returncode:raise RuntimeError('Heavy-process observation failed')
 rows=json.loads(p.stdout.decode('utf-8-sig'));rows=rows if isinstance(rows,list) else [rows]
 tokens=('run_partial_mechanism_installed_v1.py','run_full_regression_v60_stage.py','run_piecewise_mechanism_stage_v1.py','original_confirmation_worker')
 return [{'pid':r['ProcessId'],'command':r['CommandLine']} for r in rows if r['CommandLine'] and str(ROOT).lower() in r['CommandLine'].lower() and any(t in r['CommandLine'] for t in tokens)]
if '--worker' in sys.argv:
 check();p=ROOT/C['proof_protocol'];d=ROOT/'runs/original_proof_registry_v3r2'
 proof=command('prove',['prove','--config',str(p)])
 if proof['jobs']!=9 or proof['original_claim_counts']!={'proved':0,'refuted':2,'unresolved':36}:raise ValueError('Wrong scope or job count')
 print('Nine installed proof jobs generated and independently rechecked',flush=True)
 replay=command('verify',['verify-proof',str(d)])
 if replay!=proof:raise ValueError('Separate verify CLI differs')
 audit=command('audit',['audit-requirements',str(d),'--require-closed'],1)
 if len(audit['unresolved_claims'])!=36 or audit['overall_objective_achieved']:raise ValueError('Strict original closure audit failed')
 xml=OUT/'pytest.xml';result=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_unified_proof_registry_v3.py'),'--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=ROOT)
 suite=ET.parse(xml).getroot()[0]
 if result.returncode or int(suite.get('tests'))!=21 or any(int(suite.get(k,'0')) for k in ('failures','errors','skipped')):raise ValueError('Registry tests incomplete')
 claims=read_json(d/'ledger.json')['claims'];r2=next(c for c in claims if c['id']=='R2.variable_equivariance')
 if r2['status']!='unresolved' or [c['status'] for c in r2['resolved_subcontracts']]!=['proved-scoped','refuted-scoped']:raise ValueError('Score/label original scope changed')
 save_json(OUT/'result.json',{'status':'candidate-verified-original-open','jobs':9,'tests':21,'all_three_public_commands_verified':True,'strict_audit_exit_code':1,
 'proof':proof,'original_claim_counts':proof['original_claim_counts'],'score_label_scope_separated':True,'partial_proof_replayed_without_closure':True,'whole_project_complete':False})
else:
 if (OUT/'launch.json').exists():raise FileExistsError('Retain existing stage')
 record={'schema':'ncd.registry-v61-stage-launch.v1','status':'queued','pid':os.getpid(),'queued_utc':datetime.now(timezone.utc).isoformat(),'whole_project_complete':False};save_json(OUT/'launch.json',record)
 while True:
  try:jobs=active();record['observation_error']=None
  except Exception as e:record['observation_error']=str(e);save_json(OUT/'launch.json',record);time.sleep(10);continue
  record['observed_live_heavy_jobs']=jobs;save_json(OUT/'launch.json',record)
  if not jobs:break
  time.sleep(10)
 check();record.update(status='running',started_utc=datetime.now(timezone.utc).isoformat());save_json(OUT/'launch.json',record)
 remaining=(datetime.fromisoformat(C['original_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
 if remaining<=0:raise TimeoutError('Original first-stage deadline exhausted')
 execution=run_isolated([sys.executable,'-I','-u',str(Path(__file__).resolve()),'--worker'],ROOT,min(C['stage_seconds'],remaining),C['memory_bytes'])
 (OUT/'worker.stdout.log').write_bytes(execution['stdout']);(OUT/'worker.stderr.log').write_bytes(execution['stderr'])
 result=read_json(OUT/'result.json') if (OUT/'result.json').exists() else None
 passed=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources'].get('peak_job_memory_bytes',0)<=C['memory_bytes'] and result is not None
 record.update(status='verified' if passed else 'unresolved',exit_code=execution['exit_code'],resources=execution['resources'],result=result,finished_utc=datetime.now(timezone.utc).isoformat(),whole_project_complete=False)
 save_json(OUT/'launch.json',record)
