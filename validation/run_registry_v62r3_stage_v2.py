from pathlib import Path
from datetime import datetime,timezone
from xml.etree import ElementTree as ET
import json,os,subprocess,sys,time
from ncd.io import digest,read_json,save_json
from ncd.proof_process import run_isolated
ROOT=Path(__file__).resolve().parents[1]
C=read_json(ROOT/'validation/registry_v62r3_stage_protocol_v2.json')
OUT=ROOT/C['output'];OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_v62r3_stage';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
BUNDLE=ROOT/C['proof_output']

def check():
 import ncd
 if ncd.__version__!=C['version'] or not Path(ncd.__file__).resolve().is_relative_to(ROOT/C['environment']):raise ValueError('Wrong installed candidate')
 if digest(ROOT/C['proof_protocol'])!=C['proof_protocol_sha256']:raise ValueError('Changed proof protocol')
 for field in ('inputs_sha256','test_modules'):
  for name,sha in C[field].items():
   if digest(ROOT/name)!=sha:raise ValueError('Frozen source/test changed: '+name)
 preflight=read_json(ROOT/C['preflight_required'])
 if preflight['status']!='passed' or preflight['tests']!=46 or preflight['version']!=C['version']:raise ValueError('Installed preflight incomplete')
 if digest(ROOT/preflight['wheel'])!=preflight['wheel_sha256']:raise ValueError('Installed wheel changed')
 from ncd.proof_registry import validate_config
 validate_config(read_json(ROOT/C['proof_protocol']))

def active():
 query="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
 p=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,timeout=30,creationflags=0x08000000)
 if p.returncode:raise RuntimeError('Process observation failed')
 rows=json.loads(p.stdout.decode('utf-8-sig'));rows=rows if isinstance(rows,list) else [rows]
 tokens=('run_registry_v61','run_full_regression_v61','run_partial_mechanism_installed_v1.py','run_piecewise_mechanism','original_confirmation_worker','run_registry_v62_stage')
 return [{'pid':r['ProcessId'],'command':r['CommandLine']} for r in rows if r['CommandLine'] and str(ROOT).lower() in r['CommandLine'].lower() and any(t in r['CommandLine'] for t in tokens)]

def command(name,args,expected=0):
 with (OUT/(name+'.stdout.log')).open('wb') as out,(OUT/(name+'.stderr.log')).open('wb') as err:
  child=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=err,creationflags=0x08000000)
  while child.poll() is None:
   if sum(p.stat().st_size for d in (OUT,BUNDLE) if d.exists() for p in d.rglob('*') if p.is_file())>C['artifact_bytes']:
    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait();raise RuntimeError('Artifact budget exhausted; checkpoint retained')
   time.sleep(.5)
  if child.returncode!=expected:raise ValueError(name+' returned '+str(child.returncode))
 if name=='pytest':return None
 text=(OUT/(name+'.stdout.log')).read_text(encoding='utf-8');return json.loads(text[text.find('{'):])

if '--worker' in sys.argv:
 check()
 proof=command('prove',[sys.executable,'-I','-m','ncd','prove','--config',str(ROOT/C['proof_protocol'])])
 if proof['jobs']!=12 or proof['original_claim_counts']!={'proved':0,'refuted':2,'unresolved':36}:raise ValueError('Original scope/job count changed')
 save_json(OUT/'generation_checkpoint.json',{'status':'independently-rechecked-generation','proof':proof,'whole_project_complete':False})
 replay=command('verify',[sys.executable,'-I','-m','ncd','verify-proof',str(BUNDLE)])
 if replay!=proof:raise ValueError('Independent public replay differs')
 save_json(OUT/'verification_checkpoint.json',{'status':'independent-public-replay-verified','proof':replay,'whole_project_complete':False})
 audit=command('audit',[sys.executable,'-I','-m','ncd','audit-requirements',str(BUNDLE),'--require-closed'],1)
 if len(audit['unresolved_claims'])!=36 or len(audit['scoped_unresolved_contracts'])!=2 or audit['overall_objective_achieved']:raise ValueError('Strict audit/scope changed')
 command('pytest',[sys.executable,'-I','-m','pytest',*[str(ROOT/f) for f in C['test_modules']],'--import-mode=importlib','-q','--basetemp='+str(OUT/'temp_run_0000'),'--junitxml='+str(OUT/'pytest.xml')])
 suite=ET.parse(OUT/'pytest.xml').getroot()[0]
 if int(suite.get('tests'))!=C['expected_tests'] or any(int(suite.get(k,'0')) for k in ('errors','failures','skipped')):raise ValueError('Full regression incomplete')
 ledger=read_json(BUNDLE/'ledger.json');contracts=ledger['scoped_statistical_contracts']
 if len(contracts)!=7 or sum(c['status']=='proved' for c in contracts)!=5:raise ValueError('Recovery contract merge changed')
 save_json(OUT/'result.json',{'status':'candidate-verified-original-open','proof':proof,'tests':C['expected_tests'],'baseline_tests':249,'native_registry_tests':35,'all_three_public_commands_verified':True,'strict_audit_exit_code':1,'scoped_statistical_contracts':{'proved':5,'unresolved':2},'failed_prior_regression_retained':True,'whole_project_complete':False})
else:
 launch=OUT/'launch.json'
 if launch.exists():raise FileExistsError('Retain previous stage')
 record={'schema':'ncd.registry-v62-stage-launch.v1','status':'queued','pid':os.getpid(),'queued_utc':datetime.now(timezone.utc).isoformat(),'whole_project_complete':False};save_json(launch,record)
 try:
  while True:
   jobs=active();record['observed_live_heavy_jobs']=jobs;save_json(launch,record)
   if not jobs:break
   if datetime.now(timezone.utc)>=datetime.fromisoformat(C['original_deadline_utc']):raise TimeoutError('Original first-stage deadline exhausted while queued')
   time.sleep(10)
  check();record.update(status='running',started_utc=datetime.now(timezone.utc).isoformat());save_json(launch,record)
  remaining=(datetime.fromisoformat(C['original_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
  if remaining<=0:raise TimeoutError('Original first-stage deadline exhausted')
  execution=run_isolated([sys.executable,'-I','-u',str(Path(__file__).resolve()),'--worker'],ROOT,min(C['stage_seconds'],remaining),C['memory_bytes'])
  (OUT/'worker.stdout.log').write_bytes(execution['stdout']);(OUT/'worker.stderr.log').write_bytes(execution['stderr'])
  result=read_json(OUT/'result.json') if (OUT/'result.json').is_file() else None
  passed=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources'].get('peak_job_memory_bytes',0)<=C['memory_bytes'] and execution['resources']['active_processes_on_return']==0 and result is not None
  record.update(status='passed' if passed else 'unresolved',exit_code=execution['exit_code'],resources=execution['resources'],result=result)
 except Exception as e:record.update(status='unresolved',reason=str(e))
 record.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_project_complete=False);save_json(launch,record)
 sys.exit(0 if record['status']=='passed' else 1)
