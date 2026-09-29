from pathlib import Path
from datetime import datetime,timezone
import importlib,json,os,subprocess,sys,time
from ncd.io import digest,read_json,save_json
ROOT=Path(__file__).resolve().parents[1];PROTOCOL=ROOT/'validation/partial_mechanism_installed_protocol_v1.json';C=read_json(PROTOCOL);OUT=ROOT/C['output'];OUT.mkdir(exist_ok=True)
SNAP=ROOT/'validation/source_snapshot_059'
def check_files():
 for f,h in C['source_sha256'].items():
  if digest(ROOT/f)!=h:raise ValueError('Changed bound source '+f)
 for f,h in C['artifact_sha256'].items():
  if digest(ROOT/f)!=h:raise ValueError('Changed proof input '+f)
def active():
 query="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
 r=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,timeout=30,creationflags=0x08000000)
 if r.returncode:raise RuntimeError('Cannot observe heavy jobs')
 rows=json.loads(r.stdout.decode('utf-8-sig'));rows=rows if isinstance(rows,list) else [rows]
 tokens=('run_full_regression_v60_stage.py','run_piecewise_mechanism_stage_v1.py','original_confirmation_worker','replay_original_confirmation_installed.py')
 return [{'pid':row['ProcessId'],'command':row['CommandLine']} for row in rows if row['CommandLine'] and str(ROOT).lower() in row['CommandLine'].lower() and any(t in row['CommandLine'] for t in tokens)]
if '--worker' in sys.argv:
 check_files()
 import ncd,proof_workbench
 if not Path(proof_workbench.__file__).resolve().is_relative_to(ROOT/'validation/proof_extensions_env_v1') or ncd.__version__!='0.59.0':raise ValueError('Wrong historical environment')
 for f,h in C['historical_loaded_source_sha256'].items():
  if digest(importlib.import_module(f[:-3].replace('/','.').removesuffix('.__init__')).__file__)!=h or digest(SNAP/f)!=h:raise ValueError('Loaded/snapshot bytes changed')
 from ncd.frozen_mechanism_proof import export_mechanism
 from proof_workbench.piecewise_mechanism import verify_piecewise
 cert=read_json(ROOT/C['certificate']);original=read_json(ROOT/C['piecewise_protocol'])
 if export_mechanism(ROOT/original['checkpoint'])!=cert['network']:raise ValueError('Checkpoint differs')
 result=verify_piecewise(cert)
 if result!=read_json(ROOT/C['source_verification']):raise ValueError('Independent partial verification differs')
 save_json(OUT/'verification.json',result);print(json.dumps(result),flush=True)
else:
 from ncd.proof_process import run_isolated
 if (OUT/'launch.json').exists():raise FileExistsError('Retain previous replay stage')
 record={'schema':'ncd.partial-mechanism-installed-launch.v1','status':'queued','pid':os.getpid(),'queued_utc':datetime.now(timezone.utc).isoformat(),'whole_project_complete':False}
 save_json(OUT/'launch.json',record)
 while True:
  try:jobs=active();record['observation_error']=None
  except Exception as e:record['observation_error']=str(e);save_json(OUT/'launch.json',record);time.sleep(10);continue
  record['observed_live_heavy_jobs']=jobs;save_json(OUT/'launch.json',record)
  if not jobs:break
  time.sleep(10)
 check_files();record.update(status='running',started_utc=datetime.now(timezone.utc).isoformat());save_json(OUT/'launch.json',record)
 worker=ROOT/'validation/proof_extensions_env_v1/Scripts/python.exe'
 execution=run_isolated([str(worker),'-I','-u',str(Path(__file__).resolve()),'--worker'],SNAP,C['stage_seconds'],C['memory_bytes'])
 (OUT/'stdout.log').write_bytes(execution['stdout']);(OUT/'stderr.log').write_bytes(execution['stderr'])
 result=read_json(OUT/'verification.json') if (OUT/'verification.json').exists() else None
 ok=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources'].get('peak_job_memory_bytes',0)<=C['memory_bytes'] and result==read_json(ROOT/C['source_verification'])
 record.update(status='verified-partial' if ok else 'unresolved',exit_code=execution['exit_code'],resources=execution['resources'],verification=result,
 finished_utc=datetime.now(timezone.utc).isoformat(),original_requirement_closed=False,whole_project_complete=False)
 save_json(OUT/'launch.json',record)
