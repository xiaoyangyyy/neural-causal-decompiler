from pathlib import Path
from datetime import datetime,timezone
from xml.etree import ElementTree as ET
import json,os,subprocess,sys,time
from ncd.io import digest,save_json,read_json
from ncd.proof_process import run_isolated
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'validation/full_regression_v60_protocol_v1.json'
C=read_json(P);OUT=ROOT/C['output'];OUT.mkdir(exist_ok=True)
LAUNCH=ROOT/'validation/full_regression_v60_launch';LAUNCH.mkdir(exist_ok=True)

def processes():
    query="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,creationflags=0x08000000,timeout=30)
    if r.returncode:raise RuntimeError('Process observation failed')
    rows=json.loads(r.stdout.decode('utf-8-sig'));rows=rows if isinstance(rows,list) else [rows]
    tokens=['original_confirmation_worker','replay_original_confirmation_installed.py','retry_original_confirmation_replay.py','run_piecewise_mechanism_stage_v1.py']
    return [{'pid':row['ProcessId'],'command':row['CommandLine']} for row in rows if row['CommandLine'] and str(ROOT).lower() in row['CommandLine'].lower() and any(v in row['CommandLine'] for v in tokens)]

def check():
    for name,sha in C['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Regression supervisor changed')
    for n,sha in C['test_modules'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Frozen regression test changed')
    if digest(ROOT/C['candidate_protocol'])!=C['candidate_protocol_sha256']:raise ValueError('Candidate protocol changed')
    from ncd.proof_registry import sources
    if sources()!=read_json(ROOT/C['candidate_protocol'])['source_sha256']:raise ValueError('Installed candidate source changed')

if '--worker' in sys.argv:
    check();temp=OUT/'temp_run_0000'
    if temp.exists() or not temp.resolve().is_relative_to(ROOT):raise ValueError('Unsafe or reused pytest temporary target')
    args=[sys.executable,'-I','-m','pytest',*[str(ROOT/n) for n in C['test_modules']],
        '--import-mode=importlib','-q','--basetemp='+str(temp),'--junitxml='+str(OUT/'pytest.xml')]
    with (OUT/'pytest.stdout.log').open('wb') as out,(OUT/'pytest.stderr.log').open('wb') as err:
        child=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=err,creationflags=0x08000000)
        budget_failure=None
        while child.poll() is None:
            if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>C['artifact_bytes']:
                budget_failure='artifact budget exhausted';subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);break
            time.sleep(.5)
        code=child.wait()
    tests=0;issues=1
    if (OUT/'pytest.xml').is_file():
        suite=ET.parse(OUT/'pytest.xml').getroot()[0];tests=int(suite.get('tests'));issues=sum(int(suite.get(k,'0')) for k in ['errors','failures','skipped'])
    accepted=code==0 and tests==C['expected_tests'] and issues==0 and budget_failure is None
    save_json(OUT/'test_result.json',{'status':'passed' if accepted else 'unresolved','tests':tests,'issues':issues,'exit_code':code,'budget_failure':budget_failure,'original_objective_achieved':False})
    sys.exit(0 if accepted else 1)
else:
    if (LAUNCH/'launch.json').exists() and '--resume-queue' not in sys.argv:raise FileExistsError('Retain existing regression stage')
    if '--resume-queue' in sys.argv and read_json(LAUNCH/'launch.json')['status']!='queued':raise ValueError('Only a retained pre-start queue can resume')
    record={'schema':'ncd.full-regression-launch.v60.v1','status':'queued','pid':os.getpid(),'queued_utc':datetime.now(timezone.utc).isoformat(),'whole_project_complete':False,'queue_attempt':1,'previous_queue_state':'attempt_0000_launch.json'}
    save_json(LAUNCH/'launch.json',record)
    while True:
        try:active=processes();record['observation_error']=None
        except Exception as error:record['observation_error']=str(error);save_json(LAUNCH/'launch.json',record);time.sleep(10);continue
        record.update(observed_live_jobs=active,last_observed_utc=datetime.now(timezone.utc).isoformat());save_json(LAUNCH/'launch.json',record)
        if not active:break
        time.sleep(10)
    check();record.update(status='running',started_utc=datetime.now(timezone.utc).isoformat());save_json(LAUNCH/'launch.json',record)
    execution=run_isolated([sys.executable,'-I','-u',str(Path(__file__).resolve()),'--worker'],ROOT,C['stage_seconds'],C['memory_bytes'])
    (OUT/'worker.stdout.log').write_bytes(execution['stdout']);(OUT/'worker.stderr.log').write_bytes(execution['stderr'])
    test_result=read_json(OUT/'test_result.json') if (OUT/'test_result.json').is_file() else {'status':'unresolved'}
    accepted=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources'].get('peak_job_memory_bytes',0)<=C['memory_bytes'] and test_result['status']=='passed'
    record.update(status='passed' if accepted else 'unresolved',exit_code=execution['exit_code'],resources=execution['resources'],finished_utc=datetime.now(timezone.utc).isoformat(),tests=test_result,whole_project_complete=False)
    save_json(LAUNCH/'launch.json',record)
