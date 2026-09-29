from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
LAUNCH=ROOT/'validation/original_confirmation_launch'
def save(path,value):
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
previous=json.loads((LAUNCH/'attempt_0000_launch.json').read_text())
if hashlib.sha256((ROOT/'validation/original_confirmation_protocol.json').read_bytes()).hexdigest()!=previous['protocol_sha256']:
    raise RuntimeError('Frozen protocol changed')
if json.loads((ROOT/'runs/original_confirmation_v1/summary.json').read_text())['computed_worlds']!=300:raise RuntimeError('Generation incomplete')
record=dict(previous,status='running',phase='installed_replay_retry',pid=os.getpid(),child_pid=None,
    retry_started_utc=datetime.now(timezone.utc).isoformat(),retry_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    replay_script_sha256=hashlib.sha256((ROOT/'validation/replay_original_confirmation_installed.py').read_bytes()).hexdigest())
record.pop('finished_utc',None);record['installed_replay_exit_code']=None
save(LAUNCH/'launch.json',record);save(LAUNCH/'replay_retry_0001.json',record)
environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONNOUSERSITE='1')
environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
remaining=(datetime.fromisoformat(record['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
if remaining<=0:raise TimeoutError('Original deadline exhausted')
with (LAUNCH/'replay_retry_0001.stdout.log').open('ab',buffering=0) as out,(LAUNCH/'replay_retry_0001.stderr.log').open('ab',buffering=0) as err:
    child=subprocess.Popen([str(ROOT/'validation/wheel_v59_env/Scripts/python.exe'),'-I','-u',str(ROOT/'validation/replay_original_confirmation_installed.py'),record['deadline_utc']],
        cwd=ROOT/'validation/wheel_v59_run',env=environment,stdout=out,stderr=err,creationflags=0x08000000)
    record['child_pid']=child.pid;save(LAUNCH/'launch.json',record);save(LAUNCH/'replay_retry_0001.json',record)
    try:code=child.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);code=124
record.update(status='stage-ended',child_pid=None,installed_replay_exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat())
if (LAUNCH/'replay_result.json').exists():
    r=json.loads((LAUNCH/'replay_result.json').read_text());record.update(replayed_worlds=r['computed_worlds'],science_passed=r['science_passed'])
save(LAUNCH/'launch.json',record);save(LAUNCH/'replay_retry_0001.json',record)
