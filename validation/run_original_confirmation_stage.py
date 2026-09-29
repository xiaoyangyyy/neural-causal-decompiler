"""One 12-hour confirmation stage, generation then installed independent replay."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
LAUNCH=ROOT/'validation/original_confirmation_launch'
PROTOCOL=ROOT/'validation/original_confirmation_protocol.json'
INSTALLED=ROOT/'validation/wheel_v59_env/Scripts/python.exe'
SECONDS=43200

def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)

def main():
    LAUNCH.mkdir(exist_ok=True)
    previous=LAUNCH/'launch.json'
    if previous.exists():raise FileExistsError('Preserve prior stage launch; explicitly use a new launch directory for another stage')
    started=datetime.now(timezone.utc);deadline=time.monotonic()+SECONDS
    record={'schema':'ncd.original-confirmation-launch.v1','status':'starting','pid':os.getpid(),
      'started_utc':started.isoformat(),'deadline_utc':(started+timedelta(seconds=SECONDS)).isoformat(),
      'total_budget_seconds':SECONDS,'generation_budget_seconds':SECONDS//2,
      'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
      'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'release_replay_sha256':hashlib.sha256((ROOT/'validation/wheel_v59_run/status.json').read_bytes()).hexdigest(),
      'declared_worlds':300,'whole_project_complete':False}
    save(previous,record)
    study=ROOT/'runs/original_confirmation_v1';study.mkdir(exist_ok=True)
    # Make the pending 300 units visible during graph-teacher training.
    if not (study/'summary.json').exists():
        sys.path.insert(0,str(ROOT))
        from ncd.original_confirmation import _summary
        save(study/'summary.json',_summary(json.loads(PROTOCOL.read_text()),[],False))
    def phase(name,args,cwd,budget):
        record.update(status='running',phase=name);save(previous,record)
        with (LAUNCH/(name+'.stdout.log')).open('ab',buffering=0) as out,(LAUNCH/(name+'.stderr.log')).open('ab',buffering=0) as err:
            environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONNOUSERSITE='1')
            environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
            child=subprocess.Popen(args,cwd=cwd,env=environment,stdout=out,stderr=err,creationflags=0x08000000 if os.name=='nt' else 0)
            record['child_pid']=child.pid;save(previous,record)
            try:code=child.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                # This is only the supervised process tree launched above.
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                else:child.terminate()
                child.wait();code=-1
            record[name+'_exit_code']=code;record['child_pid']=None;save(previous,record)
            return code
    code=phase('generation',[sys.executable,'-u','-m','scripts.confirm_original','--protocol',str(PROTOCOL),'--resume','--seconds',str(SECONDS//2)],ROOT,SECONDS//2)
    remaining=deadline-time.monotonic()-5
    if code==0 and remaining>0:
        replay_code="import sys,json; from pathlib import Path; from ncd.original_confirmation import run_confirmation; r=run_confirmation(Path(sys.argv[1]),verify_only=True,seconds=float(sys.argv[2])); Path(sys.argv[3]).write_text(json.dumps(r,indent=2)+'\n'); print(r['state'],r['computed_worlds'],'/',r['declared_worlds'],'science_passed=',r['science_passed'])"
        phase('installed_replay',[str(INSTALLED),'-I','-u','-c',replay_code,str(PROTOCOL),str(remaining),str(LAUNCH/'replay_result.json')],ROOT/'validation/wheel_v59_run',remaining)
    summary=json.loads((study/'summary.json').read_text())
    record.update(status='stage-ended',finished_utc=datetime.now(timezone.utc).isoformat(),computed_worlds=summary['computed_worlds'],
        original_objective_achieved=False)
    if (LAUNCH/'replay_result.json').exists():
        r=json.loads((LAUNCH/'replay_result.json').read_text());record.update(replayed_worlds=r['computed_worlds'],science_passed=r['science_passed'])
    else:record.update(replayed_worlds=0,science_passed=False)
    save(previous,record)

if __name__=='__main__':main()
