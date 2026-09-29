from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'validation/piecewise_mechanism_protocol_v1.json'
LAUNCH=ROOT/'validation/piecewise_mechanism_launch_v1'
LAUNCH.mkdir(exist_ok=True)
def save(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)
if '--worker' in sys.argv:
    sys.path.insert(0,str(ROOT))
    import torch
    from ncd.io import digest
    from ncd.frozen_mechanism_proof import export_mechanism
    from proof_workbench.piecewise_mechanism import certify_piecewise,verify_piecewise,program
    c=json.loads(PROTOCOL.read_text());torch.set_num_threads(c['threads'])
    if digest(ROOT/c['checkpoint'])!=c['checkpoint_sha256'] or any(digest(ROOT/p)!=v for p,v in c['source_sha256'].items()):raise ValueError('Changed frozen proof source or checkpoint')
    out=ROOT/c['output'];out.mkdir(exist_ok=True);save(out/'protocol.json',c)
    network=export_mechanism(ROOT/c['checkpoint'])
    result=certify_piecewise(network,c['domain'],c['epsilon'],c['max_cells'],c['search_seconds'],checkpoint_path=out/'checkpoint.json')
    save(out/'certificate.json',result)
    # Bind the target independently to the same actual checkpoint before replay.
    if export_mechanism(ROOT/c['checkpoint'])!=result['network']:raise ValueError('Weight export mismatch')
    verified=verify_piecewise(result);save(out/'verification.json',verified)
    if result['status']=='proved':save(out/'program.json',program(result))
    save(out/'acceptance.json',{'status':result['status'],'verification':verified,'certificate_sha256':digest(out/'certificate.json'),
        'prototype_installed_replay_complete':False,'original_requirement_closed':False,'overall_objective_achieved':False})
    print(result['status'],len(result['nodes']),verified,flush=True)
else:
    record={'schema':'ncd.piecewise-mechanism-launch.v1','status':'queued','pid':os.getpid(),
        'queued_utc':datetime.now(timezone.utc).isoformat(),'whole_project_complete':False}
    if (LAUNCH/'launch.json').exists():raise FileExistsError('Preserve previous stage')
    save(LAUNCH/'launch.json',record)
    while json.loads((ROOT/'validation/original_confirmation_launch/launch.json').read_text())['status']=='running':time.sleep(10)
    c=json.loads(PROTOCOL.read_text());started=time.monotonic()
    record.update(status='running',started_utc=datetime.now(timezone.utc).isoformat());save(LAUNCH/'launch.json',record)
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    with (LAUNCH/'stdout.log').open('ab',buffering=0) as out,(LAUNCH/'stderr.log').open('ab',buffering=0) as err:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=environment,stdout=out,stderr=err,creationflags=0x08000000)
        record['child_pid']=child.pid;save(LAUNCH/'launch.json',record)
        try:
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
                    ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
            psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
            while child.poll() is None:
                if time.monotonic()-started>c['stage_seconds']:raise TimeoutError('Stage deadline')
                counter=Counters();counter.cb=ctypes.sizeof(counter)
                if not psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(child._handle)),ctypes.byref(counter),counter.cb):raise RuntimeError('Cannot check memory budget')
                if counter.WorkingSetSize>c['memory_bytes']:raise RuntimeError('Memory budget exhausted')
                if sum(p.stat().st_size for p in (ROOT/c['output']).rglob('*') if p.is_file())>c['artifact_bytes']:raise RuntimeError('Artifact budget exhausted')
                time.sleep(.5)
            code=child.returncode
        except (TimeoutError,RuntimeError) as error:
            record['unresolved_reason']=str(error)
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);code=124
    record.update(status='stage-ended',exit_code=code,elapsed_seconds=time.monotonic()-started,child_pid=None,finished_utc=datetime.now(timezone.utc).isoformat())
    save(LAUNCH/'launch.json',record)
