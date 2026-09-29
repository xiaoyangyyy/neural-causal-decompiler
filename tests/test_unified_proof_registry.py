from pathlib import Path
from copy import deepcopy
from fractions import Fraction as Q
import json,sys,os,ctypes
import pytest
from ncd.cdir import Node,algebraic_equivalence,canonical_expression
from ncd.exact_polynomial import prove_polynomial_equivalence,verify_polynomial_equivalence
from ncd.discovery_fidelity_proof import raw_feature,certify_discovery_box
from ncd.proof_intervals import Interval
from ncd.proof_registry import validate_config,evaluate,derive_ledger,verify_proof
from ncd.proof_process import run_isolated
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
@pytest.fixture(scope='module')
def config():return read(ROOT/'validation/original_proof_registry_protocol_v2r1.json')


def test_actual_binary_constants_cannot_be_erased_as_decimal_zero(config):
    job=next(j for j in config['jobs'] if j['id']=='binary_nonzero')
    a=Node.from_dict(job['left']);b=Node.from_dict(job['right'])
    assert not algebraic_equivalence(a,b)
    assert canonical_expression(a)=='32*x0'
    expected=Node('mul',(Node('constant',value=32),Node('var',index=0)))
    assert algebraic_equivalence(a,expected)


def test_actual_statistics_variance_floor_not_protected_cdir_division():
    data=[[Interval.point((1 if i%2 else -1)*Q(1,100000000)),Interval.point((1 if i%2 else -1)*Q(2,100000000))] for i in range(16)]
    ratio=raw_feature(data,2)
    assert ratio.lo<=0<=ratio.hi and ratio.hi-ratio.lo<Q(1,10**20)
    correlation=raw_feature(data,0)
    expected=Q(1,100000000)*Q(2,100000000)/Q(1e-6)**2
    assert correlation.lo<=expected<=correlation.hi


def test_polynomial_coefficients_are_independently_recomputed(config):
    job=next(j for j in config['jobs'] if j['id']=='binary_exact_equality')
    c=prove_polynomial_equivalence(job['left'],job['right'],1)
    assert verify_polynomial_equivalence(c)['conclusion']=='proved'
    forged=deepcopy(c);forged['left_normal_form'][0]['coefficient']='0'
    with pytest.raises(ValueError):verify_polynomial_equivalence(forged)


@pytest.mark.parametrize('change',['promotion','unknown_claim','source','snapshot','threads'])
def test_scope_and_verifier_environment_tampering_rejected(config,change):
    altered=deepcopy(config)
    if change=='promotion':altered['jobs'][0]['original_claim_closed']=True
    elif change=='unknown_claim':altered['jobs'][0]['related_claims']=['R100.invented']
    elif change=='source':altered['source_sha256']['ncd/cdir.py']='0'*64
    elif change=='snapshot':altered['historical_context']['snapshot_sha256'].clear()
    else:altered['historical_context']['inference_threads']=2
    with pytest.raises(ValueError):validate_config(altered)


def test_unsupported_operators_and_exhausted_bounds_stay_open(config):
    job=deepcopy(next(j for j in config['jobs'] if j['kind']=='exact_polynomial'))
    job['left']={'op':'tanh','args':[{'op':'var','index':0}]}
    result=evaluate(config,job)
    assert result['status']=='unresolved' and 'operator' in result['reason']
    job['left']={'op':'var','index':0}
    for _ in range(4):job['left']={'op':'square','args':[job['left']]}
    assert evaluate(config,job)['status']=='unresolved'


@pytest.mark.parametrize('forgery',['omitted_claim','false_proved'])
def test_original_claim_ledger_cannot_inherit_local_closure(config,forgery):
    ledger=read(ROOT/'runs/original_proof_milestone_v1/ledger.json')
    if forgery=='omitted_claim':ledger['claims'].pop()
    else:next(c for c in ledger['claims'] if c['id']=='R10.mechanism')['status']='proved'
    records=[{'id':j['id'],'result':{'status':'verified','verification':{'original_ledger':ledger}}} for j in config['jobs']]
    with pytest.raises(ValueError):derive_ledger(config,records)


def test_declared_job_omission_is_rejected_before_replay(tmp_path):
    source=ROOT/'runs/original_proof_registry_v2r1'
    for name in ['config.json','ledger.json','summary.json']:(tmp_path/name).write_bytes((source/name).read_bytes())
    summary=read(tmp_path/'summary.json');summary['records'].pop()
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='omitted proof jobs'):verify_proof(tmp_path)


@pytest.mark.skipif(os.name!='nt',reason='Windows job resource backend')
def test_memory_limit_reaches_real_descendant_not_only_launcher():
    code="import subprocess,sys; r=subprocess.run([sys.executable,'-c','print(\"entered\",flush=True); x=bytearray(256*1024*1024); print(\"escaped\")'],capture_output=True); print(r.returncode); print(r.stdout.decode()); print(r.stderr.decode())"
    result=run_isolated([sys.executable,'-c',code],ROOT,20,128*1024**2)
    assert result['exit_code']==0
    assert b'entered' in result['stdout'] and b'MemoryError' in result['stdout'] and b'escaped\r\n' not in result['stdout']
    assert result['resources']['descendants_included'] and result['resources']['total_processes']>=2
    # Windows counters are retained verbatim. A reported over-budget peak is
    # separately treated as unresolved by the scientific orchestration layer.


@pytest.mark.skipif(os.name!='nt',reason='Windows job resource backend')
def test_timeout_terminates_owned_descendant():
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
    result=run_isolated([sys.executable,'-c',code],ROOT,1,128*1024**2)
    assert result['resources']['timeout']
    pid=int(result['stdout'].strip())
    kernel=ctypes.WinDLL('kernel32');kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x100000,0,pid)
    if handle:
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        assert kernel.WaitForSingleObject(handle,0)==0
        kernel.CloseHandle(handle)
