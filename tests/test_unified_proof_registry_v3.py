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
def config():return read(ROOT/'validation/original_proof_registry_protocol_v3r2.json')


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
    source=ROOT/'runs/original_proof_registry_v3r2'
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


def partial_bundle(tmp_path,monkeypatch,changed=False):
    import ncd.proof_registry as api
    from ncd.io import digest
    job={'id':'partial','kind':'pending_piecewise','related_claims':['R10.mechanism'],'original_claim_closed':False}
    result={'status':'unresolved','verification':{'status':'verified','conclusion':'unresolved','proved_leaves':378,'unresolved_cells':6},'original_claim_closed':False,'artifact_sha256':'a'*64}
    cfg={'schema':'ncd.unified-proof-plan.v3','jobs':[job]};protocol=tmp_path/'external.json';protocol.write_text(json.dumps(cfg),encoding='utf-8')
    record={'id':'partial','job':job,'result':result};(tmp_path/'partial').mkdir();(tmp_path/'partial/record.json').write_text(json.dumps(record),encoding='utf-8')
    (tmp_path/'config.json').write_text(json.dumps(cfg),encoding='utf-8');(tmp_path/'ledger.json').write_text(json.dumps({'claims':[]}),encoding='utf-8')
    summary={'schema':'ncd.unified-proof-bundle.v3','protocol':str(protocol),'protocol_sha256':digest(protocol),'config_sha256':digest(tmp_path/'config.json'),'ledger_sha256':digest(tmp_path/'ledger.json'),
             'overall_objective_achieved':False,'jobs':['partial'],'records':[{'path':'partial/record.json','sha256':digest(tmp_path/'partial/record.json')}]}
    (tmp_path/'summary.json').write_text(json.dumps(summary),encoding='utf-8');calls=[]
    monkeypatch.setattr(api,'validate_config',lambda c:tmp_path)
    monkeypatch.setattr(api,'derive_ledger',lambda c,r:{'claims':[]})
    def independently_check(c,j):
        calls.append(j['id']);value=deepcopy(result)
        if changed:value['verification']['unresolved_cells']=0
        return value
    monkeypatch.setattr(api,'evaluate',independently_check)
    return api,calls


def test_verified_but_unclosed_record_is_replayed(tmp_path,monkeypatch):
    api,calls=partial_bundle(tmp_path,monkeypatch)
    result=api.verify_proof(tmp_path)
    assert calls==['partial'] and result['results']==['unresolved'] and not result['overall_objective_achieved']


def test_changed_partial_result_cannot_silently_strengthen(tmp_path,monkeypatch):
    api,calls=partial_bundle(tmp_path,monkeypatch,changed=True)
    with pytest.raises(ValueError,match='Proof changed'):api.verify_proof(tmp_path)
    assert calls==['partial']


def test_original_score_parent_not_closed_by_label_child(config):
    ledger=read(ROOT/'runs/original_proof_milestone_v1/ledger.json');records=[]
    for job in config['jobs']:
        result={'status':'verified','verification':{}}
        if job['kind']=='historical_core':result['verification']['original_ledger']=ledger
        if job['kind']=='actual_label_boundary':result['verification'].update(universal_single_label_equivariance_refuted=True,score_equivariance_refuted=False)
        records.append({'id':job['id'],'result':result})
    result=derive_ledger(config,records);claim=next(c for c in result['claims'] if c['id']=='R2.variable_equivariance')
    assert claim['status']=='unresolved' and [s['status'] for s in claim['resolved_subcontracts']]==['proved-scoped','refuted-scoped']
    assert len(result['claims'])==38 and sum(c['status']=='unresolved' for c in result['claims'])==36


def test_changed_partial_artifact_hash_is_rejected(config):
    altered=deepcopy(config);next(j for j in altered['jobs'] if j['kind']=='pending_piecewise')['artifact_sha256']='0'*64
    with pytest.raises(ValueError,match='partial proof'):validate_config(altered)


def test_older_public_schema_still_rejects_omitted_jobs(tmp_path):
    from ncd.proof_compatibility import replay
    source=ROOT/'runs/original_proof_registry_v2r1'
    for name in ('config.json','ledger.json','summary.json'):(tmp_path/name).write_bytes((source/name).read_bytes())
    summary=read(tmp_path/'summary.json');summary['records'].pop();(tmp_path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    with pytest.raises(ValueError,match='omitted proof jobs'):replay(ROOT,'verify-proof',tmp_path)


def test_original_score_contract_cannot_be_replaced_by_labels(config):
    altered=deepcopy(config);altered['original_contract_resolution']['R2.variable_equivariance']['single_label_counterexample_does_not_refute_native_score_contract']=False
    with pytest.raises(ValueError,match='contract changed'):validate_config(altered)


def test_relative_protocol_is_resolved_before_snapshot_cwd(config,monkeypatch):
    import ncd.proof_registry as api
    import ncd.proof_process as process
    from ncd.io import save_json
    job=deepcopy(next(j for j in config['jobs'] if j['kind']=='pending_piecewise'))
    job['protocol']=str(Path(job['protocol']).relative_to(ROOT)) if Path(job['protocol']).is_absolute() else job['protocol']
    captured=[]
    def isolated(command,cwd,seconds,memory):
        request=read(command[-2]);resolved=request['job'];captured.append(resolved)
        assert Path(resolved['protocol']).is_absolute() and Path(resolved['protocol'])==ROOT/job['protocol']
        assert Path(resolved['artifact']).is_absolute()
        save_json(command[-1],{'status':'verified','conclusion':'unresolved'})
        return {'exit_code':0,'stdout':b'','stderr':b'','resources':{'timeout':False,'peak_job_memory_bytes':0}}
    monkeypatch.setattr(process,'run_isolated',isolated)
    assert api.run_historical(config,job)['conclusion']=='unresolved' and len(captured)==1
