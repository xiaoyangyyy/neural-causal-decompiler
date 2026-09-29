"""Uniform proof orchestration with explicit historical semantics and open claims."""
from pathlib import Path
from copy import deepcopy
import importlib,json,subprocess,tempfile,re,shutil
from .io import read_json,save_json,digest
from . import original_proof_workflow as legacy
SCHEMA='ncd.unified-proof-plan.v2'
KINDS={'historical_core','historical_extensions','historical_workbench','historical_normalizer','pending_piecewise','exact_polynomial'}
CLOSED={'R0.observational_unique_direction','R0.uniform_finite_sample_direction'}
PACKAGES=('ncd',)
APPROVED_SNAPSHOT_MANIFESTS={'manifest.json': 'a3c8106f367bbcc84f0c14a448ce54caa68a6066860f93232f5297fae688611e', 'workbench_v1_source_manifest.json': '4e26249a5f155205fafa4eae7453085544dd21e5ab77d09f6ef5fcdf08823730', 'normalizer_v1_source_manifest.json': 'b19f19418581bdf1cd063fe1c91c913dd227c4a99d0178c5cb060b041ac51130'}

def sources():
    result={}
    for name in PACKAGES:
        module=importlib.import_module(name);base=Path(module.__file__).resolve().parent
        result.update({name+'/'+p.name:digest(p) for p in sorted(base.glob('*.py'))})
    return result

def resolve(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Artifact path escapes project')
    return path

def validate_config(config):
    if config.get('schema')!=SCHEMA or config.get('original_objective_achieved') is not False:raise ValueError('Unsupported or false-completion protocol')
    if config['source_sha256']!=sources():raise ValueError('Candidate implementation changed')
    root=Path(config['project_root']).resolve()
    if not root.is_dir() or digest(resolve(root,config['requirements']))!=config['requirements_sha256']:raise ValueError('Requirements changed')
    jobs=config['jobs'];ids=[job['id'] for job in jobs]
    if not jobs or len(ids)!=len(set(ids)) or any(not re.fullmatch('[a-z][a-z0-9_]*',i) for i in ids):raise ValueError('Invalid proof job IDs')
    if sum(j['kind']=='historical_core' for j in jobs)!=1:raise ValueError('Exactly one independently derived original ledger required')
    valid_ids={r+'.'+k for r,rows in legacy.CLAIMS.items() for k,_ in rows}
    for job in jobs:
        if job['kind'] not in KINDS or job.get('original_claim_closed') is not False:raise ValueError('Unsupported backend or scoped closure promotion')
        if not job['related_claims'] or not set(job['related_claims'])<=valid_ids:raise ValueError('Unknown original claim link')
        if job['kind']=='exact_polynomial':continue
        artifact=resolve(root,job['artifact'])
        if job['kind']=='pending_piecewise':
            if digest(resolve(root,job['protocol']))!=job['protocol_sha256']:raise ValueError('Changed pending-proof protocol')
        else:
            bound=artifact/'manifest.json' if job['kind']=='historical_core' else artifact
            if digest(bound)!=job['artifact_sha256']:raise ValueError('Historical evidence changed')
    context=config['historical_context'];snapshot=resolve(root,context['snapshot']);env=resolve(root,context['environment'])
    if not snapshot.is_dir() or not (env/'Scripts/python.exe').is_file():raise ValueError('Historical verification environment unavailable')
    approved={}
    for name,sha in APPROVED_SNAPSHOT_MANIFESTS.items():
        path=resolve(snapshot,name)
        if digest(path)!=sha:raise ValueError('Unapproved historical source manifest')
        for relative,value in read_json(path)['files'].items():
            if relative in approved and approved[relative]!=value:raise ValueError('Conflicting historical source manifests')
            approved[relative]=value
    if context['snapshot_sha256']!=approved:raise ValueError('Incomplete or altered approved historical source list')
    for name,sha in context['snapshot_sha256'].items():
        path=resolve(snapshot,name)
        if digest(path)!=sha:raise ValueError('Historical source changed')
    if type(config['memory_bytes']) is not int or not 0<config['memory_bytes']<=8*1024**3:raise ValueError('Verifier memory budget')
    if not 0<config['worker_seconds']<=43200:raise ValueError('Verifier stage budget')
    resolve(root,config['output']);return root

def run_historical(config,job):
    root=validate_config(config);context=config['historical_context']
    snapshot=resolve(root,context['snapshot']);env=resolve(root,context['environment'])
    request={'job':job,'snapshot':str(snapshot),'environment':str(env),'snapshot_sha256':context['snapshot_sha256']}
    runner=Path(__file__).with_name('proof_registry_runner.py')
    with tempfile.TemporaryDirectory(prefix='ncd-proof-') as tmp:
        request_path=Path(tmp)/'request.json';response_path=Path(tmp)/'response.json';save_json(request_path,request)
        command=[str(env/'Scripts/python.exe'),'-I',str(runner),str(request_path),str(response_path)]
        from .proof_process import run_isolated
        execution=run_isolated(command,snapshot,config['worker_seconds'],config['memory_bytes'])
        if execution['resources']['timeout']:return {'status':'unresolved','reason':'independent verifier time budget exhausted'}
        if execution['resources'].get('peak_job_memory_bytes',0)>config['memory_bytes']:return {'status':'unresolved','reason':'independent verifier reported memory budget exceeded'}
        if execution['exit_code']:raise ValueError('Independent historical verifier rejected evidence: '+execution['stderr'].decode(errors='replace')[-4000:])
        if not response_path.is_file():raise ValueError('Independent verifier produced no response')
        result=read_json(response_path)
        result['_resource_guard']={'job_memory_limit_bytes':config['memory_bytes'],'descendants_included':True,'reported_peak_within_limit':True}
        return result

def polynomial_budget(expression):
    from .cdir import Node
    node=Node.from_dict(expression)
    def measure(n):
        if n.op in ('var','constant'):return 1,int(n.op=='var')
        if n.op not in ('add','sub','mul','square'):raise ValueError('Unsupported polynomial operator')
        children=[measure(a) for a in n.args];count=1+sum(v[0] for v in children)
        degree=max(v[1] for v in children) if n.op in ('add','sub') else (2*children[0][1] if n.op=='square' else sum(v[1] for v in children))
        return count,degree
    count,degree=measure(node);return count<=64 and degree<=8

def evaluate(config,job):
    root=validate_config(config);kind=job['kind']
    if kind=='exact_polynomial':
        if not polynomial_budget(job['left']) or not polynomial_budget(job['right']):return {'status':'unresolved','reason':'polynomial certificate budget: at most 64 AST nodes and degree 8'}
        from .exact_polynomial import prove_polynomial_equivalence,verify_polynomial_equivalence
        certificate=prove_polynomial_equivalence(job['left'],job['right'],job['variables']);verification=verify_polynomial_equivalence(certificate)
        return {'status':certificate['status'],'certificate':certificate,'verification':verification,'original_claim_closed':False}
    artifact=resolve(root,job['artifact'])
    if kind=='pending_piecewise' and not artifact.is_file():return {'status':'unresolved','reason':'registered actual-target certificate has not been produced'}
    result=run_historical(config,job)
    if result.get('status')=='unresolved':return result
    return {'status':result.get('conclusion','verified'),'verification':result,'original_claim_closed':False,
        'artifact_sha256':digest(artifact/'manifest.json' if kind=='historical_core' else artifact)}

def derive_ledger(config,records):
    core=next(r for r,j in zip(records,config['jobs']) if j['kind']=='historical_core')
    if core['result']['status']=='unresolved':
        ledger=legacy.initial_ledger(resolve(Path(config['project_root']).resolve(),config['requirements']))
    else:ledger=deepcopy(core['result']['verification']['original_ledger'])
    expected={r+'.'+k for r,rows in legacy.CLAIMS.items() for k,_ in rows}
    if {c['id'] for c in ledger['claims']}!=expected or len(ledger['claims'])!=len(expected):raise ValueError('Original claims omitted or duplicated')
    if ledger['requirements_sha256']!=config['requirements_sha256'] or ledger['overall_objective_achieved']:raise ValueError('Original scope changed')
    for claim in ledger['claims']:
        if claim['status']!='unresolved' and (claim['id'] not in CLOSED or claim['status']!='refuted'):raise ValueError('Closure lacks a registered scope-matched backend')
        claim['scoped_evidence']=[{'job':r['id'],'result_status':r['result']['status'],'entails_original_claim':False} for r,j in zip(records,config['jobs']) if claim['id'] in j['related_claims']]
    return ledger

def prove(config_path,resume=False):
    config=read_json(config_path)
    if config.get('schema')!=SCHEMA:return legacy.prove(config_path,resume)
    root=validate_config(config);out=resolve(root,config['output']);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not resume:raise FileExistsError('Use --resume to retain existing proof attempts')
    frozen=out/'config.json'
    if frozen.exists() and read_json(frozen)!=config:raise ValueError('Frozen unified protocol changed')
    if resume and (out/'summary.json').is_file():
        history=out/'history';history.mkdir(exist_ok=True)
        archive=history/('revision_%04d'%len(list(history.glob('revision_*'))));archive.mkdir()
        for path in list(out.iterdir()):
            if path==history:continue
            if path.is_dir():shutil.copytree(path,archive/path.name)
            elif path.is_file():shutil.copy2(path,archive/path.name)
    save_json(frozen,config);records=[]
    for job in config['jobs']:
        path=out/job['id']/'record.json'
        previous=read_json(path) if path.is_file() else None
        if previous and previous['job']!=job:raise ValueError('Changed retained proof job')
        if previous and previous['result']['status']!='unresolved':record=previous
        else:
            record={'id':job['id'],'job':job,'result':evaluate(config,job)};save_json(path,record)
        records.append(record);save_json(out/'progress.json',{'retained_jobs':len(records),'declared_jobs':len(config['jobs']),'overall_objective_achieved':False})
        print(job['id'],record['result']['status'],flush=True)
    ledger=derive_ledger(config,records);save_json(out/'ledger.json',ledger)
    summary={'schema':'ncd.unified-proof-bundle.v2','jobs':[j['id'] for j in config['jobs']],
        'records':[{'path':j['id']+'/record.json','sha256':digest(out/j['id']/'record.json')} for j in config['jobs']],
        'config_sha256':digest(frozen),'ledger_sha256':digest(out/'ledger.json'),'overall_objective_achieved':False}
    save_json(out/'summary.json',summary);return verify_proof(out)

def verify_proof(directory):
    out=Path(directory).resolve();summary=read_json(out/'summary.json')
    if summary.get('schema')!='ncd.unified-proof-bundle.v2':return legacy.verify_proof(out)
    config=read_json(out/'config.json');validate_config(config)
    if digest(out/'config.json')!=summary['config_sha256'] or digest(out/'ledger.json')!=summary['ledger_sha256']:raise ValueError('Unified bundle hash mismatch')
    if summary['overall_objective_achieved'] is not False or summary['jobs']!=[j['id'] for j in config['jobs']] or len(summary['records'])!=len(config['jobs']):raise ValueError('False closure or omitted proof jobs')
    records=[]
    for row,job in zip(summary['records'],config['jobs']):
        if row['path']!=job['id']+'/record.json':raise ValueError('Wrong proof record path')
        path=resolve(out,row['path'])
        if digest(path)!=row['sha256']:raise ValueError('Changed proof record')
        record=read_json(path)
        if record['id']!=job['id'] or record['job']!=job:raise ValueError('Changed job specification')
        if record['result']['status']=='unresolved':
            allowed={'registered actual-target certificate has not been produced','independent verifier time budget exhausted','independent verifier reported memory budget exceeded','polynomial certificate budget: at most 64 AST nodes and degree 8'}
            if set(record['result'])!={'status','reason'} or record['result']['reason'] not in allowed:raise ValueError('Invalid retained open result')
            records.append(record);continue
        actual=evaluate(config,job)
        # A newly available pending proof requires an explicit resumed generation;
        # an old unresolved snapshot never silently adopts a stronger conclusion.
        if actual!=record['result']:raise ValueError('Proof changed; preserve the old snapshot and resume generation')
        records.append(record)
    ledger=derive_ledger(config,records)
    if ledger!=read_json(out/'ledger.json'):raise ValueError('Ledger closure or scoped evidence changed')
    return {'status':'verified','jobs':len(records),'results':[r['result']['status'] for r in records],
        'original_claim_counts':{s:sum(c['status']==s for c in ledger['claims']) for s in ('proved','refuted','unresolved')},'overall_objective_achieved':False}

def audit_requirements(directory):
    out=Path(directory).resolve()
    if read_json(out/'summary.json').get('schema')!='ncd.unified-proof-bundle.v2':return legacy.audit_requirements(out)
    verification=verify_proof(out);claims=read_json(out/'ledger.json')['claims'];groups={r:{s:0 for s in ('proved','refuted','unresolved')} for r in legacy.CLAIMS}
    for claim in claims:groups[claim['requirement']][claim['status']]+=1
    return {'schema':'ncd.unified-requirements-audit.v2','requirements':groups,'claim_count':len(claims),
        'unresolved_claims':[c['id'] for c in claims if c['status']=='unresolved'],'scoped_job_results':verification['results'],
        'overall_objective_achieved':all(c['status'] in ('proved','refuted') for c in claims)}
