"""Proof workbench entry; all conclusions preserve original quantifiers."""
import argparse,json
from pathlib import Path
from ncd.io import digest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def certify(job):
    kind=job['kind']
    if kind=='mean_information':
        from .mean_information_boundary import certify_mean_information
        return certify_mean_information(**job.get('arguments',{}))
    if kind=='decimal_conversion':
        from .polynomial_semantics import certify_decimal_conversion_boundary
        return certify_decimal_conversion_boundary()
    if kind=='finite_mdl':
        from .bounded_mdl import certify_polynomial_mdl
        return certify_polynomial_mdl(job['target'],job['grammar'],job.get('max_candidates',50000),job.get('seconds',30))
    if kind=='approximate_root_nonunique':
        from .bounded_mdl import certify_approximate_root_nonuniqueness
        return certify_approximate_root_nonuniqueness(job['checkpoint'],job.get('domain'),job.get('epsilon','1/100'))
    raise ValueError('Unsupported proof workbench job')


def verify(certificate,checkpoint=None):
    schema=certificate['schema']
    if schema=='ncd.post-normalization-mean-impossibility.v1':
        from .mean_information_boundary import verify_mean_information
        return verify_mean_information(certificate)
    if schema=='ncd.decimal-polynomial-checker-boundary.v1':
        from .polynomial_semantics import verify_decimal_conversion_boundary
        return verify_decimal_conversion_boundary(certificate)
    if schema=='ncd.finite-polynomial-mdl.v1':
        from .bounded_mdl import verify_polynomial_mdl
        return verify_polynomial_mdl(certificate)
    if schema=='ncd.approximate-minimum-semantic-nonuniqueness.v1':
        from .bounded_mdl import verify_approximate_root_nonuniqueness
        if checkpoint and str(checkpoint)!=certificate['checkpoint']:raise ValueError('Changed bound checkpoint')
        return verify_approximate_root_nonuniqueness(certificate)
    if schema=='ncd.piecewise-affine-mechanism.v1':
        from ncd.frozen_mechanism_proof import export_mechanism
        from .piecewise_mechanism import verify_piecewise
        if checkpoint is None or export_mechanism(checkpoint)!=certificate['network']:raise ValueError('Actual checkpoint required for piecewise proof')
        return verify_piecewise(certificate)
    raise ValueError('Unsupported certificate schema')


def sources():
    files=sorted(Path('proof_workbench').glob('*.py'))
    files += [Path(p) for p in ['ncd/model.py','ncd/neural_sites.py','ncd/raw_program_trace.py','ncd/cdir.py',
        'ncd/frozen_mechanism_proof.py','ncd/proof_intervals.py','ncd/discovery_fidelity_proof.py','proof_extensions/statistical_frontend.py']]
    return {p.as_posix():digest(p) for p in files}


def run(config_path,resume=False):
    config=read(config_path)
    if config['schema']!='ncd.proof-workbench-config.v1':raise ValueError('Proof config schema')
    jobs=config['jobs'];ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)) or any(not i or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in i) for i in ids):raise ValueError('Proof job IDs')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not resume:raise FileExistsError('Use --resume for retained evidence')
    frozen=out/'protocol.json'
    if frozen.exists() and read(frozen)!=config:raise ValueError('Changed frozen protocol')
    save(frozen,config);rows=[]
    for job in jobs:
        path=out/job['id']/'certificate.json'
        if path.exists():
            if read(path.parent/'job.json')!=job:raise ValueError('Changed candidate job')
            c=read(path)
        else:c=certify(job);save(path,c);save(path.parent/'job.json',job)
        v=verify(c,job.get('checkpoint'));save(path.parent/'verification.json',v)
        rows.append({'id':job['id'],'certificate':path.as_posix(),'certificate_sha256':digest(path),
            'job_sha256':digest(path.parent/'job.json'),'status':c['status'],'verification':v,'original_claim_closed':False})
        print(job['id'],c['status'],v['status'],flush=True)
    ledger=Path(config['original_ledger']);claims=read(ledger)['claims']
    manifest={'schema':'ncd.proof-workbench-bundle.v1','source_sha256':sources(),'config_sha256':digest(frozen),
        'jobs':rows,'original_ledger':ledger.as_posix(),'original_ledger_sha256':digest(ledger),
        'original_claim_counts':{s:sum(c['status']==s for c in claims) for s in ('proved','refuted','unresolved')},
        'overall_objective_achieved':False,'full_plan_implemented':False}
    save(out/'manifest.json',manifest);return manifest


def verify_bundle(path):
    manifest=read(path);protocol=Path(path).parent/'protocol.json';config=read(protocol)
    if manifest['schema']!='ncd.proof-workbench-bundle.v1' or digest(protocol)!=manifest['config_sha256']:raise ValueError('Bundle protocol changed')
    if manifest['source_sha256']!=sources():raise ValueError('Proof implementation changed')
    if digest(manifest['original_ledger'])!=manifest['original_ledger_sha256']:raise ValueError('Original ledger changed')
    claims=read(manifest['original_ledger'])['claims']
    if manifest['original_claim_counts']!={s:sum(c['status']==s for c in claims) for s in ('proved','refuted','unresolved')}:raise ValueError('Changed claim counts')
    if [row['id'] for row in manifest['jobs']]!=[job['id'] for job in config['jobs']]:raise ValueError('Missing proof job')
    results=[]
    for row,job in zip(manifest['jobs'],config['jobs']):
        path=Path(config['output'])/job['id']/'certificate.json';job_path=path.parent/'job.json'
        if path.as_posix()!=row['certificate'] or digest(path)!=row['certificate_sha256'] or digest(job_path)!=row['job_sha256'] or read(job_path)!=job:raise ValueError('Changed certificate or job')
        c=read(path);v=verify(c,job.get('checkpoint'))
        if v!=row['verification'] or c['status']!=row['status'] or row['original_claim_closed']:raise ValueError('Changed proof conclusion')
        results.append(v)
    if manifest['overall_objective_achieved'] or manifest['full_plan_implemented']:raise ValueError('False completion')
    return {'status':'verified','proof_jobs':len(results),'results':results,'overall_objective_achieved':False}


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prove');p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('verify');p.add_argument('certificate');p.add_argument('--checkpoint')
    p=sub.add_parser('verify-bundle');p.add_argument('manifest')
    args=parser.parse_args()
    if args.command=='prove':r=run(args.config,args.resume)
    elif args.command=='verify':r=verify(read(args.certificate),args.checkpoint)
    else:r=verify_bundle(args.manifest)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
