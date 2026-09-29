"""Standalone proof-extension entry point; frozen 0.59 core remains untouched."""
import argparse
import json
from pathlib import Path
from ncd.io import digest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def certify(job):
    kind=job['kind']
    if kind=='finite_sample':
        from .finite_sample_boundary import certify_finite_sample_boundary
        return certify_finite_sample_boundary(**job.get('arguments',{}))
    if kind=='decoder_boundary':
        from .graph_candidates import certify_decoder_boundary
        return certify_decoder_boundary(job['case'])
    if kind=='feature_collision':
        from .feature_collision import certify_feature_collision
        return certify_feature_collision(job['checkpoint'],**job.get('arguments',{}))
    if kind=='linear_gaussian_coupling':
        from .scm_propagation import certify_linear_gaussian_coupling
        return certify_linear_gaussian_coupling(job['reference'],job['realization'])
    raise ValueError('Unsupported proof job '+kind)


def verify(certificate,checkpoint=None):
    schema=certificate['schema']
    if schema=='ncd.finite-sample-separation-boundary.v1':
        from .finite_sample_boundary import verify_finite_sample_boundary
        return verify_finite_sample_boundary(certificate)
    if schema=='ncd.decoder-equivariance-boundary.v1':
        from .graph_candidates import verify_decoder_boundary
        return verify_decoder_boundary(certificate)
    if schema=='ncd.fixed-statistic-family-impossibility.v1':
        from .feature_collision import verify_feature_collision
        return verify_feature_collision(certificate,checkpoint)
    if schema=='ncd.linear-gaussian-intervention-coupling.v1':
        from .scm_propagation import verify_linear_gaussian_coupling
        return verify_linear_gaussian_coupling(certificate)
    raise ValueError('Unsupported proof certificate '+schema)


def sources():
    files=sorted(Path('proof_extensions').glob('*.py'))
    files += [Path(p) for p in ['ncd/statistics.py','ncd/model.py','ncd/graph_model.py','ncd/graphs.py',
        'ncd/proof_intervals.py','ncd/discovery_fidelity_proof.py']]
    return {p.as_posix():digest(p) for p in files}


def run(config_path,resume=False):
    config=read(config_path)
    if config['schema']!='ncd.original-proof-extensions-config.v1':raise ValueError('Proof config schema')
    jobs=config['jobs'];ids=[j['id'] for j in jobs]
    if len(ids)!=len(set(ids)) or any(not i or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in i) for i in ids):raise ValueError('Proof job IDs')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()) and not resume:raise FileExistsError('Use --resume to preserve existing proof results')
    frozen=output/'protocol.json'
    if frozen.exists() and read(frozen)!=config:raise ValueError('Frozen protocol mismatch')
    save(frozen,config);rows=[]
    for job in jobs:
        path=output/job['id']/'certificate.json'
        if path.exists():
            c=read(path);v=verify(c,job.get('checkpoint'))
            # Reuse is allowed only for the exact original candidate specification.
            if read(path.parent/'job.json')!=job:raise ValueError('Changed proof job')
        else:
            c=certify(job);save(path,c);save(path.parent/'job.json',job);v=verify(c,job.get('checkpoint'))
        save(path.parent/'verification.json',v)
        rows.append({'id':job['id'],'certificate':path.as_posix(),'certificate_sha256':digest(path),
            'job_sha256':digest(path.parent/'job.json'),'verification':v,'status':c['status'],
            'original_claim_closed':False})
        print(job['id'],c['status'],v['status'],flush=True)
    ledger=Path(config['original_ledger']);claims=read(ledger)['claims']
    manifest={'schema':'ncd.original-proof-extensions-bundle.v1','config_sha256':digest(frozen),
        'source_sha256':sources(),'original_ledger':ledger.as_posix(),'original_ledger_sha256':digest(ledger),
        'original_claim_counts':{s:sum(c['status']==s for c in claims) for s in ('proved','refuted','unresolved')},
        'jobs':rows,'overall_objective_achieved':False,'full_plan_implemented':False,
        'released_wheel_contains_extensions':False}
    save(output/'manifest.json',manifest);return manifest


def verify_bundle(path):
    manifest=read(path);config_path=Path(path).parent/'protocol.json';config=read(config_path)
    if manifest['schema']!='ncd.original-proof-extensions-bundle.v1' or digest(config_path)!=manifest['config_sha256']:raise ValueError('Bundle protocol hash mismatch')
    if manifest['source_sha256']!=sources():raise ValueError('Proof source hash mismatch')
    if digest(manifest['original_ledger'])!=manifest['original_ledger_sha256']:raise ValueError('Original ledger mismatch')
    claims=read(manifest['original_ledger'])['claims']
    counts={s:sum(c['status']==s for c in claims) for s in ('proved','refuted','unresolved')}
    if manifest['original_claim_counts']!=counts:raise ValueError('Original claim count mismatch')
    if [row['id'] for row in manifest['jobs']]!=[j['id'] for j in config['jobs']]:raise ValueError('Missing or reordered proof job')
    results=[]
    for row,job in zip(manifest['jobs'],config['jobs']):
        certificate=Path(config['output'])/job['id']/'certificate.json'
        if row['certificate']!=certificate.as_posix() or digest(certificate)!=row['certificate_sha256']:raise ValueError('Certificate artifact mismatch')
        job_path=certificate.parent/'job.json'
        if digest(job_path)!=row['job_sha256'] or read(job_path)!=job:raise ValueError('Proof job mismatch')
        result=verify(read(certificate),job.get('checkpoint'))
        if result!=row['verification'] or read(certificate)['status']!=row['status'] or row['original_claim_closed']:raise ValueError('Proof result mismatch')
        results.append(result)
    if manifest['overall_objective_achieved'] or manifest['full_plan_implemented'] or manifest['released_wheel_contains_extensions']:raise ValueError('Invalid completion/release marker')
    return {'status':'verified','proof_jobs':len(results),'results':results,'overall_objective_achieved':False}


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prove');p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('verify');p.add_argument('certificate');p.add_argument('--checkpoint')
    p=sub.add_parser('verify-bundle');p.add_argument('manifest')
    args=parser.parse_args()
    if args.command=='prove':result=run(args.config,args.resume)
    elif args.command=='verify':result=verify(read(args.certificate),args.checkpoint)
    else:result=verify_bundle(args.manifest)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
