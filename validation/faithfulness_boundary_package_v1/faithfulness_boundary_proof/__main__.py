"""Portable generation/replay; no checkout import needed after bundle export."""
from pathlib import Path
import argparse,hashlib,json,sys
from .boundary import construct,verify,REQUIREMENTS_SHA,GENERATOR_SHA


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def validate_protocol(c):
    if c.get('schema')!='ncd.faithfulness-boundary-plan.v1' or c.get('original_objective_achieved') is not False:
        raise ValueError('Wrong protocol/original promotion')
    if c.get('requirements_sha256')!=REQUIREMENTS_SHA or c.get('generator_sha256')!=GENERATOR_SHA:
        raise ValueError('Wrong source bindings')
    actual={p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')}
    if actual!=c['loaded_source_sha256']:raise ValueError('Changed installed checker')
    expected=[{'id':mode.replace('-','_')+'_n'+str(n),'nodes':n,'samples':96,'delta':'1/100','mode':mode}
        for n in (3,5,8) for mode in ('population','faithful-finite-sample')]
    if c.get('jobs')!=expected:raise ValueError('Declared cases omitted or changed')
    return expected


def replay(folder):
    folder=Path(folder).resolve();manifest=read(folder/'manifest.json')
    c=read(folder/'protocol.json');jobs=validate_protocol(c)
    names={'protocol.json','requirements.md','generator.py','scoped_ledger.json'}|{j['id']+'.json' for j in jobs}
    if manifest.get('schema')!='ncd.faithfulness-boundary-bundle.v1' or set(manifest['files'])!=names or manifest.get('original_objective_achieved') is not False:
        raise ValueError('Manifest changed')
    for name,expected in manifest['files'].items():
        if name not in names or sha(folder/name)!=expected:raise ValueError('Artifact hash mismatch: '+name)
    if sha(folder/'requirements.md')!=REQUIREMENTS_SHA or sha(folder/'generator.py')!=GENERATOR_SHA:
        raise ValueError('Original bytes changed')
    results=[];ledger=[]
    for job in jobs:
        certificate=read(folder/(job['id']+'.json'))
        if certificate['nodes']!=job['nodes'] or certificate['mode']!=job['mode'] or certificate['samples']!=job['samples'] or certificate['requested_error_upper']!=job['delta']:
            raise ValueError('Job contract changed')
        result=verify(certificate);results.append({'id':job['id'],**result})
        ledger.append({'id':job['id'],'status':'refuted','certificate_sha256':sha(folder/(job['id']+'.json')),
            'evidence_class':'true_causal_correctness','scope':certificate['scope'],
            'original_claim_entailment':False,'bounds':certificate['bounds'],'uncovered':certificate['uncovered']})
    if read(folder/'scoped_ledger.json')!=ledger:raise ValueError('Scoped ledger differs from independent replay')
    return {'status':'verified','jobs':len(jobs),'results':results,'original_objective_achieved':False}


def generate(config):
    path=Path(config).resolve();c=read(path);jobs=validate_protocol(c)
    root=Path(c['project_root']).resolve();out=(root/c['output']).resolve()
    if not out.is_relative_to(root):raise ValueError('Unsafe output path')
    if (out/'manifest.json').exists():raise FileExistsError('Retain previous bundle')
    out.mkdir(exist_ok=True,parents=True)
    if sha(root/'docs/FULL_REQUIREMENTS.md')!=REQUIREMENTS_SHA or sha(root/'ncd/multiverse.py')!=GENERATOR_SHA:
        raise ValueError('Original project source changed')
    (out/'protocol.json').write_bytes(path.read_bytes())
    (out/'requirements.md').write_bytes((root/'docs/FULL_REQUIREMENTS.md').read_bytes())
    (out/'generator.py').write_bytes((root/'ncd/multiverse.py').read_bytes())
    ledger=[]
    for job in jobs:
        cert=construct(job['nodes'],job['samples'],job['delta'],job['mode'])
        name=job['id']+'.json';write(out/name,cert)
        ledger.append({'id':job['id'],'status':'refuted','certificate_sha256':sha(out/name),
            'evidence_class':'true_causal_correctness','scope':cert['scope'],'original_claim_entailment':False,
            'bounds':cert['bounds'],'uncovered':cert['uncovered']})
    write(out/'scoped_ledger.json',ledger)
    names=['protocol.json','requirements.md','generator.py','scoped_ledger.json']+[j['id']+'.json' for j in jobs]
    write(out/'manifest.json',{'schema':'ncd.faithfulness-boundary-bundle.v1',
        'files':{name:sha(out/name) for name in names},'original_objective_achieved':False})
    return replay(out)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prove');a.add_argument('--config',required=True)
    a=sub.add_parser('verify-proof');a.add_argument('bundle')
    args=p.parse_args()
    result=generate(args.config) if args.command=='prove' else replay(args.bundle)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
