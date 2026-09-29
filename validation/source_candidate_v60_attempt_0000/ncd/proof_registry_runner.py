"""Isolated historical verifier; this script never imports candidate semantics."""
from pathlib import Path
import argparse,json,hashlib,sys

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('request');parser.add_argument('response');args=parser.parse_args()
    request=read(args.request);root=Path(request['snapshot']).resolve();env=Path(request['environment']).resolve()
    if Path.cwd().resolve()!=root or not Path(sys.executable).resolve().is_relative_to(env):raise ValueError('Historical isolation mismatch')
    for relative,sha in request['snapshot_sha256'].items():
        target=(root/relative).resolve()
        if not target.is_relative_to(root) or digest(target)!=sha:raise ValueError('Snapshot mismatch '+relative)
    import ncd,proof_extensions,proof_workbench,normalizer_proof,torch
    from importlib.metadata import version
    if version('neural-causal-decompiler')!='0.59.0':raise ValueError('Wrong historical core')
    for module in (ncd,proof_extensions,proof_workbench,normalizer_proof):
        installed=Path(module.__file__).resolve().parent
        if not installed.is_relative_to(env):raise ValueError('Historical source import')
        expected=root/installed.name
        if {p.name for p in installed.glob('*.py')}!={p.name for p in expected.glob('*.py')}:raise ValueError('Historical module set changed')
        if any(digest(p)!=digest(installed/p.name) for p in expected.glob('*.py')):raise ValueError('Historical installed/source bytes differ')
    torch.set_num_threads(2);job=request['job'];kind=job['kind']
    if kind=='historical_core':
        from ncd.original_proof_workflow import verify_proof
        result=verify_proof(job['artifact']);ledger=read(Path(job['artifact'])/'ledger.json')
        result={'verification':result,'original_ledger':ledger}
    elif kind=='historical_extensions':
        from proof_extensions.__main__ import verify_bundle
        result=verify_bundle(job['artifact'])
    elif kind=='historical_workbench':
        from proof_workbench.__main__ import verify_bundle
        result=verify_bundle(job['artifact'])
    elif kind=='historical_normalizer':
        from normalizer_proof.__main__ import verify_bundle
        # The unchanged frozen suffix must denote real operations.
        checkpoint=read(Path(job['artifact']).parent/'certificate.json')['checkpoint']
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
        if any(not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Non-real parameters')
        result=verify_bundle(job['artifact'])
    elif kind=='pending_piecewise':
        from ncd.frozen_mechanism_proof import export_mechanism
        from proof_workbench.piecewise_mechanism import verify_piecewise
        protocol=read(job['protocol']);certificate=read(job['artifact'])
        if digest(protocol['checkpoint'])!=protocol['checkpoint_sha256']:raise ValueError('Wrong target weights')
        for source,sha in protocol['source_sha256'].items():
            if digest(source)!=sha:raise ValueError('Changed mechanism proof source')
        if certificate['network']!=export_mechanism(protocol['checkpoint']) or certificate['domain']!=protocol['domain']:raise ValueError('Changed mechanism target/domain')
        from fractions import Fraction
        if certificate['epsilon']!=str(Fraction(protocol['epsilon'])) or certificate['search_cells']>protocol['max_cells']:raise ValueError('Changed mechanism gate/budget')
        result=verify_piecewise(certificate)
    else:raise ValueError('Unregistered historical backend')
    Path(args.response).write_text(json.dumps(result,sort_keys=True,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
