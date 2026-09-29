from pathlib import Path
import argparse,json
from .realization import certify_normalizer,verify_normalizer,execute_exact,key
from .audit import audit_actual_interventions
from ncd.io import digest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def prove(config_path):
    config=read(config_path)
    if config['schema']!='ncd.normalizer-proof-protocol.v1' or config['original_objective_achieved'] is not False:raise ValueError('Invalid frozen protocol scope')
    if digest(config['checkpoint'])!=config['checkpoint_sha256']:raise ValueError('Target weights mismatch')
    for path,sha in config['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen source mismatch '+path)
    cert=certify_normalizer(config['checkpoint']);verification=verify_normalizer(cert)
    audit=audit_actual_interventions(cert)
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    write(out/'certificate.json',cert);write(out/'verification.json',verification);write(out/'program.json',cert['program']);write(out/'intervention_audit.json',audit)
    manifest={'schema':'ncd.normalizer-proof-bundle.v1','protocol':str(config_path),'protocol_sha256':digest(config_path),
        'source_sha256':config['source_sha256'],'files':{name:digest(out/name) for name in ['certificate.json','verification.json','program.json','intervention_audit.json']},
        'proof_conclusion':'proved','scope':'actual normalization prefix under real semantics; encoder/head retained','original_objective_achieved':False}
    write(out/'manifest.json',manifest);return verify_bundle(out/'manifest.json')

def verify_bundle(path):
    path=Path(path);manifest=read(path)
    if manifest['schema']!='ncd.normalizer-proof-bundle.v1' or manifest['original_objective_achieved'] is not False:raise ValueError('Wrong bundle scope')
    if digest(manifest['protocol'])!=manifest['protocol_sha256']:raise ValueError('Changed protocol')
    config=read(manifest['protocol'])
    if config.get('schema')!='ncd.normalizer-proof-protocol.v1' or config.get('original_objective_achieved') is not False:raise ValueError('Changed protocol scope')
    if config['source_sha256']!=manifest['source_sha256']:raise ValueError('Unbound source list')
    for name,sha in manifest['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('Source mismatch')
    expected_files={'certificate.json','verification.json','program.json','intervention_audit.json'}
    if set(manifest['files'])!=expected_files:raise ValueError('Incomplete proof bundle')
    for name,sha in manifest['files'].items():
        if digest(path.parent/name)!=sha:raise ValueError('Artifact mismatch')
    cert=read(path.parent/'certificate.json')
    if cert['checkpoint']!=config['checkpoint'] or cert['fx_export']['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('Wrong configured target')
    if cert['program']!=read(path.parent/'program.json'):raise ValueError('Program artifact mismatch')
    result=verify_normalizer(cert)
    if result!=read(path.parent/'verification.json'):raise ValueError('Verification result mismatch')
    if audit_actual_interventions(cert)!=read(path.parent/'intervention_audit.json'):raise ValueError('Intervention diagnostic replay mismatch')
    if manifest['proof_conclusion']!='proved' or manifest['scope']!='actual normalization prefix under real semantics; encoder/head retained':raise ValueError('Changed scope')
    return result

def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prove');p.add_argument('--config',required=True)
    p=commands.add_parser('verify');p.add_argument('certificate')
    p=commands.add_parser('verify-bundle');p.add_argument('manifest')
    p=commands.add_parser('execute');p.add_argument('--program',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prove':result=prove(args.config)
    elif args.command=='verify':result=verify_normalizer(read(args.certificate))
    elif args.command=='verify-bundle':result=verify_bundle(args.manifest)
    else:
        data=read(args.input);result=execute_exact(read(args.program),data['data'],data.get('sources'),data.get('mask'));write(args.output,result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
