from pathlib import Path
import hashlib,json,sys

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));env=Path(request['environment']).resolve();snapshot=Path(request['snapshot']).resolve()
 if not Path(sys.executable).resolve().is_relative_to(env):raise ValueError('Legacy environment mismatch')
 import ncd,torch
 from importlib.metadata import version
 package=Path(ncd.__file__).resolve().parent
 if not package.is_relative_to(env) or version('neural-causal-decompiler')!=request['version']:raise ValueError('Wrong legacy package/version')
 expected={Path(name).name:sha for name,sha in request['source_sha256'].items()}
 if set(expected)!={p.name for p in package.glob('*.py')} or any(digest(package/name)!=sha or digest(snapshot/'ncd'/name)!=sha for name,sha in expected.items()):raise ValueError('Legacy package bytes changed')
 torch.set_num_threads(6)
 if request['version'] in ('0.60.0.dev1','0.61.0.dev2','0.62.0.dev3','0.63.0.dev1','0.64.0.dev2'):from ncd import proof_registry as api
 else:from ncd import original_proof_workflow as api
 op=request['operation'];path=Path(request['argument'])
 if op=='prove':result=api.prove(path,request['resume'])
 elif op=='verify-proof':result=api.verify_proof(path)
 elif op=='audit-requirements':result=api.audit_requirements(path)
 else:raise ValueError('Unregistered compatibility operation')
 Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
