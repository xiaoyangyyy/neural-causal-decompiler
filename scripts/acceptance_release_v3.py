"""Exercise the installed release through its public CLI, from outside the source root."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path(__file__).resolve().parents[1]
installed=root/'validation/wheel_v3_env'
output=root/'validation/wheel_v3_run'
if output.exists():
    raise FileExistsError('Preserve previous release evidence: use a new output directory')
output.mkdir()
env=dict(os.environ,PYTHONPATH=str(installed),NCD_PROJECT_ROOT=str(root))
wheel=root/'dist/neural_causal_decompiler-0.3.0-py3-none-any.whl'
sha=lambda data:hashlib.sha256(data).hexdigest()
with zipfile.ZipFile(wheel) as archive:
    files={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in (root/'ncd').rglob('*.py')}
    for name,digest in files.items():
        assert sha(archive.read(name))==digest,name
        assert sha((installed/name).read_bytes())==digest,name
subprocess.run([sys.executable,'-c',"import ncd;from pathlib import Path;assert Path(ncd.__file__).resolve().is_relative_to(Path(r'"+str(installed)+"'));assert ncd.__version__=='0.3.0';print(ncd.__file__)"],cwd=root/'validation',env=env,check=True)
commands=[]
for name,seed in [('joint',191),('numeric',291),('guided',391),('relational',491)]:
    command=[name,'--output',str(output/name),'--quick','--seed',str(seed)]
    if name in ('numeric','guided'):command+=['--source',str(output/'joint')]
    commands.extend([command,['verify-'+name,str(output/name)]])
results=[]
for index,args in enumerate(commands):
    print('Running installed CLI:',args[0],flush=True)
    log=output/f'{index:02d}_{args[0]}.log'
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u','-m','ncd',*args],cwd=root/'validation',env=env,stdout=stream,stderr=subprocess.STDOUT)
    results.append({'command':args,'returncode':result.returncode,'log':log.name})
    (output/'status.json').write_text(json.dumps({'state':'running' if result.returncode==0 else 'failed','commands':results},indent=2),encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'CLI failed: {args[0]}, see {log}')
record={'state':'verified','version':'0.3.0','wheel_sha256':sha(wheel.read_bytes()),'source_modules':files,'commands':results,'scope':'fresh quick runs and independent replay of all four added experiment commands; scientific claims not certified'}
(output/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Installed release verified',flush=True)
