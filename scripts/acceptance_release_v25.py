"""Installed-wheel acceptance for active-intervention graph programs."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path(__file__).resolve().parents[1];installed=root/'validation/wheel_v25_env';out=root/'validation/wheel_v25_run';wheel=root/'dist/neural_causal_decompiler-0.25.0-py3-none-any.whl'
if out.exists():raise FileExistsError('Preserve prior 0.25 acceptance')
out.mkdir();sha=lambda b:hashlib.sha256(b).hexdigest();modules={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in (root/'ncd').glob('*.py')}
with zipfile.ZipFile(wheel) as z:
 for name,d in modules.items():assert sha(z.read(name))==d and sha((installed/name).read_bytes())==d,name
env=dict(os.environ,PYTHONPATH=str(installed),NCD_PROJECT_ROOT=str(root));check="import ncd;from pathlib import Path;assert ncd.__version__=='0.25.0';assert Path(ncd.__file__).resolve().is_relative_to(Path(r'"+str(installed)+"'));print(ncd.__file__)";subprocess.run([sys.executable,'-c',check],cwd=root/'validation',env=env,check=True)
commands=[['active-intervention-graph','--output',str(out/'active-intervention-graph'),'--quick','--seed','4892'],['verify-active-intervention-graph',str(out/'active-intervention-graph')]];records=[]
for i,args in enumerate(commands):
 print('installed CLI',args[0],flush=True);log=out/f'{i}_{args[0]}.log'
 with log.open('w',encoding='utf-8') as stream:r=subprocess.run([sys.executable,'-u','-m','ncd',*args],cwd=root/'validation',env=env,stdout=stream,stderr=subprocess.STDOUT)
 records.append({'command':args,'returncode':r.returncode,'log':log.name})
 if r.returncode:raise RuntimeError(f'{args[0]} failed: {log}')
(out/'status.json').write_text(json.dumps({'state':'verified','version':'0.25.0','wheel_sha256':sha(wheel.read_bytes()),'source_modules':modules,'commands':records,'scope':'fresh active-intervention-graph quick run and complete replay from isolated installed wheel; science not certified'},indent=2),encoding='utf-8');print('0.25 wheel verified',flush=True)











