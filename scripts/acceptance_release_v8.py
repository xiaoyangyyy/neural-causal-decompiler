"""Installed-wheel acceptance for conservative causal guidance."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
root=Path(__file__).resolve().parents[1];installed=root/'validation/wheel_v8_env';out=root/'validation/wheel_v8_run';wheel=root/'dist/neural_causal_decompiler-0.8.0-py3-none-any.whl'
if out.exists():raise FileExistsError('Preserve prior 0.8 acceptance')
out.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest();modules={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in (root/'ncd').glob('*.py')}
with zipfile.ZipFile(wheel) as z:
 for name,d in modules.items():assert sha(z.read(name))==d and sha((installed/name).read_bytes())==d,name
env=dict(os.environ,PYTHONPATH=str(installed),NCD_PROJECT_ROOT=str(root));check="import ncd;from pathlib import Path;assert ncd.__version__=='0.8.0';assert Path(ncd.__file__).resolve().is_relative_to(Path(r'"+str(installed)+"'));print(ncd.__file__)"
subprocess.run([sys.executable,'-c',check],cwd=root/'validation',env=env,check=True)
commands=[['conservative-guided','--source',str(root/'runs/joint_seed191'),'--output',str(out/'conservative-guided'),'--quick','--seed','1592'],['verify-conservative-guided',str(out/'conservative-guided')]]
records=[]
for i,args in enumerate(commands):
 print('installed CLI',args[0],flush=True);log=out/f'{i}_{args[0]}.log'
 with log.open('w',encoding='utf-8') as stream:r=subprocess.run([sys.executable,'-u','-m','ncd',*args],cwd=root/'validation',env=env,stdout=stream,stderr=subprocess.STDOUT)
 records.append({'command':args,'returncode':r.returncode,'log':log.name})
 if r.returncode:raise RuntimeError(f'{args[0]} failed: {log}')
record={'state':'verified','version':'0.8.0','wheel_sha256':sha(wheel.read_bytes()),'source_modules':modules,'commands':records,
 'scope':'fresh conservative-guided quick run and full regeneration replay from isolated installed wheel; science not certified'}
(out/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print('0.8 wheel verified',flush=True)
