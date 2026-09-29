from pathlib import Path
import json,os,hashlib,sys
from ncd.proof_process import run_isolated
import ncd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/wheel_v64r2_preflight_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_v64r2_supervised';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
if ncd.__version__!='0.64.0.dev2' or not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/wheel_v64r2_env'):raise ValueError('Wrong resource supervisor installation')
path=OUT/'supervision.json'
if path.exists():raise FileExistsError('Retain earlier installed preflight')
execution=run_isolated([sys.executable,'-I','-u',str(ROOT/'validation/check_registry_v64r2_installed_v1.py')],ROOT,3600,8*1024**3)
(OUT/'supervisor.stdout.log').write_bytes(execution['stdout']);(OUT/'supervisor.stderr.log').write_bytes(execution['stderr'])
passed=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources']['active_processes_on_return']==0 and (OUT/'status.json').is_file()
record={'schema':'ncd.registry-v64-installed-supervision.v1','status':'passed' if passed else 'unresolved','exit_code':execution['exit_code'],'resources':execution['resources'],'original_objective_achieved':False}
path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(record));sys.exit(0 if passed else 1)
