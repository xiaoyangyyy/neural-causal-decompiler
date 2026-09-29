from pathlib import Path
import json,sys,os,hashlib
from ncd.proof_process import run_isolated
import ncd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/actual_graph_installed_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_actual_graph_supervisor_v1';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
if ncd.__version__!='0.63.0.dev1' or not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/wheel_v63_env'):raise ValueError('Wrong frozen resource supervisor')
source=Path(ncd.__file__).resolve().parent
expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'validation/source_candidate_v63/ncd').glob('*.py')}
if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}!=expected:raise ValueError('Supervisor source changed')
record=OUT/'supervision.json'
if record.exists():raise FileExistsError('Retain previous supervised installed proof')
execution=run_isolated([str(ROOT/'validation/actual_graph_env_v1/Scripts/python.exe'),'-I','-u',str(ROOT/'validation/check_actual_graph_installed_v1.py')],ROOT,1800,8*1024**3)
(OUT/'supervisor.stdout.log').write_bytes(execution['stdout']);(OUT/'supervisor.stderr.log').write_bytes(execution['stderr'])
passed=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources']['active_processes_on_return']==0 and (OUT/'status.json').exists()
result={'schema':'ncd.actual-graph-supervised.v1','status':'verified-scoped' if passed else 'unresolved','exit_code':execution['exit_code'],'resources':execution['resources'],'stage_seconds':1800,'threads':2,'source_candidate':'validation/source_candidate_v63','original_objective_achieved':False}
record.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(result,sort_keys=True));sys.exit(0 if passed else 1)
