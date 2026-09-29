from pathlib import Path
import json,os,sys
from ncd.proof_process import run_isolated
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/faithfulness_boundary_installed_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_faithfulness_installed_supervisor_v1';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP))
if (OUT/'supervision.json').exists():raise FileExistsError('Retain previous run')
execution=run_isolated([str(ROOT/'validation/faithfulness_boundary_env_v1/Scripts/python.exe'),'-I',
    '-u',str(ROOT/'validation/check_faithfulness_installed_v1.py')],OUT,300,268435456)
(OUT/'worker.stdout.log').write_bytes(execution['stdout']);(OUT/'worker.stderr.log').write_bytes(execution['stderr'])
passed=execution['exit_code']==0 and not execution['resources']['timeout'] and execution['resources']['active_processes_on_return']==0
record={'status':'passed' if passed else 'unresolved','exit_code':execution['exit_code'],'resources':execution['resources'],
    'original_objective_achieved':False,'no_neural_training':True}
(OUT/'supervision.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(record));sys.exit(0 if passed else 1)
