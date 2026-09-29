from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1]
TEMP=ROOT/'validation/task_temp_E_kernel_v63';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(ROOT/'validation/source_candidate_v63'))
from ncd.proof_registry import validate_config,evaluate,merge_recovery_contracts,KERNEL_KINDS
from ncd.io import read_json,save_json,digest
config=read_json(ROOT/'validation/original_proof_registry_protocol_v5.json');validate_config(config)
OUT=ROOT/'validation/theory_kernel_preflight_v63';OUT.mkdir()
records=[]
for job in config['jobs']:
 if job['kind'] not in KERNEL_KINDS:continue
 result=evaluate(config,job)
 if result['status']!='verified' or result['verification'].get('original_claim_closed') is not False:raise ValueError('Kernel scope/replay failed')
 record={'id':job['id'],'job':job,'result':result};records.append(record);save_json(OUT/(job['id']+'.json'),record)
contracts=merge_recovery_contracts(records)
counts={s:sum(c['status']==s for c in contracts) for s in ('proved','refuted','unresolved')}
if len(contracts)!=7 or counts!={'proved':5,'refuted':2,'unresolved':0}:raise ValueError('Wrong scoped closure merge: '+str(counts))
for c in contracts:
 if c['status']=='refuted' and (not c['refutation_jobs'] or c['proof_jobs']):raise ValueError('Contradictory refutation ledger')
 if c['inputs']['samples']==1000000 and c['status']!='proved':raise ValueError('Old valid evidence was lost')
save_json(OUT/'merged_recovery_contracts.json',contracts)
save_json(OUT/'status.json',{'status':'four-installed-theory-kernels-and-scope-merge-verified','kernel_jobs':4,'scoped_recovery_contracts':counts,'older_failed_attempts_preserved':True,'original_claim_counts':{'proved':0,'refuted':2,'unresolved':36},'source_candidate_promoted':False,'whole_project_complete':False,'proof_protocol_sha256':digest(ROOT/'validation/original_proof_registry_protocol_v5.json')})
print('Four installed theory kernels verified; seven scoped recovery contracts = five proved + two strictly refuted; original requirements remain open')
