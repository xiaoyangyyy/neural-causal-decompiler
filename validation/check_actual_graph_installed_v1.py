from pathlib import Path
from xml.etree import ElementTree as ET
import json,os,sys,subprocess,hashlib,importlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/actual_graph_installed_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_actual_graph_installed_v1';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
config=read(ROOT/'validation/actual_graph_protocol_v1.json')
package=importlib.import_module('graph_ssa_proof');folder=Path(package.__file__).resolve().parent
if not folder.is_relative_to(ROOT/'validation/actual_graph_env_v1'):raise ValueError('Source checkout imported')
actual={'graph_ssa_proof/'+p.name:digest(p) for p in folder.glob('*.py')}
if actual!=config['source_sha256']:raise ValueError('Installed kernel byte mismatch')
if str(ROOT) in sys.path or str(ROOT/'graph_ssa_proof') in sys.path:raise ValueError('Checkout import path leak')
manifest=ROOT/config['output']/'manifest.json'
def command(name,args,timeout):
 child=subprocess.run([sys.executable,'-I',*args],cwd=OUT,capture_output=True,timeout=timeout)
 (OUT/(name+'.stdout.log')).write_bytes(child.stdout);(OUT/(name+'.stderr.log')).write_bytes(child.stderr)
 if child.returncode:raise RuntimeError(name+' failed; retain logs')
 return child
if manifest.exists():raise FileExistsError('Retain previous installed proof output')
first=command('prove',['-m','graph_ssa_proof','prove','--config',str(ROOT/'validation/actual_graph_protocol_v1.json')],600)
result=json.loads(first.stdout)
second=command('verify',['-m','graph_ssa_proof','verify-proof',str(manifest)],600)
replay=json.loads(second.stdout)
if result!=replay or replay['status']!='verified' or replay['proved_feature_domain_cases']!=12 or not replay['physical_module_map_independently_checked'] or any(r['mathematical_error']!='0' or r['declared_interchange_sites']!=14 for r in replay['results']) or replay['original_objective_achieved'] or replay['raw_frontend_included'] or replay['hardware_rounding_covered'] or replay['mdl_minimality_proved']:raise ValueError('Installed proof scope changed')
xml=OUT/'pytest.xml';baseline=OUT/'temp_run_0000'
command('pytest',['-m','pytest',str(ROOT/'tests/test_graph_ssa_proof.py'),'--import-mode=importlib','-q','--basetemp='+str(baseline),'--junitxml='+str(xml)],600)
suite=ET.parse(xml).getroot()[0]
if int(suite.get('tests'))!=77 or any(int(suite.get(k,'0')) for k in ('errors','failures','skipped')):raise ValueError('Installed checks incomplete')
reports=[]
for file in baseline.glob('test_real_hooks_composed_inter*/graph_intervention_diagnostics.json'):
 report=read(file)
 if report['all_domains_have_device_label_guarantee'] or report['statistical_independent_worlds'] or report['original_claim_closed'] or not report['actual_14_modules_executed_per_case'] or len(report['diagnostic_cases'])!=31:raise ValueError('Diagnostics strengthened or dropped')
 reports.append(report)
# Independently require the whole 4 by 3 domain grid.
expected={(r['id'],n) for r in read(ROOT/config['target_manifest'])['records'] for n in r['nodes']}
if {(r['target'],r['nodes']) for r in reports}!=expected or len(reports)!=12:raise ValueError('Missing diagnostic domain')
mismatches=sum(len(c['label_disagreements']) for r in reports for c in r['diagnostic_cases'])
if not mismatches or any(not any(c['label_disagreements'] for c in r['diagnostic_cases']) for r in reports if r['nodes']==8):raise ValueError('Previously observed device label failures disappeared from coverage')
summary={'schema':'ncd.actual-graph-diagnostics.v1','reports':sorted(reports,key=lambda r:(r['target'],r['nodes'])),'diagnostic_intervention_cases':372,'observed_label_disagreements':mismatches,'device_label_identity':'observed-failed','strict_mathematical_refutation':False,'maximum_observed_score_error':max(c['maximum_score_error'] for r in reports for c in r['diagnostic_cases']),'original_claim_closed':False}
save(OUT/'diagnostics.json',summary)
wheel=next((ROOT/'validation/actual_graph_dist_v1').glob('*.whl'))
status={'schema':'ncd.actual-graph-installed.v1','status':'verified-scoped','tests':77,'test_gate':'Weights, math certificate tampering, actual read/write execution and retained floating label failures; NOT hardware label identity','verification':replay,'source_installed_bytes_equal':True,'source_checkout_not_on_import_path':True,'source_sha256':actual,'wheel_sha256':digest(wheel),'bundle_manifest_sha256':digest(manifest),'junit_sha256':digest(xml),'diagnostics_sha256':digest(OUT/'diagnostics.json'),'test_source_sha256':digest(ROOT/'tests/test_graph_ssa_proof.py'),'checker_sha256':digest(__file__),'stage_threads':2,'original_claim_closed':False,'original_objective_achieved':False}
save(OUT/'status.json',status)
print('Twelve actual graph feature-domain proofs and fourteen-site maps replayed; 77 installed checks passed; device label disagreements retained')
