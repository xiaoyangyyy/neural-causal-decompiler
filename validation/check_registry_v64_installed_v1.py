from pathlib import Path
from xml.etree import ElementTree as ET
import json,sys,subprocess,os,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/wheel_v64_preflight_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_v64_preflight';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
import ncd
if ncd.__version__!='0.64.0.dev1' or not Path(ncd.__file__).resolve().is_relative_to(ROOT/'validation/wheel_v64_env'):raise ValueError('Source checkout imported')
full=read(ROOT/'validation/original_proof_registry_protocol_v6.json');loaded={'ncd/'+p.name:digest(p) for p in Path(ncd.__file__).resolve().parent.glob('*.py')}
if loaded!=full['source_sha256'] or len(loaded)!=140:raise ValueError('Installed source mismatch')
protocol=ROOT/'validation/original_graph_registry_public_preflight_protocol_v64.json';config=read(protocol);bundle=ROOT/config['output']
def command(name,args,expected=0,timeout=900):
 p=subprocess.run([sys.executable,'-I',*args],cwd=OUT,capture_output=True,timeout=timeout)
 (OUT/(name+'.stdout.log')).write_bytes(p.stdout);(OUT/(name+'.stderr.log')).write_bytes(p.stderr)
 if p.returncode!=expected:raise ValueError(name+' failed; retain logs')
 return p.stdout.decode('utf-8')
text=command('prove',['-m','ncd','prove','--config',str(protocol)]);proof=json.loads(text[text.find('{'):])
replay=json.loads(command('verify',['-m','ncd','verify-proof',str(bundle)]))
if proof!=replay or proof['jobs']!=4 or proof['original_claim_counts']!={'proved':0,'refuted':2,'unresolved':36}:raise ValueError('Native graph public replay changed')
audit=json.loads(command('audit',['-m','ncd','audit-requirements',str(bundle),'--require-closed'],1))
if audit['overall_objective_achieved'] or len(audit['unresolved_claims'])!=36 or len(audit['scoped_program_realization_contracts'])!=12 or audit['scoped_unresolved_program_contracts']:raise ValueError('Strict graph scope/audit changed')
ledger=read(bundle/'ledger.json')
if len(ledger['scoped_program_realization_contracts'])!=12 or any(c['contract']['status']!='proved' for c in ledger['scoped_program_realization_contracts']):raise ValueError('Graph domains not independently registered')
xml=OUT/'pytest.xml'
command('pytest',['-m','pytest',str(ROOT/'tests/test_unified_proof_registry_v6.py'),str(ROOT/'tests/test_original_proof_integration.py'),'--import-mode=importlib','-q','--basetemp='+str(OUT/'temp_run_0000'),'--junitxml='+str(xml)])
suite=ET.parse(xml).getroot()[0]
if int(suite.get('tests'))!=79 or any(int(suite.get(k,'0')) for k in ('errors','failures','skipped')):raise ValueError('Native preflight checks incomplete')
wheel=next((ROOT/'validation/dist_v64').glob('*.whl'))
status={'schema':'ncd.installed-registry-preflight.v64.v1','status':'passed','version':'0.64.0.dev1','tests':79,'native_registry_tests':68,'baseline_integration_tests':11,'source_modules':140,'source_sha256':loaded,'public_native_preflight_jobs':4,'public_prove_verify_and_strict_audit_checked':True,'strict_audit_exit_code':1,'scoped_graph_program_domains_proved':12,'original_claim_counts':proof['original_claim_counts'],'wheel':wheel.relative_to(ROOT).as_posix(),'wheel_sha256':digest(wheel),'junit_sha256':digest(xml),'native6_four_job_bundle_sha256':digest(bundle/'summary.json'),'schema5_dispatch_fixed':True,'complete_14_job_replay_pending':True,'not_promoted':True,'whole_project_complete':False}
(OUT/'status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n',encoding='utf-8');print('Native6 public four-job replay passed; twelve graph domains registered;79 installed tests passed')
