from pathlib import Path
from xml.etree import ElementTree as ET
from fractions import Fraction as Q
import sys,os,json,importlib,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/estimator_failure_installed_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_estimator_failure_installed_v1';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=json.loads((ROOT/'validation/estimator_failure_protocol_v1.json').read_text(encoding='utf-8'))
for name in ('estimator_failure_proof','finite_graph_proof'):
 package=importlib.import_module(name);folder=Path(package.__file__).resolve().parent
 if not folder.is_relative_to(ROOT/'validation/estimator_failure_env_v1'):raise ValueError('Source import')
 expected={Path(f).name:h for f,h in config['source_sha256'].items() if f.startswith(name+'/')}
 if set(expected)!={p.name for p in folder.glob('*.py')} or any(digest(folder/f)!=h for f,h in expected.items()):raise ValueError('Source identity changed')
manifest=ROOT/'runs/fixed_gaussian_estimator_failure_v1/manifest.json'
p=subprocess.run([sys.executable,'-I','-m','estimator_failure_proof','verify-proof',str(manifest)],cwd=OUT,capture_output=True,timeout=60)
(OUT/'replay.stdout.log').write_bytes(p.stdout);(OUT/'replay.stderr.log').write_bytes(p.stderr)
if p.returncode:raise ValueError('Installed replay failed')
result=json.loads(p.stdout)
if len(result['cases'])!=2 or any(r['requested_recovery_success_probability_gate']!='refuted' or Q(r['family_failure_strict_lower'])<=Q(1,100) or r['all_estimators_refuted'] for r in result['cases'].values()):raise ValueError('Wrong refutation scope')
xml=OUT/'pytest.xml'
p=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_estimator_failure_proof.py'),'--import-mode=importlib','-q','--basetemp='+str(OUT/'temp_run_0000'),'--junitxml='+str(xml)],cwd=OUT,capture_output=True,timeout=120)
(OUT/'pytest.stdout.log').write_bytes(p.stdout);(OUT/'pytest.stderr.log').write_bytes(p.stderr)
suite=ET.parse(xml).getroot()[0]
if p.returncode or int(suite.get('tests'))!=16 or any(int(suite.get(k,'0')) for k in ('errors','failures','skipped')):raise ValueError('Installed tests failed')
wheel=ROOT/'validation/estimator_failure_dist_v1/ncd_estimator_failure_proof-0.1.0-py3-none-any.whl'
status={'schema':'ncd.estimator-failure-installed.v1','status':'verified','tests':16,'verification':result,'wheel_sha256':digest(wheel),'bundle_sha256':digest(manifest),'source_installed_bytes_equal':True,'source_checkout_not_on_import_path':True,'proof_kernel_requires_torch':False,'junit_sha256':digest(xml),'test_source_sha256':digest(ROOT/'tests/test_estimator_failure_proof.py'),'checker_sha256':digest(__file__),'original_objective_achieved':False}
(OUT/'status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('Two strict fixed-estimator failures independently replayed; 16 installed tests passed')
