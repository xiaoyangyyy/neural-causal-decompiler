from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json,os,sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/faithfulness_boundary_installed_v1';OUT.mkdir(exist_ok=True)
TEMP=ROOT/'validation/task_temp_E_faithfulness_installed_v1';TEMP.mkdir(exist_ok=True)
os.environ.update(TEMP=str(TEMP),TMP=str(TEMP),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
os.environ.pop('NCD_FAITHFULNESS_TEST_SOURCE',None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
import faithfulness_boundary_proof as kernel
if kernel.__version__!='0.1.0' or not Path(kernel.__file__).resolve().is_relative_to(ROOT/'validation/faithfulness_boundary_env_v1'):
    raise ValueError('Not isolated installed kernel')
protocol=ROOT/'validation/faithfulness_boundary_protocol_v1.json';c=read(protocol)
loaded={p.name:sha(p) for p in Path(kernel.__file__).resolve().parent.glob('*.py')}
if loaded!=c['loaded_source_sha256'] or len(loaded)!=3:raise ValueError('Installed source differs')
if str(ROOT/'validation/faithfulness_boundary_package_v1') in sys.path:raise ValueError('Source checkout on import path')
def command(name,args):
    p=subprocess.run([sys.executable,'-I',*args],cwd=OUT,capture_output=True,timeout=180)
    (OUT/(name+'.stdout.log')).write_bytes(p.stdout);(OUT/(name+'.stderr.log')).write_bytes(p.stderr)
    if p.returncode:raise ValueError(name+' failed; retain logs')
    return p.stdout.decode('utf-8')
proof=json.loads(command('prove',['-m','faithfulness_boundary_proof','prove','--config',str(protocol)]))
bundle=ROOT/c['output']
replay=json.loads(command('verify',['-m','faithfulness_boundary_proof','verify-proof',str(bundle)]))
if proof!=replay or proof['jobs']!=6 or proof['original_objective_achieved']:raise ValueError('Independent public replay changed')
portable=OUT/'portable_bundle';shutil.copytree(bundle,portable)
if json.loads(command('portable_verify',['-m','faithfulness_boundary_proof','verify-proof',str(portable)]))!=proof:raise ValueError('Portable replay changed')
xml=OUT/'pytest.xml'
command('pytest',['-m','pytest',str(ROOT/'tests/test_faithfulness_boundary_v1.py'),'--import-mode=importlib','-q',
    '--basetemp='+str(OUT/'test_temp'),'--junitxml='+str(xml)])
suite=ET.parse(xml).getroot()[0]
if int(suite.get('tests'))!=68 or any(int(suite.get(k,'0')) for k in ('errors','failures','skipped')):raise ValueError('Installed regression incomplete')
wheel=next((ROOT/'validation/faithfulness_boundary_dist_v1').glob('*.whl'))
status={'schema':'ncd.faithfulness-boundary-installed.v1','status':'passed','tests':68,'source_tests':68,
    'source_installed_bytes_equal':True,'loaded_source_sha256':loaded,'no_neural_training':True,
    'floating_sampler_law_not_claimed':True,'scoped_counterexamples':6,'nodes':[3,5,8],
    'public_generation_independent_and_portable_replays_checked':True,'original_objective_achieved':False,
    'proof':proof,'junit_sha256':sha(xml),'source_final_junit_sha256':sha(ROOT/'validation/pytest_faithfulness_source_final_v1.xml'),
    'wheel':wheel.relative_to(ROOT).as_posix(),'wheel_sha256':sha(wheel),'protocol_sha256':sha(protocol),
    'manifest_sha256':sha(bundle/'manifest.json'),'test_source_sha256':sha(ROOT/'tests/test_faithfulness_boundary_v1.py')}
(OUT/'status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('Six rational population/faithful-finite-sample boundaries independently replayed;68 installed tests passed')
