from pathlib import Path
from importlib.metadata import version
from xml.etree import ElementTree as ET
import importlib,json,subprocess,sys
import neural_ssa_proof,ncd
from ncd.io import digest,read_json,save_json
ROOT=Path(__file__).resolve().parents[1];ENV=ROOT/'validation/proof_extensions_env_v1';SNAP=ROOT/'validation/source_snapshot_059';OUT=ROOT/'validation/neural_ssa_installed_v1';OUT.mkdir(exist_ok=True)
assert Path(neural_ssa_proof.__file__).resolve().is_relative_to(ENV) and Path(ncd.__file__).resolve().is_relative_to(ENV)
assert version('ncd-neural-ssa-proof')=='0.1.0' and ncd.__version__=='0.59.0'
protocol=read_json(ROOT/'validation/neural_ssa_protocol_v1.json')
for name,sha in protocol['source_sha256'].items():
 module=name[:-3].replace('/','.').removesuffix('.__init__');assert digest(importlib.import_module(module).__file__)==sha and digest(SNAP/name)==sha
for manifest in ('manifest.json','workbench_v1_source_manifest.json','normalizer_v1_source_manifest.json'):
 assert (SNAP/manifest).is_file()
def command(name,module,args,code=0):
 r=subprocess.run([sys.executable,'-I','-m',module,*args],cwd=SNAP,capture_output=True,timeout=300)
 (OUT/(name+'.stdout.log')).write_bytes(r.stdout);(OUT/(name+'.stderr.log')).write_bytes(r.stderr)
 if r.returncode!=code:raise ValueError(name+': '+r.stderr.decode(errors='replace'))
 return json.loads(r.stdout.decode('utf-8'))
r=command('prove-resume','neural_ssa_proof',['prove','--config','validation/neural_ssa_protocol_v1.json','--resume'])
assert r['status']=='verified' and len(r['jobs'])==2
print('Both installed proof jobs independently replayed',flush=True)
for job in protocol['jobs']:
 command('verify-'+job['id'],'neural_ssa_proof',['verify-proof','runs/full_neural_ssa_realization_v1/'+job['id']+'/certificate.json','--program','runs/full_neural_ssa_realization_v1/'+job['id']+'/program.json'])
input_path=OUT/'input.json';save_json(input_path,{'data':[[0,0]]*16})
executed=command('execute','neural_ssa_proof',['execute','--program','runs/full_neural_ssa_realization_v1/teacher1193/program.json','--input',str(input_path),'--output',str(OUT/'execution.json')])
assert len(executed['logits'])==4
strict=command('strict-audit','ncd',['audit-requirements','runs/original_proof_milestone_v1','--require-closed'],1)
assert not strict['overall_objective_achieved'] and len(strict['unresolved_claims'])==36
xml=OUT/'pytest.xml';code=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_neural_ssa_proof.py'),'--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=SNAP).returncode
suite=ET.parse(xml).getroot()[0]
assert code==0 and int(suite.get('tests'))==12 and all(int(suite.get(k,'0'))==0 for k in ('failures','errors','skipped'))
status={'schema':'ncd.neural-ssa-installed.v1','status':'verified-original-open','installed_source_bytes_equal':True,'snapshot_replay':True,'jobs':r['jobs'],'tests':12,
 'diagnostic_executions':96,'strict_original_audit_exit_code':1,'unresolved_original_claims':36,'whole_project_complete':False,'full_plan_implemented':False,
 'wheel_sha256':digest(ROOT/'dist/ncd_neural_ssa_proof-0.1.0-py3-none-any.whl'),'bundle_sha256':digest(ROOT/'runs/full_neural_ssa_realization_v1/manifest.json'),
 'junit_sha256':digest(xml),'checker_sha256':digest(Path(__file__)),'test_source_sha256':digest(ROOT/'tests/test_neural_ssa_proof.py')}
save_json(OUT/'status.json',status);print('Isolated full-network computation accepted, original completion rejected',flush=True)
