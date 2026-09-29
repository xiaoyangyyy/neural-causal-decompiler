from pathlib import Path
from importlib.metadata import version
from xml.etree import ElementTree as ET
import importlib,json,subprocess,sys
import neural_boundary_proof
from ncd.io import read_json,digest,save_json
ROOT=Path(__file__).resolve().parents[1];ENV=ROOT/'validation/proof_extensions_env_v1';SNAP=ROOT/'validation/source_snapshot_059';OUT=ROOT/'validation/neural_boundary_installed_v1';OUT.mkdir(exist_ok=True)
assert Path(neural_boundary_proof.__file__).resolve().is_relative_to(ENV) and version('ncd-neural-boundary-proof')=='0.1.0'
c=read_json(ROOT/'validation/neural_boundary_protocol_v1.json')
for f,h in c['source_sha256'].items():assert digest(importlib.import_module(f[:-3].replace('/','.').removesuffix('.__init__')).__file__)==h and digest(SNAP/f)==h
r=subprocess.run([sys.executable,'-I','-m','neural_boundary_proof','prove','--config','validation/neural_boundary_protocol_v1.json','--resume'],cwd=SNAP,capture_output=True,timeout=120)
(OUT/'replay.stdout.log').write_bytes(r.stdout);(OUT/'replay.stderr.log').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode(errors='replace');result=json.loads(r.stdout)
assert result['conclusion']=='refuted' and result['target_reachability_proved'] and not result['score_equivariance_refuted']
print('Installed actual-network strict counterexample independently verified',flush=True)
xml=OUT/'pytest.xml';r=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_neural_boundary_proof.py'),'--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=SNAP)
suite=ET.parse(xml).getroot()[0];assert r.returncode==0 and int(suite.get('tests'))==5 and all(int(suite.get(k,'0'))==0 for k in ('errors','failures','skipped'))
save_json(OUT/'status.json',{'schema':'ncd.neural-boundary-installed.v1','status':'verified','verification':result,'tests':5,'source_installed_bytes_equal':True,'snapshot_replay':True,
 'wheel_sha256':digest(ROOT/'dist/ncd_neural_boundary_proof-0.1.0-py3-none-any.whl'),'bundle_sha256':digest(ROOT/'runs/actual_label_swap_boundary_v1/manifest.json'),
 'junit_sha256':digest(xml),'checker_sha256':digest(Path(__file__)),'test_source_sha256':digest(ROOT/'tests/test_neural_boundary_proof.py'),'original_objective_achieved':False})
print('Five installed tests passed; original audit integration remains pending',flush=True)
