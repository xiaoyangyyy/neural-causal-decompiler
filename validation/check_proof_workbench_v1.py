from pathlib import Path
from importlib.metadata import version
from copy import deepcopy
from xml.etree import ElementTree as ET
import json,subprocess,sys,torch
import ncd,proof_extensions,proof_workbench
from ncd.io import digest
from proof_workbench.__main__ import verify_bundle,verify
ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'validation/proof_extensions_env_v1'
RUN=ROOT/'validation/proof_workbench_installed_v1';RUN.mkdir(exist_ok=True)
for module in (ncd,proof_extensions,proof_workbench):
    package=Path(module.__file__).resolve().parent
    if not package.is_relative_to(ENV):raise ValueError('Source import')
    files=sorted((ROOT/package.name).glob('*.py'))
    if {p.name for p in files}!={p.name for p in package.glob('*.py')} or any(digest(p)!=digest(package/p.name) for p in files):raise ValueError('Installed bytes mismatch')
if version('ncd-proof-workbench')!='0.1.0' or version('neural-causal-decompiler')!='0.59.0':raise ValueError('Installed version mismatch')
torch.set_num_threads(2)
result=verify_bundle(ROOT/'runs/original_proof_workbench_bundle_v1/manifest.json')
print('four proof jobs independently replayed',flush=True)
read=lambda p:json.loads(Path(p).read_text())
c=read(ROOT/'runs/original_proof_workbench_bundle_v1/finite_mdl/certificate.json');c['complete']=False
try:verify(c)
except ValueError:pass
else:raise ValueError('Forged enumeration completeness accepted')
c=read(ROOT/'runs/original_proof_workbench_bundle_v1/mean_information/certificate.json');c['statistical_success_probability_upper']='1'
try:verify(c)
except ValueError:pass
else:raise ValueError('Forged probability bound accepted')
xml=RUN/'pytest.xml'
code=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_semantic_proof_workbench.py'),
    str(ROOT/'tests/test_piecewise_mechanism_workbench.py'),'--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=ROOT).returncode
suite=ET.parse(xml).getroot()[0]
if code or int(suite.get('tests'))!=9 or any(int(suite.get(k,'0')) for k in ('failures','errors','skipped')):raise ValueError('Installed workbench tests incomplete')
status={'schema':'ncd.installed-proof-workbench.v1','status':'verified','version':'0.1.0','core_version':'0.59.0',
    'proof_jobs':result,'installed_tests':9,'test_xml_sha256':digest(xml),'source_installed_byte_identity':True,
    'workbench_modules':6,'enumeration_forgery_rejected':True,'probability_forgery_rejected':True,
    'wheel_sha256':digest(ROOT/'dist/ncd_proof_workbench-0.1.0-py3-none-any.whl'),
    'original_claims_closed':0,'overall_objective_achieved':False}
(RUN/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print('installed proof workbench accepted',flush=True)
