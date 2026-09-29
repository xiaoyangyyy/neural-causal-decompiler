from pathlib import Path
from importlib.metadata import version
from copy import deepcopy
import json,subprocess,sys,tempfile,torch
from xml.etree import ElementTree as ET
import ncd,proof_extensions
from ncd.io import digest
from proof_extensions.__main__ import verify_bundle,verify
from proof_extensions.statistics_rule import verify_rule_box
ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'validation/proof_extensions_env_v1'
RUN=ROOT/'validation/proof_extensions_installed_v1'
RUN.mkdir(exist_ok=True)
# Files used for evidence binding are read from root; imports must come from wheel.
for module in (ncd,proof_extensions):
    package=Path(module.__file__).resolve().parent
    if not package.is_relative_to(ENV):raise ValueError('Source import in installed replay')
    expected=sorted((ROOT/package.name).glob('*.py'))
    if {p.name for p in expected}!={p.name for p in package.glob('*.py')} or any(digest(p)!=digest(package/p.name) for p in expected):raise ValueError('Installed module bytes differ')
if version('ncd-proof-extensions')!='0.1.0' or version('neural-causal-decompiler')!='0.59.0':raise ValueError('Installed version mismatch')
torch.set_num_threads(2)
result=verify_bundle(ROOT/'runs/original_proof_extensions_bundle_v1/manifest.json')
print('five proof jobs replayed',flush=True)
old=ROOT/'runs/original_proof_milestone_v1/historical_discovery_local_box'
load=lambda p:json.loads(Path(p).read_text())
rule=verify_rule_box(load(old/'network.json'),load(old/'program.json'),load(ROOT/'runs/actual_statistics_rule_revalidation_v1/certificate.json'))
print('correct historical statistics Rule replayed',flush=True)
c=load(ROOT/'runs/original_proof_extensions_bundle_v1/feature_collision/certificate.json');forged=deepcopy(c);forged['strict_network_labels']=[0,0]
try:verify(forged)
except ValueError:pass
else:raise ValueError('Forged collision accepted')
with tempfile.TemporaryDirectory(dir=RUN,prefix='wrong_weights_') as folder:
    state=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=True)
    state['state_dict']['head.2.bias'][0]+=1
    wrong=Path(folder)/'discoverer.pt';torch.save(state,wrong)
    try:verify(c,str(wrong))
    except ValueError:pass
    else:raise ValueError('Wrong network weights accepted')
print('forged certificate and wrong weights rejected',flush=True)
xml=RUN/'pytest.xml'
code=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_original_proof_extensions.py'),
    '--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=ROOT).returncode
suite=ET.parse(xml).getroot()[0]
if code or int(suite.get('tests'))!=23 or any(int(suite.get(k,'0')) for k in ('failures','errors','skipped')):raise ValueError('Incomplete installed tests')
status={'schema':'ncd.installed-proof-extensions.v1','status':'verified','extension_version':'0.1.0','core_version':'0.59.0',
    'core_modules':131,'extension_modules':9,'imports':{'ncd':str(ncd.__file__),'proof_extensions':str(proof_extensions.__file__)},
    'proof_bundle':result,'actual_statistics_rule':rule,'certificate_tampering_rejected':True,'wrong_weights_rejected':True,
    'installed_tests':23,'test_xml_sha256':digest(xml),
    'extension_wheel_sha256':digest(ROOT/'dist/ncd_proof_extensions-0.1.0-py3-none-any.whl'),
    'rebuilt_core_wheel_sha256':digest(ROOT/'validation/proof_extensions_package_v1/core_dist/neural_causal_decompiler-0.59.0-py3-none-any.whl'),
    'core_module_bytes_identical_to_original_059_release':True,'original_claims_closed':0,'overall_objective_achieved':False}
(RUN/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print('installed acceptance verified',flush=True)
