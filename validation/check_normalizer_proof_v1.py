from pathlib import Path
from importlib.metadata import version
from copy import deepcopy
from xml.etree import ElementTree as ET
import json,subprocess,sys,torch
import ncd,normalizer_proof
from ncd.io import digest
from normalizer_proof.__main__ import verify_bundle,read
from normalizer_proof.realization import verify_normalizer
ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'validation/proof_extensions_env_v1'
RUN=ROOT/'validation/normalizer_proof_installed_v1';RUN.mkdir(exist_ok=True)
counts={}
for module in (ncd,normalizer_proof):
    package=Path(module.__file__).resolve().parent
    if not package.is_relative_to(ENV):raise ValueError('Source import')
    files=sorted((ROOT/package.name).glob('*.py'))
    if {p.name for p in files}!={p.name for p in package.glob('*.py')} or any(digest(p)!=digest(package/p.name) for p in files):raise ValueError('Installed bytes mismatch')
    counts[package.name]=len(files)
if version('ncd-normalizer-proof')!='0.1.0' or version('neural-causal-decompiler')!='0.59.0':raise ValueError('Installed version mismatch')
torch.set_num_threads(2)
state=torch.load(ROOT/'runs/oblique_seed1193/teacher.pt',map_location='cpu',weights_only=True)['state_dict']
if any(not bool(torch.isfinite(value).all()) for value in state.values()):raise ValueError('Non-real frozen parameters')
parameter_count=sum(value.numel() for value in state.values())
result=verify_bundle(ROOT/'runs/actual_normalizer_composition_v1/manifest.json')
print('actual network proof and 96 intervention cases independently replayed',flush=True)
cert=read(ROOT/'runs/actual_normalizer_composition_v1/certificate.json')
for field in ('weight','mask','closure'):
    forged=deepcopy(cert)
    if field=='weight':forged['fx_export']['checkpoint_sha256']='0'*64
    elif field=='mask':forged['masks'].pop()
    else:forged['original_R5_closed']=True
    try:verify_normalizer(forged)
    except ValueError:pass
    else:raise ValueError('Accepted '+field+' forgery')
xml=RUN/'pytest.xml'
code=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_normalizer_proof.py'),'--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=ROOT).returncode
suite=ET.parse(xml).getroot()[0]
if code or int(suite.get('tests'))!=12 or any(int(suite.get(k,'0')) for k in ('failures','errors','skipped')):raise ValueError('Installed tests incomplete')
audit=read(ROOT/'runs/actual_normalizer_composition_v1/intervention_audit.json')
status={'schema':'ncd.installed-normalizer-proof.v1','status':'verified','version':'0.1.0','core_version':'0.59.0',
    'all_frozen_parameters_finite':True,'parameter_count':parameter_count,
    'proof_verification':result,'installed_tests':12,'test_xml_sha256':digest(xml),'source_installed_byte_identity':True,'module_counts':counts,
    'weight_mismatch_rejected':True,'omitted_interventions_rejected':True,'false_original_closure_rejected':True,
    'intervention_executions':audit['executions'],'max_diagnostic_output_error':audit['max_final_absolute_error'],
    'wheel_sha256':digest(ROOT/'dist/ncd_normalizer_proof-0.1.0-py3-none-any.whl'),
    'original_claims_closed':0,'overall_objective_achieved':False}
(RUN/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print('installed normalizer proof accepted',flush=True)
