from pathlib import Path
from importlib.metadata import version
from xml.etree import ElementTree as ET
import json,subprocess,sys
import ncd
from ncd.io import digest,save_json,read_json
ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'validation/wheel_v60_env';SOURCE=ROOT/'validation/source_candidate_v60/ncd'
RUN=ROOT/'validation/wheel_v60_run';RUN.mkdir(exist_ok=True)
package=Path(ncd.__file__).resolve().parent
if not package.is_relative_to(ENV) or ncd.__version__!='0.60.0.dev1' or version('neural-causal-decompiler')!='0.60.0.dev1':raise ValueError('Installed candidate isolation/version mismatch')
files=sorted(SOURCE.glob('*.py'))
if {p.name for p in files}!={p.name for p in package.glob('*.py')} or any(digest(p)!=digest(package/p.name) for p in files):raise ValueError('Candidate source/installed bytes mismatch')
def command(name,arguments,expected_code=0):
    result=subprocess.run([sys.executable,'-I','-m','ncd',*arguments],cwd=ROOT,capture_output=True,timeout=1200)
    (RUN/(name+'.stdout.log')).write_bytes(result.stdout);(RUN/(name+'.stderr.log')).write_bytes(result.stderr)
    if result.returncode!=expected_code:raise ValueError(name+' CLI exit '+str(result.returncode)+' '+result.stderr.decode(errors='replace'))
    output=result.stdout.decode('utf-8');return json.loads(output[output.index('{'):])
protocol=ROOT/'validation/original_proof_registry_protocol_v2r1.json'
proof=command('prove',['prove','--config',str(protocol),'--resume'])
if proof['jobs']!=7 or proof['original_claim_counts']!={'proved':0,'refuted':2,'unresolved':36}:raise ValueError('Unified job/claim scope changed')
print('installed prove resumed all seven jobs, retaining prior revision',flush=True)
audit=command('audit',['audit-requirements',str(ROOT/'runs/original_proof_registry_v2r1'),'--require-closed'],1)
if len(audit['unresolved_claims'])!=36 or audit['overall_objective_achieved']:raise ValueError('Strict original completion audit failed')
print('installed strict completion audit correctly returned 1',flush=True)
xml=RUN/'pytest.xml'
code=subprocess.run([sys.executable,'-I','-m','pytest',str(ROOT/'tests/test_unified_proof_registry.py'),str(ROOT/'tests/test_cdir.py'),'--import-mode=importlib','-q','--junitxml='+str(xml)],cwd=ROOT).returncode
suite=ET.parse(xml).getroot()[0]
if code or int(suite.get('tests'))!=16 or any(int(suite.get(k,'0')) for k in ('failures','errors','skipped')):raise ValueError('Targeted candidate tests incomplete')
from ncd.original_confirmation import check_protocol
if len(check_protocol(read_json(ROOT/'validation/original_confirmation_protocol.json'),ROOT))!=300:raise ValueError('Live confirmation source changed')
status={'schema':'ncd.unified-proof-candidate-installed.v1','status':'candidate-verified-original-open','version':'0.60.0.dev1','modules':len(files),
    'source_installed_byte_identity':True,'installed_prove_cli':proof,'strict_audit_exit_code':1,'unresolved_original_claims':36,'targeted_tests':16,
    'all_previous_closed_jobs_independently_replayed':True,'history_preserved':True,'new_statistic_floor_and_binary_constant_semantics_tested':True,
    'descendant_memory_refusal_tested':True,'own_descendant_timeout_termination_tested':True,
    'historical_inference_threads':6,'new_confirmation_training_threads':2,
    'legacy_source_protocol_unchanged':True,'candidate_changes_promoted_to_live_core':False,'full_regression_completed':False,
    'wheel_sha256':digest(ROOT/'dist/neural_causal_decompiler-0.60.0.dev1-py3-none-any.whl'),
    'junit_sha256':digest(xml),'checker_sha256':digest(Path(__file__)),'bundle_summary_sha256':digest(ROOT/'runs/original_proof_registry_v2r1/summary.json'),
    'whole_project_complete':False,'full_plan_implemented':False}
save_json(RUN/'status.json',status)
print('isolated unified proof candidate accepted; full regression/promotion still pending',flush=True)
