"""Aggregate checked milestone evidence without closing stronger originals."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
release=read('validation/wheel_v59_run/status.json')
assert release['status']=='verified' and release['regression_tests']==249 and release['installed_module_count']==131
assert len(release['mixed_cases'])==8 and release['legacy_phase_cases_replayed']==8
assert release['all_mixed_full_training_replayed'] and release['all_mixed_standalone_programs_executed']
assert release['confirmation_development_worlds_replayed']==3
assert sha('runs/original_proof_milestone_v1/config.json')==sha('validation/original_proof_milestone_protocol.json')
ledger=read('runs/original_proof_milestone_v1/ledger.json');proof=read('runs/original_proof_milestone_v1/summary.json')
assert len(ledger['claims'])==38 and len(proof['records'])==6
assert all(c[k] for c in ledger['claims'] for k in ('assumptions','target_network','input_domain','interventions','error_contract','proof_dependencies'))
groups={f'R{i}':{'proved':0,'refuted':0,'unresolved':0} for i in range(14)}
for c in ledger['claims']:groups[c['requirement']][c['status']]+=1
open_claims=[c['id'] for c in ledger['claims'] if c['status']=='unresolved']
assert len(open_claims)==36 and sum(c['status']=='refuted' for c in ledger['claims'])==2
value={'schema':'ncd.original-project-milestone-acceptance.v1','status':'first-milestone-accepted-original-open',
  'milestone':'atomic claim contracts, Gaussian boundary and frozen-network fidelity/CEGIS loop',
  'requirement_groups':groups,'claim_count':38,'unresolved_claims':open_claims,
  'scoped_proof_records':proof['records'],'regression_tests':249,'installed_module_count':131,
  'mixed_full_training_and_program_cases':8,'legacy_phase_cases':8,'development_confirmation_worlds_replayed':3,
  'overall_objective_achieved':False,'full_plan_implemented':False,
  'formal_confirmation_declared_worlds':300,'formal_confirmation_result':'pending',
  'bindings':{name:sha(name) for name in ['validation/wheel_v59_run/status.json','validation/pytest_v59.xml',
     'validation/original_proof_milestone_protocol.json','runs/original_proof_milestone_v1/manifest.json',
     'validation/original_confirmation_protocol.json','docs/ORIGINAL_PROOF_METHOD.md','docs/ORIGINAL_PROOF_CLAIMS.md']}}
(ROOT/'validation/original_project_acceptance.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(value['status'],len(open_claims),'original claims remain unresolved')
