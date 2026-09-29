"""Independent installed-package replay for the original proof milestone."""
from pathlib import Path
from fractions import Fraction as Q
from importlib.metadata import version
from xml.etree import ElementTree as ET
import json
import sys
import importlib.util
import ncd
from ncd.io import read_json,save_json,digest
from ncd.original_proof_workflow import verify_proof,audit_requirements
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import verify_weighted
from ncd.phase_initial_handoff import verify_phase_handoff
from ncd.transition_consistency_lower import verify_transition_lower
from ncd.joint_phase_initial_handoff import verify_joint_handoff
from ncd.nonlinear_action_closure import verify_nonlinear_closure,verify_nonlinear_program
from ncd.trained_mixed_realization import mixed_witness
from ncd.original_confirmation import run_confirmation

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation/wheel_v59_run'


def main():
    if ncd.__version__!='0.59.0' or version('neural-causal-decompiler')!='0.59.0':raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to(ROOT/'validation/wheel_v59_env'):raise ValueError('Imported source rather than installed wheel')
    expected=read_json(RUN/'release_contract.json')
    sources=sorted((ROOT/'ncd').glob('*.py'));installed=sorted(package.glob('*.py'))
    if len(sources)!=expected['module_count'] or {p.name for p in sources}!={p.name for p in installed}:raise ValueError('Module set changed')
    if any(digest(p)!=digest(package/p.name) for p in sources):raise ValueError('Installed/source bytes differ')
    suites=ET.parse(ROOT/'validation/pytest_v59.xml').getroot().findall('.//testsuite')
    if sum(int(s.get('tests','0')) for s in suites)!=expected['test_count'] or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):raise ValueError('Incomplete regression evidence')
    proof=verify_proof(ROOT/'runs/original_proof_milestone_v1')
    audit=audit_requirements(ROOT/'runs/original_proof_milestone_v1')
    if proof['jobs']!=6 or audit['claim_count']!=38 or len(audit['unresolved_claims'])!=36 or audit['overall_objective_achieved']:raise ValueError('Original requirement scope was changed')
    spec=importlib.util.spec_from_file_location('installed_mixed_acceptance',ROOT/'scripts/acceptance_mixed_joint_phase.py')
    mixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(mixed)
    cases=[]
    acceptance=read_json(ROOT/'validation/mixed_joint_phase_acceptance.json')
    pairs={(seed,d) for seed in (7101,7102) for d in (8,32,64,128)}
    if acceptance['status']!='accepted' or len(acceptance['cases'])!=8 or {(r['seed'],r['state_dim']) for r in acceptance['cases']}!=pairs:raise ValueError('Mixed declared profiles changed')
    for row in acceptance['cases']:
        seed,d=row['seed'],row['state_dim'];base=ROOT/f'runs/trained_mixed_joint_v1/seed_{seed}/d_{d}'
        for name,key in (('system.json','model_sha256'),('training.json','training_sha256'),('checkpoint.npz','checkpoint_sha256'),
            ('recurrent_certificate.json','recurrent_certificate_sha256'),('mixed_witness.json','mixed_witness_sha256'),
            ('initial_certificate.json','initial_certificate_sha256'),('closure_certificate.json','closure_certificate_sha256'),
            ('program.json','program_sha256'),('lower_certificate.json','lower_certificate_sha256')):
            if digest(base/name)!=row[key]:raise ValueError('Mixed artifact identity changed')
        system,training=mixed.replay_training(base)
        recurrent=read_json(base/'recurrent_certificate.json');initial=read_json(base/'initial_certificate.json')
        verify_weighted(system,recurrent);verify_joint_handoff(system,recurrent,initial)
        closure=read_json(base/'closure_certificate.json');program=read_json(base/'program.json')
        verify_nonlinear_closure(system,recurrent,initial,closure)
        verify_nonlinear_program(system,recurrent,initial,closure,program)
        verify_transition_lower(system,read_json(base/'lower_certificate.json'))
        standalone=mixed.smoke(system,program,'17/100')
        if mixed_witness(system)!=read_json(base/'mixed_witness.json'):raise ValueError('Mixed phase witness changed')
        if row['minimum_state_interval']!=[82,135] or row['minimum_initial_label_count']!=81:raise ValueError('Mixed conclusions changed')
        cases.append({'seed':seed,'state_dim':d,'minimum_state_interval':[82,135],'action_templates':row['action_template_count'],'full_training_replayed':True,'standalone_execution':standalone})
        print('installed mixed replay',seed,d,flush=True)
    # Replay all eight historical phase proofs/programs using the new installed code.
    for seed in (6101,6102):
        for d in (8,32,64,128):
            suffix=Path(f'seed_{seed}/d_{d}')
            system=ContinuousReLUSystem.from_dict(read_json(ROOT/'runs/trained_nonlinear_global_v1'/suffix/'system.json'))
            recurrent=read_json(ROOT/'runs/trained_nonlinear_global_v1'/suffix/'certificate.json')
            base=ROOT/'runs/phase_initial_handoff_v1'/suffix
            initial=read_json(base/'initial_certificate.json');closure=read_json(base/'closure_certificate.json')
            verify_weighted(system,recurrent);verify_phase_handoff(system,recurrent,initial)
            verify_nonlinear_closure(system,recurrent,initial,closure)
            verify_nonlinear_program(system,recurrent,initial,closure,read_json(base/'program.json'))
            verify_transition_lower(system,read_json(ROOT/'runs/transition_consistency_lower_v1'/suffix/'certificate.json'))
    development=run_confirmation(ROOT/'validation/original_confirmation_smoke_v3_protocol.json',verify_only=True,seconds=300)
    if development['computed_worlds']!=3 or development['state']!='verified':raise ValueError('Installed confirmation development replay incomplete')
    wheels=list(RUN.glob('neural_causal_decompiler-0.59.0-*.whl'))
    if len(wheels)!=1:raise ValueError('Expected one installed wheel')
    result={'schema':'ncd.wheel-v59-replay.v1','status':'verified','version':ncd.__version__,
        'checker_sha256':digest(Path(__file__)),'mixed_training_replay_helper_sha256':digest(ROOT/'scripts/acceptance_mixed_joint_phase.py'),
        'confirmation_development_worlds_replayed':development['computed_worlds'],'formal_confirmation_not_completed':True,
        'installed_module_count':len(sources),'regression_tests':expected['test_count'],'all_modules_byte_identical':True,
        'original_proof':proof,'original_claim_count':38,'original_unresolved_claim_count':36,
        'mixed_cases':cases,'all_mixed_full_training_replayed':True,'all_mixed_standalone_programs_executed':True,'legacy_phase_cases_replayed':8,'whole_project_complete':False,
        'wheel_sha256':digest(wheels[0]),'junit_sha256':digest(ROOT/'validation/pytest_v59.xml'),
        'proof_manifest_sha256':digest(ROOT/'runs/original_proof_milestone_v1/manifest.json'),
        'mixed_acceptance_sha256':digest(ROOT/'validation/mixed_joint_phase_acceptance.json')}
    save_json(RUN/'status.json',result)
    print('installed replay verified; original requirements remain open',flush=True)

if __name__=='__main__':main()
