"""Installed-wheel replay of minimum initial labels and whole-machine lowers."""
from fractions import Fraction as Q
import hashlib
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree as ET
import ncd
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.initial_grid_synthesis import verify_initial_grid
from ncd.io import read_json,save_json
from ncd.nonlinear_action_closure import (verify_nonlinear_closure,verify_nonlinear_program,
    program_initial,program_output,program_step)
from ncd.phase_initial_handoff import verify_phase_handoff
from ncd.transition_consistency_lower import verify_transition_lower
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation/wheel_v58_run'
STUDY=ROOT/'runs/trained_nonlinear_global_v1'
OLD_INITIAL=ROOT/'runs/optimal_initial_grid_v1'
OLD_PROGRAM=ROOT/'runs/nonlinear_action_closure_v1'
CERT_ROOT=ROOT/'runs/phase_initial_handoff_v1'
LOWER_ROOT=ROOT/'runs/transition_consistency_lower_v1'
ACCEPTANCE=ROOT/'validation/phase_initial_handoff_acceptance.json'
PRIOR=ROOT/'validation/nonlinear_action_closure_acceptance.json'
LOWER_ACCEPTANCE=ROOT/'validation/transition_consistency_lower_acceptance.json'
JUNIT=ROOT/'validation/pytest_v58.xml'
UPPERS={8:132,32:139,64:135,128:136}
RECURRENT={8:51,32:58,64:54,128:55}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.58.0' or version('neural-causal-decompiler')!='0.58.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation/wheel_v58_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if (len(sources)!=118 or {p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed modules differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.58.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one wheel')
    suites=ET.parse(JUNIT).getroot().findall('.//testsuite')
    tests=sum(int(s.get('tests','0')) for s in suites)
    if tests!=221 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('Complete regression evidence missing')
    acceptance=read_json(ACCEPTANCE)
    lower_acceptance=read_json(LOWER_ACCEPTANCE)
    pairs={(seed,d) for seed in (6101,6102) for d in (8,32,64,128)}
    for data in (acceptance,lower_acceptance):
        if (data['status']!='accepted' or len(data['cases'])!=8
                or {(r['seed'],r['state_dim']) for r in data['cases']}!=pairs):
            raise ValueError('Frozen eight-case acceptance missing')
    if (acceptance['previous_acceptance_sha256']!=sha(PRIOR)
            or not acceptance['all_new_closed_graphs_and_standalone_programs_replayed']
            or not acceptance['all_initial_label_minima_proved']
            or not acceptance['all_legacy_graphs_and_programs_replayed']
            or lower_acceptance['previous_acceptance_sha256']!=sha(ACCEPTANCE)
            or not lower_acceptance['all_transition_consistency_lowers_replayed']):
        raise ValueError('Prior/upper/lower acceptance binding failed')
    cases=[]
    for row in acceptance['cases']:
        seed,d=row['seed'],row['state_dim']
        suffix=Path(f'seed_{seed}')/f'd_{d}'
        source=STUDY/suffix
        model=source/'system.json'
        recurrent_path=source/'certificate.json'
        old_initial_path=OLD_INITIAL/suffix/'certificate.json'
        old_proof=OLD_PROGRAM/suffix/'certificate.json'
        old_program_path=OLD_PROGRAM/suffix/'program.json'
        initial_path=CERT_ROOT/suffix/'initial_certificate.json'
        proof_path=CERT_ROOT/suffix/'closure_certificate.json'
        program_path=CERT_ROOT/suffix/'program.json'
        lower_path=LOWER_ROOT/suffix/'certificate.json'
        lower_row=next(r for r in lower_acceptance['cases'] if (r['seed'],r['state_dim'])==(seed,d))
        for path,key in ((model,'model_sha256'),(recurrent_path,'recurrent_certificate_sha256'),
            (old_initial_path,'legacy_initial_certificate_sha256'),(old_proof,'legacy_closure_certificate_sha256'),
            (old_program_path,'legacy_program_sha256'),(initial_path,'phase_initial_certificate_sha256'),
            (proof_path,'closure_certificate_sha256'),(program_path,'program_sha256')):
            if sha(path)!=row[key]:
                raise ValueError('Frozen program evidence changed')
        if (sha(lower_path)!=lower_row['lower_certificate_sha256']
                or sha(model)!=lower_row['model_sha256']
                or sha(program_path)!=lower_row['program_sha256']
                or lower_row['minimum_state_interval']!=[82,UPPERS[d]]
                or row['new_minimum_state_interval']!=[81,UPPERS[d]]):
            raise ValueError('Whole-machine interval binding failed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        recurrent=read_json(recurrent_path)
        old_initial=read_json(old_initial_path)
        old_certificate=read_json(old_proof)
        old_program=read_json(old_program_path)
        initial=read_json(initial_path)
        certificate=read_json(proof_path)
        program=read_json(program_path)
        witness=phase_witnesses(system)
        old=verify_initial_grid(system,recurrent,old_initial)
        old_upper=verify_nonlinear_closure(system,recurrent,old_initial,old_certificate)
        old_portable=verify_nonlinear_program(system,recurrent,old_initial,old_certificate,old_program)
        handoff=verify_phase_handoff(system,recurrent,initial)
        upper=verify_nonlinear_closure(system,recurrent,initial,certificate)
        portable=verify_nonlinear_program(system,recurrent,initial,certificate,program)
        if (witness!=row['phase_witnesses'] or any(abs(Q(x))<=Q(1,10000) for x in witness.values())
                or verify_weighted(system,recurrent)['lower_bound']!=81
                or old['upper_bound']!=(297 if d==8 else 324)
                or old_upper['upper_bound']!=row['old_minimum_state_interval'][1]
                or old_portable['status']!='verified'
                or handoff!={'status':'verified','initial_state_count':81,'recurrent_state_count':216,
                             'upper_bound':297,'minimum_initial_label_count':81}
                or verify_transition_lower(system,read_json(lower_path))!={'status':'verified','packing_lower_bound':81,'lower_bound':82}
                or upper!={'status':'verified','initial_state_count':81,'recurrent_state_count':RECURRENT[d],'upper_bound':UPPERS[d]}
                or portable!={'status':'verified','upper_bound':UPPERS[d],'action_template_count':1}):
            raise ValueError('Installed initial/whole-machine proof mismatch')
        point=(Q(1),)*system.state_dim
        concrete=list(point)
        state=program_initial(program,point)
        actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,128),Q(127,128)),(Q(1,2),Q(1,2)),(Q(1),Q(1)))
        for time in range(len(actions)+1):
            if max(abs(a-b) for a,b in zip(program_output(program,state),value(system.observation,concrete)))>Q(recurrent['epsilon']):
                raise ValueError('Installed standalone program smoke failed')
            if time<len(actions):
                state=program_step(program,state,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
        cases.append({'seed':seed,'state_dim':d,'minimum_state_interval':[82,UPPERS[d]],
            'minimum_initial_label_count':81,'initial_states':81,'recurrent_states':RECURRENT[d],
            'program_sha256':sha(program_path),'lower_certificate_sha256':sha(lower_path),'action_template_count':1})
        print(f'verified installed program/lower seed={seed} d={d}',flush=True)
    status={'state':'verified','version':ncd.__version__,'wheel_sha256':sha(wheels[0]),
        'wheel_name':wheels[0].name,'import_file':(package/'__init__.py').relative_to(ROOT).as_posix(),
        'installed_modules_byte_identical':len(sources),'acceptance_sha256':sha(ACCEPTANCE),
        'lower_acceptance_sha256':sha(LOWER_ACCEPTANCE),'junit_sha256':sha(JUNIT),
        'cases':cases,'case_count':len(cases),'all_phase_crossing_witnesses_replayed':True,
        'all_legacy_graphs_and_programs_replayed':True,'all_initial_label_minima_proved':True,
        'all_transition_consistency_lowers_replayed':True,'all_standalone_programs_replayed':True,
        'installed_standalone_program_smoke':'verified','tests_passed':tests}
    save_json(RUN/'status.json',status)
    print('verified',len(cases),'phase-aware portable programs and whole-machine lowers',flush=True)


if __name__=='__main__':main()
