"""Isolated installed-wheel replay of nonlinear closures and portable programs."""
from __future__ import annotations
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
from ncd.nonlinear_action_closure import (
    verify_nonlinear_closure,verify_nonlinear_program,
    program_initial,program_output,program_step)
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation'/'wheel_v57_run'
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
INITIAL=ROOT/'runs'/'optimal_initial_grid_v1'
CERT_ROOT=ROOT/'runs'/'nonlinear_action_closure_v1'
ACCEPTANCE=ROOT/'validation'/'nonlinear_action_closure_acceptance.json'
PRIOR=ROOT/'validation'/'optimal_initial_grid_acceptance.json'
JUNIT=ROOT/'validation'/'pytest_v57.xml'
UPPERS={8:132,32:166,64:162,128:163}
RECURRENT={8:51,32:58,64:54,128:55}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.57.0' or version('neural-causal-decompiler')!='0.57.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v57_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if ({p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed modules differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.57.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one wheel')
    suites=ET.parse(JUNIT).getroot().findall('.//testsuite')
    tests=sum(int(s.get('tests','0')) for s in suites)
    if tests!=216 or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('Complete regression evidence missing')
    acceptance=read_json(ACCEPTANCE)
    pairs={(seed,d) for seed in (6101,6102) for d in (8,32,64,128)}
    if (acceptance['status']!='accepted' or len(acceptance['cases'])!=8
            or {(r['seed'],r['state_dim']) for r in acceptance['cases']}!=pairs
            or acceptance['previous_acceptance_sha256']!=sha(PRIOR)
            or not acceptance['all_standalone_programs_replayed']):
        raise ValueError('Frozen eight-case acceptance missing')
    cases=[]
    for row in acceptance['cases']:
        seed,dimension=row['seed'],row['state_dim']
        source=STUDY/f'seed_{seed}'/f'd_{dimension}'
        model=source/'system.json'
        recurrent_path=source/'certificate.json'
        initial_path=INITIAL/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
        target=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'
        proof=target/'certificate.json'
        program_path=target/'program.json'
        for path,key in ((model,'model_sha256'),(recurrent_path,'recurrent_certificate_sha256'),
                         (initial_path,'initial_grid_certificate_sha256'),
                         (proof,'closure_certificate_sha256'),(program_path,'program_sha256')):
            if sha(path)!=row[key]:
                raise ValueError('Frozen program evidence changed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        recurrent=read_json(recurrent_path)
        initial=read_json(initial_path)
        certificate=read_json(proof)
        program=read_json(program_path)
        witness=phase_witnesses(system)
        lower=verify_weighted(system,recurrent)
        old=verify_initial_grid(system,recurrent,initial)
        upper=verify_nonlinear_closure(system,recurrent,initial,certificate)
        portable=verify_nonlinear_program(system,recurrent,initial,certificate,program)
        count=81 if dimension==8 else 108
        if (witness!=row['phase_witnesses']
                or any(abs(Q(x))<=Q(1,10000) for x in witness.values())
                or lower['status']!='certified' or lower['lower_bound']!=81
                or old['upper_bound']!=(297 if dimension==8 else 324)
                or upper!={'status':'verified','initial_state_count':count,
                           'recurrent_state_count':RECURRENT[dimension],'upper_bound':UPPERS[dimension]}
                or portable!={'status':'verified','upper_bound':UPPERS[dimension],'action_template_count':1}):
            raise ValueError('Installed nonlinear program proof mismatch')
        point=(Q(1),)*system.state_dim
        concrete=list(point)
        state=program_initial(program,point)
        actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,128),Q(127,128)),(Q(1,2),Q(1,2)),(Q(1),Q(1)))
        for time in range(len(actions)+1):
            output=program_output(program,state)
            actual=value(system.observation,concrete)
            if max(abs(a-b) for a,b in zip(output,actual))>Q(recurrent['epsilon']):
                raise ValueError('Installed standalone program smoke failed')
            if time<len(actions):
                state=program_step(program,state,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
        cases.append({'seed':seed,'state_dim':dimension,
            'minimum_state_interval':[81,UPPERS[dimension]],
            'initial_states':count,'recurrent_states':RECURRENT[dimension],
            'program_sha256':sha(program_path),'action_template_count':1})
        print(f'verified installed program seed={seed} d={dimension}',flush=True)
    status={'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':(package/'__init__.py').relative_to(ROOT).as_posix(),
        'installed_modules_byte_identical':len(sources),
        'acceptance_sha256':sha(ACCEPTANCE),'junit_sha256':sha(JUNIT),
        'cases':cases,'case_count':len(cases),
        'all_phase_crossing_witnesses_replayed':True,
        'all_lower_initial_and_closure_certificates_replayed':True,
        'all_standalone_programs_replayed':True,
        'installed_standalone_program_smoke':'verified','tests_passed':tests}
    save_json(RUN/'status.json',status)
    print('verified',len(cases),'complete nonlinear closures and portable programs',flush=True)


if __name__=='__main__':main()
