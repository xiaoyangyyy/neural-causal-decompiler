"""Isolated installed-wheel replay of exact optimal initial-grid proofs."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree as ET
import ncd
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.initial_grid_synthesis import (
    verify_initial_grid,initial_grid_initial,initial_grid_output,initial_grid_step)
from ncd.io import read_json,save_json
from ncd.reachable_two_stage import verify_two_stage
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation'/'wheel_v56_run'
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
CERT_ROOT=ROOT/'runs'/'optimal_initial_grid_v1'
OLD_CERT=ROOT/'runs'/'nonlinear_two_stage_v1'
ACCEPTANCE=ROOT/'validation'/'optimal_initial_grid_acceptance.json'
PRIOR=ROOT/'validation'/'nonlinear_two_stage_acceptance.json'
JUNIT=ROOT/'validation'/'pytest_v56.xml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.56.0' or version('neural-causal-decompiler')!='0.56.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v56_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if ({p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed modules differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.56.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one wheel')
    suites=ET.parse(JUNIT).getroot().findall('.//testsuite')
    tests=sum(int(s.get('tests','0')) for s in suites)
    if tests!=213 or any(int(s.get(key,'0')) for s in suites
                         for key in ('failures','errors','skipped')):
        raise ValueError('Complete regression evidence missing')
    acceptance=read_json(ACCEPTANCE)
    expected_pairs={(seed,d) for seed in (6101,6102) for d in (8,32,64,128)}
    if (acceptance['status']!='accepted' or len(acceptance['cases'])!=8
            or {(r['seed'],r['state_dim']) for r in acceptance['cases']}!=expected_pairs
            or acceptance['previous_acceptance_sha256']!=sha(PRIOR)):
        raise ValueError('Frozen eight-case acceptance missing')
    cases=[]
    for row in acceptance['cases']:
        seed,dimension=row['seed'],row['state_dim']
        source=STUDY/f'seed_{seed}'/f'd_{dimension}'
        model=source/'system.json'
        recurrent_path=source/'certificate.json'
        proof=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
        old_proof=OLD_CERT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
        for path,key in ((model,'model_sha256'),
                         (recurrent_path,'recurrent_certificate_sha256'),
                         (old_proof,'previous_two_stage_certificate_sha256'),
                         (proof,'initial_grid_certificate_sha256')):
            if sha(path)!=row[key]:
                raise ValueError('Frozen proof evidence changed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        recurrent=read_json(recurrent_path)
        certificate=read_json(proof)
        witness=phase_witnesses(system)
        lower=verify_weighted(system,recurrent)
        old=verify_two_stage(system,recurrent,read_json(old_proof))
        upper=verify_initial_grid(system,recurrent,certificate)
        initial=81 if dimension==8 else 108
        if (witness!=row['phase_witnesses']
                or any(abs(Q(x))<=Q(1,10000) for x in witness.values())
                or lower['status']!='certified' or lower['lower_bound']!=81
                or lower['upper_bound']!='2250'
                or old!={'status':'verified','initial_state_count':243,
                         'recurrent_state_count':216,'upper_bound':459}
                or upper!={'status':'verified','initial_state_count':initial,
                           'exact_minimum_initial_grid_count':initial,
                           'recurrent_state_count':216,'upper_bound':initial+216}):
            raise ValueError('Installed exact proof mismatch')
        point=tuple(Q(1) for _ in range(system.state_dim))
        concrete=list(point)
        state=initial_grid_initial(system,certificate,point)
        actions=tuple((a,)*system.action_dim for a in (Q(0),Q(1,128),Q(1,2),Q(127,128),Q(1)))
        for time in range(len(actions)+1):
            output=initial_grid_output(system,recurrent,certificate,state)
            actual=value(system.observation,concrete)
            if max(abs(a-b) for a,b in zip(output,actual))>Q(recurrent['epsilon']):
                raise ValueError('Installed machine smoke failed')
            if time<len(actions):
                state=initial_grid_step(system,recurrent,certificate,state,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
        cases.append({'seed':seed,'state_dim':dimension,
            'minimum_state_interval':[81,initial+216],
            'exact_minimum_initial_grid_count':initial})
    status={'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':(package/'__init__.py').relative_to(ROOT).as_posix(),
        'installed_modules_byte_identical':len(sources),
        'acceptance_sha256':sha(ACCEPTANCE),'junit_sha256':sha(JUNIT),
        'cases':cases,'case_count':len(cases),
        'all_phase_crossing_witnesses_replayed':True,
        'all_lower_and_upper_certificates_replayed':True,
        'all_initial_optimality_certificates_replayed':True,
        'installed_machine_smoke':'verified','tests_passed':tests}
    save_json(RUN/'status.json',status)
    print('verified',len(cases),'phase-crossing cases; exact initial grids 81/108',flush=True)


if __name__=='__main__':
    main()
