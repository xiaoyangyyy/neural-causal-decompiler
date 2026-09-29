"""Isolated-wheel replay of eight trained nonlinear finite realizations."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json,save_json
from ncd.reachable_two_stage import (
    two_stage_initial,two_stage_output,two_stage_step,verify_two_stage)
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation'/'wheel_v55_run'
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
CERT_ROOT=ROOT/'runs'/'nonlinear_two_stage_v1'
ACCEPTANCE=ROOT/'validation'/'nonlinear_two_stage_acceptance.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.55.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v55_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if ({p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed modules differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.55.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one wheel')
    acceptance=read_json(ACCEPTANCE)
    if acceptance['status']!='accepted' or len(acceptance['cases'])!=8:
        raise ValueError('Eight-case acceptance missing')
    cases=[]
    for row in acceptance['cases']:
        seed,dimension=row['seed'],row['state_dim']
        source=STUDY/f'seed_{seed}'/f'd_{dimension}'
        model=source/'system.json'
        recurrent_path=source/'certificate.json'
        proof=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
        for path,key in ((model,'model_sha256'),
                         (recurrent_path,'recurrent_certificate_sha256'),
                         (proof,'two_stage_certificate_sha256')):
            if sha(path)!=row[key]:
                raise ValueError('Frozen nonlinear evidence changed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        recurrent=read_json(recurrent_path)
        witness=phase_witnesses(system)
        lower=verify_weighted(system,recurrent)
        upper=verify_two_stage(system,recurrent,read_json(proof))
        if (witness!=row['phase_witnesses']
                or any(abs(Q(x))<=Q(1,10000) for x in witness.values())
                or lower['lower_bound']!=81 or lower['upper_bound']!='2250'
                or upper!={'status':'verified','initial_state_count':243,
                           'recurrent_state_count':216,'upper_bound':459}):
            raise ValueError('Installed nonlinear proof mismatch')
        point=tuple(Q(1) for _ in range(system.state_dim))
        concrete=list(point)
        state=two_stage_initial(system,point)
        actions=(tuple(Q(0) for _ in range(system.action_dim)),
                 tuple(Q(1) for _ in range(system.action_dim)))
        for time in range(len(actions)+1):
            output=two_stage_output(system,recurrent,state)
            actual=value(system.observation,concrete)
            if max(abs(a-b) for a,b in zip(output,actual))>Q(recurrent['epsilon']):
                raise ValueError('Installed nonlinear machine smoke failed')
            if time<len(actions):
                state=two_stage_step(system,recurrent,state,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
        cases.append({'seed':seed,'state_dim':dimension,
                      'minimum_state_interval':[81,459]})
    status={
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':(package/'__init__.py').relative_to(ROOT).as_posix(),
        'installed_modules_byte_identical':len(sources),
        'cases':cases,'case_count':len(cases),
        'all_phase_crossing_witnesses_replayed':True,
        'all_lower_and_upper_certificates_replayed':True,
        'installed_machine_smoke':'verified',
        'tests_passed':212,
    }
    save_json(RUN/'status.json',status)
    print('verified',len(cases),'phase-crossing cases [81,459]',flush=True)


if __name__=='__main__':
    main()
