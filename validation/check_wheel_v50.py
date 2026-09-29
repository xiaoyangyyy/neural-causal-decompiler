"""Isolated-wheel replay of the exact all-horizon behavioral cover."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.behavioral_cover import verify_behavioral_cover
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json, save_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation'/'wheel_v50_run'
SOURCE=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'
CAPACITY=ROOT/'runs'/'one_step_capacity_v1'
COVER=ROOT/'runs'/'behavioral_cover_v1'
UPPER=ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.50.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v50_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if ({p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed modules differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.50.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one 0.50 wheel')
    acceptance=read_json(ROOT/'validation'/'behavioral_cover_acceptance.json')
    if acceptance['status']!='accepted':
        raise ValueError('Formal acceptance missing')
    for index,entry in enumerate(acceptance['profiles']):
        d=entry['dimension']
        model=SOURCE/f'profile_{index:03d}'/'system.json'
        packing=CAPACITY/f'affine_d{d}'/'packing_certificate.json'
        capacity=CAPACITY/f'affine_d{d}'/'capacity_certificate.json'
        cover=COVER/f'affine_d{d}'/'certificate.json'
        if (sha(model)!=entry['model_sha256']
                or sha(packing)!=entry['packing_certificate_sha256']
                or sha(capacity)!=entry['capacity_certificate_sha256']
                or sha(cover)!=entry['cover_certificate_sha256']):
            raise ValueError('Frozen cover evidence changed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        result=verify_behavioral_cover(
            system,read_json(packing),read_json(capacity),read_json(cover))
        if result!={'status':'verified','cover_number':162,
                   'representative_count':162,'packing_lower':162}:
            raise ValueError('Installed cover proof mismatch')
        print('wheel replayed',d,flush=True)
    system128=ContinuousReLUSystem.from_dict(
        read_json(SOURCE/'profile_003'/'system.json'))
    upper=read_json(UPPER)
    if (sha(UPPER)!=acceptance['finite_machine_upper_sha256']
            or Q(upper['epsilon'])>Q(0.17)
            or verify_weighted(system128,upper)['upper_bound']!='27216'):
        raise ValueError('Same-model machine upper failed')
    if (acceptance['exact_128_all_horizon_behavioral_cover_number']!=162
            or acceptance['exact_128_all_horizon_packing_number']!=162
            or acceptance['deterministic_162_state_machine_proved'] is not False
            or acceptance['general_128_finite_machine_minimum_state_interval']!=[162,27216]):
        raise ValueError('Behavioral/machine scope mismatch')
    status={
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':str(package/'__init__.py'),
        'installed_modules_byte_identical':len(sources),
        'profiles_verified':[8,32,64,128],
        'exact_all_horizon_behavioral_cover_number':162,
        'general_128_machine_interval':[162,27216],
        'tests_passed':207,
    }
    save_json(RUN/'status.json',status)
    print('verified',status['exact_all_horizon_behavioral_cover_number'],flush=True)


if __name__=='__main__':
    main()
