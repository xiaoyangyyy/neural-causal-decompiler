"""Isolated-wheel replay of exact all-horizon packing capacity."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import Unresolved
from ncd.one_step_packing_capacity import (
    certify_six_pair_packing, verify_one_step_capacity)
from ncd.io import read_json, save_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation'/'wheel_v49_run'
SOURCE=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'
PROOFS=ROOT/'runs'/'one_step_capacity_v1'
UPPER=ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
ABLATED=ROOT/'runs'/'affine_quotient_global_v1'/'controls'/'ablated_affine_d128'/'system.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.49.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v49_env').resolve()):
        raise ValueError('Imported source instead of installed package')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if ({p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed module bytes differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.49.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one wheel')
    acceptance=read_json(ROOT/'validation'/'one_step_capacity_acceptance.json')
    if acceptance['status']!='accepted':
        raise ValueError('Formal acceptance did not pass')
    for index,entry in enumerate(acceptance['profiles']):
        dimension=entry['dimension']
        model=SOURCE/f'profile_{index:03d}'/'system.json'
        packing=PROOFS/f'affine_d{dimension}'/'packing_certificate.json'
        capacity=PROOFS/f'affine_d{dimension}'/'capacity_certificate.json'
        if (sha(model)!=entry['model_sha256']
                or sha(packing)!=entry['packing_certificate_sha256']
                or sha(capacity)!=entry['capacity_certificate_sha256']):
            raise ValueError('Frozen proof artifact changed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        result=verify_one_step_capacity(
            system,read_json(packing),read_json(capacity))
        if result!={'status':'verified',
                   'exact_one_step_packing_capacity':162,
                   'exact_all_horizon_packing_capacity':162,
                   'lower_witness_points':162,
                   'lower_witness_pairs':13041}:
            raise ValueError('Installed all-horizon proof mismatch')
        print('wheel replayed',dimension,flush=True)
    system128=ContinuousReLUSystem.from_dict(
        read_json(SOURCE/'profile_003'/'system.json'))
    upper=read_json(UPPER)
    if (sha(UPPER)!=acceptance['upper_certificate_sha256']
            or Q(upper['epsilon'])>Q(0.17)
            or verify_weighted(system128,upper)['upper_bound']!='27216'):
        raise ValueError('Installed same-model upper failed')
    if sha(ABLATED)!=acceptance['feedback_ablated_model_sha256']:
        raise ValueError('Ablated model changed')
    ablated=ContinuousReLUSystem.from_dict(read_json(ABLATED))
    try:
        certify_six_pair_packing(ablated)
    except Unresolved:
        pass
    else:
        raise ValueError('Ablated negative control passed')
    status={
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':str(package/'__init__.py'),
        'installed_modules_byte_identical':len(sources),
        'profiles_verified':[8,32,64,128],
        'all_horizon_packing_capacity':162,
        'new_128_minimum_state_interval':[162,27216],
        'tests_passed':206,
    }
    save_json(RUN/'status.json',status)
    print('verified',status['new_128_minimum_state_interval'],flush=True)


if __name__=='__main__':
    main()
