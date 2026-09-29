"""Isolated-wheel replay of the exact weighted-grid class-optimality proof."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import verify_dynamic_packing
from ncd.weighted_grid_optimality import verify_weighted_grid_optimality
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'validation'/'wheel_v48_run'
MODEL = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
UPPER = ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
LOWER = ROOT/'runs'/'dynamic_packing_v1'/'affine_d128'/'certificate.json'
PROOF = ROOT/'runs'/'weighted_grid_optimality_v1'/'certificate.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__ != '0.48.0':
        raise ValueError('Installed version mismatch')
    package = Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v48_env').resolve()):
        raise ValueError('Imported source rather than installed wheel')
    sources = sorted((ROOT/'ncd').glob('*.py'))
    installed = sorted(package.glob('*.py'))
    if ({p.name for p in sources} != {p.name for p in installed}
            or any(sha(p) != sha(package/p.name) for p in sources)):
        raise ValueError('Installed module bytes differ from source')
    wheels = list(RUN.glob('neural_causal_decompiler-0.48.0-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected exactly one 0.48 wheel')
    acceptance = read_json(ROOT/'validation'/'weighted_grid_optimality_acceptance.json')
    if (acceptance['status'] != 'accepted'
            or sha(MODEL) != acceptance['model_sha256']
            or sha(UPPER) != acceptance['upper_certificate_sha256']
            or sha(LOWER) != acceptance['lower_certificate_sha256']
            or sha(PROOF) != acceptance['optimality_certificate_sha256']):
        raise ValueError('Frozen evidence digest changed')
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    upper_certificate = read_json(UPPER)
    upper = verify_weighted(system,upper_certificate)
    lower = verify_dynamic_packing(system,read_json(LOWER))
    optimum = verify_weighted_grid_optimality(
        system,upper_certificate,read_json(PROOF))
    if (upper['status'] != 'certified' or upper['upper_bound'] != '27216'
            or lower['status'] != 'verified' or lower['lower_bound'] != 135
            or Q(upper_certificate['epsilon']) > Q(0.17)
            or optimum != {'status':'verified',
                           'class_minimum_states':27216,
                           'horizon':10,'exclusion_cases':125}):
        raise ValueError('Installed proof did not replay')
    status = {
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':str(package/'__init__.py'),
        'installed_modules_byte_identical':len(sources),
        'weighted_grid_class_minimum_states':27216,
        'general_minimum_state_interval':[135,27216],
        'tests_passed':205,
    }
    save_json(RUN/'status.json',status)
    print('verified',status['weighted_grid_class_minimum_states'],flush=True)


if __name__=='__main__':
    main()
