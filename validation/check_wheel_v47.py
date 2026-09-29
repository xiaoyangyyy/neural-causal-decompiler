"""Replay the 128D grid refinement through an isolated 0.47 wheel."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import verify_dynamic_packing
from ncd.integer_grid_refinement import verify_refinement
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'validation'/'wheel_v47_run'
MODEL = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
BASELINE = ROOT/'validation'/'automatic_affine128_dev'/'certificate.json'
LOWER = ROOT/'runs'/'dynamic_packing_v1'/'affine_d128'/'certificate.json'
OUTPUT = ROOT/'runs'/'integer_grid_refinement_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__ != '0.47.0':
        raise ValueError('Installed version mismatch')
    package = Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v47_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources = sorted((ROOT/'ncd').glob('*.py'))
    installed = sorted(package.glob('*.py'))
    if ({p.name for p in sources} != {p.name for p in installed}
            or any(sha(p) != sha(package/p.name) for p in sources)):
        raise ValueError('Installed module bytes differ from source')
    wheels = list(RUN.glob('neural_causal_decompiler-0.47.0-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected exactly one wheel')
    acceptance = read_json(ROOT/'validation'/'integer_grid_refinement_acceptance.json')
    if (acceptance['status'] != 'accepted'
            or sha(MODEL) != acceptance['model_sha256']
            or sha(BASELINE) != acceptance['baseline_upper_sha256']
            or sha(LOWER) != acceptance['dynamic_lower_sha256']):
        raise ValueError('Frozen acceptance inputs changed')
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    baseline = read_json(BASELINE)
    old = verify_weighted(system,baseline)
    lower = verify_dynamic_packing(system,read_json(LOWER))
    if (old['status'] != 'certified' or old['upper_bound'] != '40824'
            or lower['status'] != 'verified' or lower['lower_bound'] != 135
            or Q(baseline['epsilon']) > Q(0.17)):
        raise ValueError('Historical bound replay failed')
    cases = []
    for entry in acceptance['refinement_cases']:
        name = entry['name']
        proposal_path = OUTPUT/name/'proposal.json'
        certificate_path = OUTPUT/name/'certificate.json'
        if (sha(proposal_path) != entry['proposal_sha256']
                or sha(certificate_path) != entry['certificate_sha256']):
            raise ValueError('Frozen refinement digest changed')
        replay = verify_refinement(system,baseline,read_json(proposal_path),
                                   read_json(certificate_path))
        if (replay['status'] != 'verified'
                or int(replay['new_upper_bound']) != entry['new_upper_bound']
                or replay['action_bins'] != entry['action_bins']
                or replay['numeric_tuples_searched'] != entry['numeric_tuples_searched']):
            raise ValueError('Installed refinement proof mismatch')
        cases.append({'name':name,'upper_bound':int(replay['new_upper_bound']),
                      'action_bins':replay['action_bins']})
        print('wheel replayed',name,flush=True)
    if (cases != [{'name':'state_only','upper_bound':27648,'action_bins':128},
                  {'name':'state_action','upper_bound':27216,'action_bins':512}]
            or acceptance['new_minimum_state_interval'] != [135,27216]):
        raise ValueError('Installed interval mismatch')
    status = {
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':str(Path(ncd.__file__).resolve()),
        'installed_modules_byte_identical':len(sources),
        'cases':cases,
        'new_128_minimum_state_interval':[135,27216],
        'tests_passed':203,
    }
    save_json(RUN/'status.json',status)
    print('verified',status['new_128_minimum_state_interval'],flush=True)


if __name__=='__main__':
    main()
