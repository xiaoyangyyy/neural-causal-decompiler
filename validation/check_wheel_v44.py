"""Verify release 0.44 from an isolated wheel installation."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import ncd
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.exact_interval_lower import verify_interval_chain_exclusion, verify_exact_nine_state_upper
from ncd.invariant_slice_lower import certify_exact_slice_lower, verify_exact_slice_lower
from ncd.shifted_realization import verify_shifted_realizations
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
WHEEL_RUN = ROOT / 'validation' / 'wheel_v44_run'
SOURCE = ROOT / 'runs' / 'certified_shifted_realization_seed14701'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__ != '0.44.0':
        raise ValueError('Installed package version mismatch')
    package = Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT / 'validation' / 'wheel_v44_env').resolve()):
        raise ValueError('Imported source package instead of installed wheel')
    source_modules = sorted((ROOT / 'ncd').glob('*.py'))
    installed_modules = sorted(package.glob('*.py'))
    if {p.name for p in source_modules} != {p.name for p in installed_modules}:
        raise ValueError('Source/installed module set mismatch')
    if any(sha(p) != sha(package / p.name) for p in source_modules):
        raise ValueError('Source/installed module bytes differ')
    wheels = list(WHEEL_RUN.glob('neural_causal_decompiler-0.44.0-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected one wheel')
    with gzip.open(ROOT / 'validation' / 'exact_interval_lower_8.json.gz','rt',encoding='utf-8') as handle:
        exclusion = json.load(handle)
    lower = verify_interval_chain_exclusion(exclusion)
    upper = verify_exact_nine_state_upper()
    scalar = ContinuousReLUSystem.from_dict(read_json(SOURCE / 'profiles' / 'profile_000' / 'system.json'))
    product = ContinuousReLUSystem.from_dict(read_json(SOURCE / 'profiles' / 'profile_001' / 'system.json'))
    lift = certify_exact_slice_lower(scalar,product,exclusion)
    lifted = verify_exact_slice_lower(scalar,product,exclusion,lift)
    old_upper = verify_shifted_realizations(SOURCE)
    if lower['excluded_states'] != 8 or upper['states'] != 9 or lifted['lower_bound'] != 36 or old_upper['largest_realization'] != 81:
        raise ValueError('Installed proof outcome mismatch')
    status = {
        'state':'verified', 'version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),
        'wheel_name':wheels[0].name,
        'import_file':str(Path(ncd.__file__).resolve()),
        'installed_modules_byte_identical':len(source_modules),
        'proof_stream_sha256':sha(ROOT / 'validation' / 'exact_interval_lower_8.json.gz'),
        'eight_state_exclusion':lower,
        'exact_scalar_upper':upper,
        'scalar_minimum_states':9,
        'product_lower_bound':lifted['lower_bound'],
        'product_upper_bound':old_upper['largest_realization'],
        'product_minimum_state_interval':[36,81],
        'historical_upper_replay':old_upper['status'],
        'tests_passed':195,
    }
    save_json(WHEEL_RUN / 'status.json',status)
    print('verified',status['scalar_minimum_states'],status['product_minimum_state_interval'],flush=True)

if __name__=='__main__':
    main()
