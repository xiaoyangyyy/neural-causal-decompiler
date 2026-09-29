"""Replay the eight-state exclusion and its two-dimensional slice lift."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.exact_interval_lower import verify_interval_chain_exclusion, verify_exact_nine_state_upper
from ncd.invariant_slice_lower import certify_exact_slice_lower, verify_exact_slice_lower
from ncd.shifted_realization import verify_shifted_realizations, verify_shifted_realization
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs' / 'certified_shifted_realization_seed14701'
PROOF = ROOT / 'validation' / 'exact_interval_lower_8.json.gz'
OUTPUT = ROOT / 'runs' / 'exact_interval_lower_v1'
STATUS = ROOT / 'validation' / 'exact_interval_lower_acceptance.json'


def main():
    scalar_path = SOURCE / 'profiles' / 'profile_000' / 'system.json'
    product_path = SOURCE / 'profiles' / 'profile_001' / 'system.json'
    scalar = ContinuousReLUSystem.from_dict(read_json(scalar_path))
    product = ContinuousReLUSystem.from_dict(read_json(product_path))
    with gzip.open(PROOF, 'rt', encoding='utf-8') as handle:
        exclusion = json.load(handle)
    scalar_check = verify_interval_chain_exclusion(exclusion)
    exact_upper = verify_exact_nine_state_upper()
    upper_run = verify_shifted_realizations(SOURCE)
    scalar_upper = verify_shifted_realization(
        scalar, read_json(SOURCE / 'profiles' / 'profile_000' / 'profile.json')['certificate'])
    product_upper = verify_shifted_realization(
        product, read_json(SOURCE / 'profiles' / 'profile_001' / 'profile.json')['certificate'])
    if (scalar_check['excluded_states'] != 8
            or scalar_upper['upper_bound'] != 9
            or exact_upper['states'] != 9
            or product_upper['upper_bound'] != 81):
        raise ValueError('Historical upper bound or exact exclusion failed replay')
    lift = certify_exact_slice_lower(scalar, product, exclusion)
    lift_check = verify_exact_slice_lower(scalar, product, exclusion, lift)
    if lift_check['lower_bound'] != 36 or lift_check['slice_count'] != 4:
        raise ValueError('Exact slice lift did not reach 36 states')
    save_json(OUTPUT / 'certificate.json', lift)
    if verify_exact_slice_lower(scalar, product, exclusion,
                                read_json(OUTPUT / 'certificate.json')) != lift_check:
        raise ValueError('Stored exact slice certificate failed replay')
    summary = {
        'schema': 'ncd.exact-interval-lower-acceptance.v1',
        'status': 'accepted',
        'scalar_model_sha256': digest(scalar_path),
        'product_model_sha256': digest(product_path),
        'scalar_exclusion_sha256': digest(PROOF),
        'scalar_exclusion_check': scalar_check,
        'exact_scalar_upper': exact_upper,
        'slice_certificate_sha256': digest(OUTPUT / 'certificate.json'),
        'scalar_minimum_states': 9,
        'product_lower_bound': 36,
        'product_upper_bound': 81,
        'product_minimum_state_interval': [36,81],
        'upper_manifest_sha256': digest(SOURCE / 'manifest.json'),
        'upper_run_status': upper_run['status'],
    }
    save_json(OUTPUT / 'summary.json', summary)
    save_json(STATUS, summary)
    print(summary['scalar_minimum_states'],summary['product_minimum_state_interval'],flush=True)

if __name__=='__main__':
    main()
