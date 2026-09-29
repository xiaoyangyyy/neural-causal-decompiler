"""Freeze and replay exact bounds for large separable continuous ReLU systems."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
from ncd.continuous_cover import benchmark_cover_system
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import digest, read_json, save_json
from ncd.separable_product_bounds import (
    Unresolved, certify_product_bounds, verify_product_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / 'validation' / 'exact_interval_lower_8.json.gz'
OUTPUT = ROOT / 'runs' / 'separable_product_bounds_v1'
STATUS = ROOT / 'validation' / 'separable_product_bounds_acceptance.json'
DIMS = (2, 8, 32, 128)
TRAINED = ROOT / 'runs' / 'learned_local_global_v1' / 'seed_7201' / 'd_128' / 'system.json'


def _scalar_proof():
    with gzip.open(PROOF,'rt',encoding='utf-8') as handle:
        return json.load(handle)


def _profile(d, proof, generate):
    directory = OUTPUT / f'd_{d:03d}'
    model_path = directory / 'system.json'
    cert_path = directory / 'certificate.json'
    if generate:
        save_json(model_path, benchmark_cover_system(d).to_dict())
    system = ContinuousReLUSystem.from_dict(read_json(model_path))
    if system.state_dim != d:
        raise ValueError('Stored profile dimension mismatch')
    if generate:
        save_json(cert_path, certify_product_bounds(system,proof))
    result = verify_product_bounds(system,proof,read_json(cert_path))
    return {
        'dimension':d,
        'system_sha256':digest(model_path),
        'certificate_sha256':digest(cert_path),
        'coefficients_checked':result['coefficients_checked'],
        'lower_bound':result['lower_bound'],
        'upper_bound':result['upper_bound'],
        'horizon':result['horizon'],
    }


def _representation_control(proof, generate):
    directory = OUTPUT / 'd_008_hidden_gauge'
    model_path = directory / 'system.json'
    cert_path = directory / 'certificate.json'
    if generate:
        data = deepcopy(benchmark_cover_system(8).to_dict())
        for key in ('transition','observation'):
            network = data[key]
            permutation = list(reversed(range(len(network['weights'][0]))))
            network['weights'][0] = [network['weights'][0][j] for j in permutation]
            network['biases'][0] = [network['biases'][0][j] for j in permutation]
            network['weights'][1] = [
                [row[j] for j in permutation] for row in network['weights'][1]]
            network['weights'][0] = [[2*x for x in row] for row in network['weights'][0]]
            network['biases'][0] = [2*x for x in network['biases'][0]]
            network['weights'][1] = [[x/2 for x in row] for row in network['weights'][1]]
        save_json(model_path,data)
    system = ContinuousReLUSystem.from_dict(read_json(model_path))
    if generate:
        save_json(cert_path,certify_product_bounds(system,proof))
    result = verify_product_bounds(system,proof,read_json(cert_path))
    if (result['dimension'] != 8 or result['lower_bound'] != '147456'
            or result['upper_bound'] != '43046721'):
        raise ValueError('Hidden-permuted product bounds changed')
    return {'system_sha256':digest(model_path),
            'certificate_sha256':digest(cert_path),
            'dimension':8,
            'lower_bound':result['lower_bound'],
            'upper_bound':result['upper_bound']}


def _negative_controls(proof):
    original = benchmark_cover_system(2)
    changed = deepcopy(original.to_dict())
    changed['transition']['weights'][1][0][1] = 0.015625
    changed['transition']['biases'][1][0] -= 0.015625
    coupled = ContinuousReLUSystem.from_dict(changed)
    try:
        certify_product_bounds(coupled,proof)
    except Unresolved:
        coupled_rejected = True
    else:
        coupled_rejected = False
    trained = ContinuousReLUSystem.from_dict(read_json(TRAINED))
    try:
        certify_product_bounds(trained,proof)
    except Unresolved:
        trained_rejected = True
    else:
        trained_rejected = False
    if not coupled_rejected or not trained_rejected:
        raise ValueError('Nonseparable negative control was incorrectly certified')
    return {'coupled_2d_rejected':coupled_rejected,
            'trained_128d_rejected':trained_rejected,
            'trained_128d_system_sha256':digest(TRAINED)}


def main(verify=False):
    proof = _scalar_proof()
    profiles = []
    for d in DIMS:
        profile = _profile(d,proof,not verify)
        profiles.append(profile)
        print('replayed',d,profile['lower_bound'],profile['upper_bound'],flush=True)
    representation = _representation_control(proof,not verify)
    if representation['system_sha256'] == profiles[1]['system_sha256']:
        raise ValueError('Hidden representation control did not change the network')
    print('replayed hidden gauge transform',flush=True)
    negative = _negative_controls(proof)
    summary = {
        'schema':'ncd.separable-product-acceptance.v1',
        'status':'accepted',
        'scalar_exclusion_sha256':digest(PROOF),
        'profiles':profiles,
        'representation_control':representation,
        'negative_controls':negative,
        'claim':'exact implicit finite-state bounds for frozen separable products',
        'boundary':'does not certify coupled or trained 128D networks',
    }
    if verify:
        if read_json(STATUS) != summary or read_json(OUTPUT/'summary.json') != summary:
            raise ValueError('Stored separable-product acceptance replay mismatch')
    else:
        save_json(OUTPUT/'summary.json',summary)
        save_json(STATUS,summary)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    main(args.verify)
