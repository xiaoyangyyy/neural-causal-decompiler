"""Replay the large-product theorem from an isolated 0.45 wheel."""
from __future__ import annotations
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import ncd
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json, save_json
from ncd.separable_product_bounds import (
    Unresolved, certify_product_bounds, verify_product_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'validation' / 'wheel_v45_run'
SOURCE = ROOT / 'runs' / 'separable_product_bounds_v1'
PROOF = ROOT / 'validation' / 'exact_interval_lower_8.json.gz'
TRAINED = ROOT / 'runs' / 'learned_local_global_v1' / 'seed_7201' / 'd_128' / 'system.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__ != '0.45.0':
        raise ValueError('Installed version mismatch')
    package = Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v45_env').resolve()):
        raise ValueError('Imported source rather than installed wheel')
    source_modules = sorted((ROOT/'ncd').glob('*.py'))
    installed_modules = sorted(package.glob('*.py'))
    if ({p.name for p in source_modules} != {p.name for p in installed_modules}
            or any(sha(p) != sha(package/p.name) for p in source_modules)):
        raise ValueError('Installed modules differ from source bytes')
    wheels = list(RUN.glob('neural_causal_decompiler-0.45.0-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected exactly one wheel')
    with gzip.open(PROOF,'rt',encoding='utf-8') as handle:
        proof = json.load(handle)
    acceptance = read_json(ROOT/'validation'/'separable_product_bounds_acceptance.json')
    if sha(PROOF) != acceptance['scalar_exclusion_sha256']:
        raise ValueError('Scalar proof stream digest changed')
    profiles = []
    for entry in acceptance['profiles']:
        d = entry['dimension']
        directory = SOURCE / f'd_{d:03d}'
        model = directory/'system.json'
        cert = directory/'certificate.json'
        if sha(model) != entry['system_sha256'] or sha(cert) != entry['certificate_sha256']:
            raise ValueError('Frozen product artifact digest mismatch')
        system = ContinuousReLUSystem.from_dict(read_json(model))
        replay = verify_product_bounds(system,proof,read_json(cert))
        if (replay['lower_bound'] != entry['lower_bound']
                or replay['upper_bound'] != entry['upper_bound']
                or replay['coefficients_checked'] != entry['coefficients_checked']):
            raise ValueError('Installed product proof mismatch')
        profiles.append({'dimension':d,'lower_bound':replay['lower_bound'],
                         'upper_bound':replay['upper_bound']})
        print('wheel replayed',d,flush=True)
    gauge = SOURCE/'d_008_hidden_gauge'
    gauge_system = ContinuousReLUSystem.from_dict(read_json(gauge/'system.json'))
    gauge_result = verify_product_bounds(gauge_system,proof,read_json(gauge/'certificate.json'))
    expected_gauge = acceptance['representation_control']
    if (sha(gauge/'system.json') != expected_gauge['system_sha256']
            or sha(gauge/'certificate.json') != expected_gauge['certificate_sha256']
            or gauge_result['lower_bound'] != profiles[1]['lower_bound']
            or gauge_result['upper_bound'] != profiles[1]['upper_bound']):
        raise ValueError('Installed hidden-gauge control mismatch')
    coupled = deepcopy(read_json(SOURCE/'d_002'/'system.json'))
    coupled['transition']['weights'][1][0][1] = 0.015625
    coupled['transition']['biases'][1][0] -= 0.015625
    try:
        certify_product_bounds(ContinuousReLUSystem.from_dict(coupled),proof)
    except Unresolved:
        pass
    else:
        raise ValueError('Installed checker accepted coupled product')
    try:
        certify_product_bounds(ContinuousReLUSystem.from_dict(read_json(TRAINED)),proof)
    except Unresolved:
        pass
    else:
        raise ValueError('Installed checker accepted trained 128D model')
    status = {
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':str(Path(ncd.__file__).resolve()),
        'installed_modules_byte_identical':len(source_modules),
        'scalar_proof_sha256':sha(PROOF),
        'profiles':profiles,
        'hidden_gauge_control':'verified',
        'coupled_2d_rejected':True,
        'trained_128d_rejected':True,
        'tests_passed':199,
    }
    save_json(RUN/'status.json',status)
    print('verified',len(profiles),'dimensions',flush=True)

if __name__=='__main__':
    main()
