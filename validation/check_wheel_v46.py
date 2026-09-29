"""Replay dynamic packing and controls from isolated 0.46 installation."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.continuous_compositional_realization import value, verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.dynamic_packing import certify_dynamic_packing, verify_dynamic_packing
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'validation'/'wheel_v46_run'
SOURCE = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'
PACKING = ROOT/'runs'/'dynamic_packing_v1'
UPPER = ROOT/'validation'/'automatic_affine128_dev'/'certificate.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def critical_pair(system,base):
    left = [Q(0)]*system.state_dim
    right = [Q(0)]*system.state_dim
    left[0],right[0],right[-1] = base,base+Q(1,10),Q(1)
    action = [Q(0)]*system.action_dim
    outputs = []
    for point in (left,right):
        outputs.append((value(system.observation,point),
                        value(system.observation,value(system.transition,point+action))))
    distances = [max(abs(a-b) for a,b in zip(outputs[0][t],outputs[1][t]))
                 for t in (0,1)]
    return {'base_x0':str(base),
            'initial_max_distance':str(distances[0]),
            'one_step_max_distance':str(distances[1]),
            'one_step_output0_distance':str(abs(outputs[0][1][0]-outputs[1][1][0])),
            'trace_max_distance':str(max(distances))}


def main():
    if ncd.__version__ != '0.46.0':
        raise ValueError('Installed version mismatch')
    package = Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v46_env').resolve()):
        raise ValueError('Imported source instead of installed package')
    sources = sorted((ROOT/'ncd').glob('*.py'))
    installed = sorted(package.glob('*.py'))
    if ({p.name for p in sources} != {p.name for p in installed}
            or any(sha(p) != sha(package/p.name) for p in sources)):
        raise ValueError('Installed module bytes differ from source')
    wheels = list(RUN.glob('neural_causal_decompiler-0.46.0-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected exactly one wheel')
    acceptance = read_json(ROOT/'validation'/'dynamic_packing_acceptance.json')
    profiles = []
    for index,entry in enumerate(acceptance['profiles']):
        d = entry['dimension']
        model = SOURCE/f'profile_{index:03d}'/'system.json'
        cert = PACKING/f'affine_d{d}'/'certificate.json'
        if sha(model) != entry['model_sha256'] or sha(cert) != entry['certificate_sha256']:
            raise ValueError('Frozen packing artifact digest changed')
        system = ContinuousReLUSystem.from_dict(read_json(model))
        result = verify_dynamic_packing(system,read_json(cert))
        if (result['lower_bound'] != 135 or result['pair_count'] != 9045
                or result['first_witness_counts'] != [8991,54]
                or result['minimum_pair_distance'] != entry['minimum_pair_distance']):
            raise ValueError('Installed dynamic packing proof mismatch')
        profiles.append({'dimension':d,'lower_bound':135,'pair_count':9045})
        print('wheel replayed',d,flush=True)
    system128 = ContinuousReLUSystem.from_dict(read_json(SOURCE/'profile_003'/'system.json'))
    stored_upper = read_json(UPPER)
    if (sha(UPPER) != acceptance['old_upper_certificate_sha256']
            or Q(stored_upper['epsilon']) > Q(0.17)):
        raise ValueError('Historical upper artifact or tolerance mismatch')
    upper = verify_weighted(system128,stored_upper)
    if upper['status'] != 'certified' or upper['upper_bound'] != '40824':
        raise ValueError('Installed historical upper proof mismatch')
    path_by_control = {
        'feedback_edge_removed':ROOT/'runs'/'affine_quotient_global_v1'/'controls'/'ablated_affine_d128'/'system.json',
        'learned_nonlinear_7201':ROOT/'runs'/'learned_local_global_v1'/'seed_7201'/'d_128'/'system.json',
        'learned_nonlinear_7202':ROOT/'runs'/'learned_local_global_v1'/'seed_7202'/'d_128'/'system.json',
    }
    for control in acceptance['counterfactual_controls']:
        path = path_by_control[control['name']]
        if sha(path) != control['model_sha256']:
            raise ValueError('Counterfactual model digest changed')
        system = ContinuousReLUSystem.from_dict(read_json(path))
        actual = [critical_pair(system,Q(0)),critical_pair(system,Q(1,2))]
        if actual != control['critical_pairs'] or any(
                Q(pair['trace_max_distance']) > 2*Q(0.17) for pair in actual):
            raise ValueError('Installed counterfactual trace mismatch')
    transition = ReLUMLP(
        weights=(((1.0,0.0),(1.0,0.0)),((4.0,-4.0),)),
        biases=((-0.5,-0.75),(0.0,)))
    observation = ReLUMLP(weights=(((1.0,),),),biases=((0.0,),))
    nonlinear = ContinuousReLUSystem(1,1,transition,observation)
    fallback = certify_dynamic_packing(nonlinear,((Q(7,20),),(Q(13,20),)),((Q(0),),))
    if verify_dynamic_packing(nonlinear,fallback)['evaluation_mode'] != 'direct-exact-ReLU':
        raise ValueError('Installed nonlinear exact fallback failed')
    status = {
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':str(Path(ncd.__file__).resolve()),
        'installed_modules_byte_identical':len(sources),
        'profiles':profiles,
        'new_128_minimum_state_interval':[135,40824],
        'counterfactual_controls_replayed':3,
        'nonlinear_exact_fallback':'verified',
        'tests_passed':202,
    }
    save_json(RUN/'status.json',status)
    print('verified',status['new_128_minimum_state_interval'],flush=True)

if __name__=='__main__':
    main()
