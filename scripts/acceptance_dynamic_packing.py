"""Replay one-step exact dynamic packing on frozen large neural systems."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path
from ncd.affine_observability import _affine_network
from ncd.continuous_compositional_realization import value, verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import certify_staggered_ring_packing, verify_dynamic_packing
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs' / 'certified_continuous_scale_seed4701' / 'profiles'
OUTPUT = ROOT / 'runs' / 'dynamic_packing_v1'
STATUS = ROOT / 'validation' / 'dynamic_packing_acceptance.json'
UPPER = ROOT / 'validation' / 'automatic_affine128_dev' / 'certificate.json'
ABLATED = ROOT / 'runs' / 'affine_quotient_global_v1' / 'controls' / 'ablated_affine_d128' / 'system.json'
DIMS = (8,32,64,128)


def _profile(index,dimension,generate):
    model = SOURCE / f'profile_{index:03d}' / 'system.json'
    system = ContinuousReLUSystem.from_dict(read_json(model))
    if system.state_dim != dimension:
        raise ValueError('Frozen scale profile dimension mismatch')
    target = OUTPUT / f'affine_d{dimension}' / 'certificate.json'
    if generate:
        save_json(target,certify_staggered_ring_packing(system))
    result = verify_dynamic_packing(system,read_json(target))
    if (result['lower_bound'] != 135 or result['pair_count'] != 9045
            or result['first_witness_counts'] != [8991,54]
            or result['evaluation_mode'] != 'globally-affine'):
        raise ValueError('Dynamic packing pattern mismatch')
    return {'dimension':dimension,'model_sha256':digest(model),
            'certificate_sha256':digest(target),
            'lower_bound':result['lower_bound'],
            'pair_count':result['pair_count'],
            'first_witness_counts':result['first_witness_counts'],
            'minimum_pair_distance':result['minimum_pair_distance'],
            'evaluation_mode':result['evaluation_mode']}


def _critical_pair(system,base):
    left = [Q(0)]*system.state_dim
    right = [Q(0)]*system.state_dim
    left[0] = base
    right[0] = base+Q(1,10)
    right[-1] = Q(1)
    action = [Q(0)]*system.action_dim
    y0_left = value(system.observation,left)
    y0_right = value(system.observation,right)
    y1_left = value(system.observation,value(system.transition,left+action))
    y1_right = value(system.observation,value(system.transition,right+action))
    initial = max(abs(a-b) for a,b in zip(y0_left,y0_right))
    next_distance = max(abs(a-b) for a,b in zip(y1_left,y1_right))
    return {'base_x0':str(base),
            'initial_max_distance':str(initial),
            'one_step_max_distance':str(next_distance),
            'one_step_output0_distance':str(abs(y1_left[0]-y1_right[0])),
            'trace_max_distance':str(max(initial,next_distance))}


def _controls():
    paths = {'feedback_edge_removed':ABLATED}
    for seed in (7201,7202):
        paths[f'learned_nonlinear_{seed}'] = (
            ROOT/'runs'/'learned_local_global_v1'/f'seed_{seed}'/'d_128'/'system.json')
    controls = []
    threshold = 2*Q(0.17)
    for name,path in paths.items():
        system = ContinuousReLUSystem.from_dict(read_json(path))
        pairs = [_critical_pair(system,Q(0)),_critical_pair(system,Q(1,2))]
        if any(Q(pair['trace_max_distance']) > threshold for pair in pairs):
            raise ValueError('Negative control unexpectedly separates the critical pair')
        item = {'name':name,'model_sha256':digest(path),
                'critical_pairs':pairs,
                'candidate_135_packing_fails':True}
        if name == 'feedback_edge_removed':
            transition,_,_ = _affine_network(system.transition)
            if transition[0][system.state_dim-1] != 0:
                raise ValueError('Ablated feedback edge remains active')
            item['effective_feedback_edge'] = '0'
        controls.append(item)
    return controls


def main(verify=False):
    profiles = []
    for index,dimension in enumerate(DIMS):
        profile = _profile(index,dimension,not verify)
        profiles.append(profile)
        print('replayed',dimension,profile['lower_bound'],flush=True)
    model128 = ContinuousReLUSystem.from_dict(
        read_json(SOURCE/'profile_003'/'system.json'))
    transition,_,_ = _affine_network(model128.transition)
    edge = transition[0][model128.state_dim-1]
    if edge != Q(0.3):
        raise ValueError('Original effective feedback edge changed')
    old_upper = read_json(UPPER)
    upper_result = verify_weighted(model128,old_upper)
    if (upper_result['status'] != 'certified'
            or upper_result['upper_bound'] != '40824'
            or Q(old_upper['epsilon']) > Q(0.17)):
        raise ValueError('Historical upper bound is not compatible')
    controls = _controls()
    summary = {
        'schema':'ncd.dynamic-packing-acceptance.v1',
        'status':'accepted',
        'epsilon_exact_float':str(Q(0.17)),
        'profiles':profiles,
        'original_128_feedback_edge':str(edge),
        'counterfactual_controls':controls,
        'old_upper_certificate_sha256':digest(UPPER),
        'new_128_minimum_state_interval':[135,40824],
        'scope':'all unit initial states, all continuous action words, all times',
        'boundary':'one-step witness does not separate the critical pair in ablated or learned nonlinear controls',
    }
    if verify:
        if read_json(STATUS) != summary or read_json(OUTPUT/'summary.json') != summary:
            raise ValueError('Frozen dynamic packing acceptance replay mismatch')
    else:
        save_json(OUTPUT/'summary.json',summary)
        save_json(STATUS,summary)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    main(args.verify)
