"""Exact one-step packing capacity across frozen affine ring profiles."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import Unresolved
from ncd.io import digest, read_json, save_json
from ncd.one_step_packing_capacity import (
    certify_one_step_capacity, certify_six_pair_packing,
    verify_one_step_capacity)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'
OUTPUT = ROOT/'runs'/'one_step_capacity_v1'
STATUS = ROOT/'validation'/'one_step_capacity_acceptance.json'
PREVIOUS = ROOT/'validation'/'weighted_grid_optimality_acceptance.json'
UPPER = ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
ABLATED = ROOT/'runs'/'affine_quotient_global_v1'/'controls'/'ablated_affine_d128'/'system.json'
DIMS = (8,32,64,128)


def _profile(index,dimension,generate):
    model = SOURCE/f'profile_{index:03d}'/'system.json'
    system = ContinuousReLUSystem.from_dict(read_json(model))
    if system.state_dim != dimension:
        raise ValueError('Frozen affine profile dimension mismatch')
    directory = OUTPUT/f'affine_d{dimension}'
    packing_path = directory/'packing_certificate.json'
    capacity_path = directory/'capacity_certificate.json'
    if generate:
        packing = certify_six_pair_packing(system)
        save_json(packing_path,packing)
        save_json(capacity_path,certify_one_step_capacity(system,packing))
    packing = read_json(packing_path)
    capacity = read_json(capacity_path)
    replay = verify_one_step_capacity(system,packing,capacity)
    if (replay['status'] != 'verified'
            or replay['exact_one_step_packing_capacity'] != 162
            or replay['exact_all_horizon_packing_capacity'] != 162
            or replay['lower_witness_pairs'] != 13041
            or capacity['lower_first_witness_counts'] != [12960,81]):
        raise ValueError('One-step capacity profile mismatch')
    return {'dimension':dimension,
            'model_sha256':digest(model),
            'packing_certificate_sha256':digest(packing_path),
            'capacity_certificate_sha256':digest(capacity_path),
            'exact_one_step_packing_capacity':162,
            'exact_all_horizon_packing_capacity':162,
            'first_witness_counts':capacity['lower_first_witness_counts']}


def main(verify=False):
    previous = read_json(PREVIOUS)
    model128 = SOURCE/'profile_003'/'system.json'
    if (previous['status'] != 'accepted'
            or previous['model_sha256'] != digest(model128)
            or previous['upper_certificate_sha256'] != digest(UPPER)):
        raise ValueError('Previous same-model upper changed')
    profiles=[]
    for index,dimension in enumerate(DIMS):
        item=_profile(index,dimension,not verify)
        profiles.append(item)
        print('replayed',dimension,item['exact_one_step_packing_capacity'],flush=True)
    system128 = ContinuousReLUSystem.from_dict(read_json(model128))
    upper_certificate=read_json(UPPER)
    upper=verify_weighted(system128,upper_certificate)
    if (upper['status']!='certified' or upper['upper_bound']!='27216'
            or Q(upper_certificate['epsilon'])>Q(0.17)):
        raise ValueError('Same-model upper replay failed')
    ablated=ContinuousReLUSystem.from_dict(read_json(ABLATED))
    try:
        certify_six_pair_packing(ablated)
    except Unresolved:
        ablated_fails=True
    else:
        raise ValueError('Ablated feedback control unexpectedly passes')
    summary={
        'schema':'ncd.one-step-packing-capacity-acceptance.v1',
        'status':'accepted',
        'epsilon_exact_float':str(Q(0.17)),
        'profiles':profiles,
        'upper_certificate_sha256':digest(UPPER),
        'feedback_ablated_model_sha256':digest(ABLATED),
        'feedback_ablated_six_pair_candidate_fails':ablated_fails,
        'exact_128_one_step_packing_capacity':162,
        'exact_128_all_horizon_packing_capacity':162,
        'new_128_minimum_state_interval':[162,27216],
        'boundary':'packing capacity for common-action output traces does not determine deterministic finite-machine minimality',
    }
    if verify:
        if read_json(STATUS)!=summary or read_json(OUTPUT/'summary.json')!=summary:
            raise ValueError('One-step capacity acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
        save_json(OUTPUT/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    main(args.verify)
