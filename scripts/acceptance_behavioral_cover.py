"""Exact all-horizon behavioral cover and matching packing lower."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.behavioral_cover import certify_behavioral_cover, verify_behavioral_cover
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import digest, read_json, save_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'
PACKING=ROOT/'runs'/'one_step_capacity_v1'
OUTPUT=ROOT/'runs'/'behavioral_cover_v1'
PREVIOUS=ROOT/'validation'/'one_step_capacity_acceptance.json'
UPPER=ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
STATUS=ROOT/'validation'/'behavioral_cover_acceptance.json'
DIMS=(8,32,64,128)


def _profile(index,dimension,previous,generate):
    model=SOURCE/f'profile_{index:03d}'/'system.json'
    packing=PACKING/f'affine_d{dimension}'/'packing_certificate.json'
    capacity=PACKING/f'affine_d{dimension}'/'capacity_certificate.json'
    certificate=OUTPUT/f'affine_d{dimension}'/'certificate.json'
    if (digest(model)!=previous['model_sha256']
            or digest(packing)!=previous['packing_certificate_sha256']
            or digest(capacity)!=previous['capacity_certificate_sha256']):
        raise ValueError('Prior packing-capacity evidence changed')
    system=ContinuousReLUSystem.from_dict(read_json(model))
    if system.state_dim!=dimension:
        raise ValueError('Frozen model dimension mismatch')
    packed=read_json(packing)
    capped=read_json(capacity)
    if generate:
        save_json(certificate,certify_behavioral_cover(system,packed,capped))
    replay=verify_behavioral_cover(
        system,packed,capped,read_json(certificate))
    if replay!={'status':'verified','cover_number':162,
               'representative_count':162,'packing_lower':162}:
        raise ValueError('Behavioral covering proof changed')
    return {'dimension':dimension,
            'model_sha256':digest(model),
            'packing_certificate_sha256':digest(packing),
            'capacity_certificate_sha256':digest(capacity),
            'cover_certificate_sha256':digest(certificate),
            'exact_all_horizon_behavioral_cover_number':162}


def main(verify=False):
    previous=read_json(PREVIOUS)
    if (previous['status']!='accepted'
            or previous['exact_128_all_horizon_packing_capacity']!=162):
        raise ValueError('Prior all-horizon packing acceptance changed')
    profiles=[]
    for index,dimension in enumerate(DIMS):
        entry=_profile(index,dimension,previous['profiles'][index],not verify)
        profiles.append(entry)
        print('replayed',dimension,entry['exact_all_horizon_behavioral_cover_number'],flush=True)
    model128=SOURCE/'profile_003'/'system.json'
    system128=ContinuousReLUSystem.from_dict(read_json(model128))
    upper=read_json(UPPER)
    if (digest(UPPER)!=previous['upper_certificate_sha256']
            or verify_weighted(system128,upper)['upper_bound']!='27216'
            or Q(upper['epsilon'])>Q(0.17)):
        raise ValueError('Same-model finite-machine upper failed')
    summary={
        'schema':'ncd.behavioral-cover-acceptance.v1',
        'status':'accepted',
        'epsilon_exact_float':str(Q(0.17)),
        'profiles':profiles,
        'finite_machine_upper_sha256':digest(UPPER),
        'exact_128_all_horizon_behavioral_cover_number':162,
        'exact_128_all_horizon_packing_number':162,
        'general_128_finite_machine_minimum_state_interval':[162,27216],
        'deterministic_162_state_machine_proved':False,
        'boundary':'behavioral trajectory centers are not a transition-closed finite machine',
    }
    if verify:
        if read_json(STATUS)!=summary or read_json(OUTPUT/'summary.json')!=summary:
            raise ValueError('Behavioral cover acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
        save_json(OUTPUT/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    main(args.verify)
