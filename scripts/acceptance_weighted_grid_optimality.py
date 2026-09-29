"""Replay exact optimality within the weighted uniform-grid certificate class."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import verify_dynamic_packing
from ncd.io import digest, read_json, save_json
from ncd.weighted_grid_optimality import (
    certify_weighted_grid_optimality, verify_weighted_grid_optimality)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
UPPER = ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
LOWER = ROOT/'runs'/'dynamic_packing_v1'/'affine_d128'/'certificate.json'
PREVIOUS = ROOT/'validation'/'integer_grid_refinement_acceptance.json'
CERTIFICATE = ROOT/'runs'/'weighted_grid_optimality_v1'/'certificate.json'
STATUS = ROOT/'validation'/'weighted_grid_optimality_acceptance.json'


def main(verify=False):
    previous = read_json(PREVIOUS)
    if (previous['status'] != 'accepted'
            or previous['model_sha256'] != digest(MODEL)
            or previous['dynamic_lower_sha256'] != digest(LOWER)
            or previous['refinement_cases'][-1]['certificate_sha256'] != digest(UPPER)):
        raise ValueError('Previous same-model interval evidence changed')
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    upper_certificate = read_json(UPPER)
    upper = verify_weighted(system, upper_certificate)
    lower = verify_dynamic_packing(system, read_json(LOWER))
    if (upper['status'] != 'certified' or upper['upper_bound'] != '27216'
            or lower['status'] != 'verified' or lower['lower_bound'] != 135
            or Q(upper_certificate['epsilon']) > Q(0.17)):
        raise ValueError('Finite-realization interval does not replay')
    if not verify:
        save_json(CERTIFICATE,
                  certify_weighted_grid_optimality(system, upper_certificate))
    certificate = read_json(CERTIFICATE)
    replay = verify_weighted_grid_optimality(system, upper_certificate, certificate)
    if (replay['status'] != 'verified'
            or replay['class_minimum_states'] != 27216
            or replay['horizon'] != 10
            or replay['exclusion_cases'] != 125):
        raise ValueError('Weighted-grid optimality proof changed')
    summary = {
        'schema':'ncd.weighted-grid-optimality-acceptance.v1',
        'status':'accepted',
        'model_sha256':digest(MODEL),
        'upper_certificate_sha256':digest(UPPER),
        'lower_certificate_sha256':digest(LOWER),
        'optimality_certificate_sha256':digest(CERTIFICATE),
        'epsilon_upper_exact_rational':upper_certificate['epsilon'],
        'epsilon_lower_exact_float':str(Q(0.17)),
        'weighted_uniform_grid_class_minimum_states':27216,
        'finite_realization_minimum_state_interval':[135,27216],
        'all_finite_realizations_optimum_proved':False,
        'finite_neumann_horizon':10,
        'no_extra_axis_exclusion_cases':125,
    }
    if verify:
        if read_json(STATUS) != summary:
            raise ValueError('Weighted-grid optimality acceptance mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified weighted-grid optimum',result['weighted_uniform_grid_class_minimum_states'])
