"""Certified state/action grid refinement for frozen affine 128D model."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path
from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dynamic_packing import verify_dynamic_packing
from ncd.integer_grid_refinement import certify_refinement, verify_refinement
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
BASELINE = ROOT/'validation'/'automatic_affine128_dev'/'certificate.json'
LOWER = ROOT/'runs'/'dynamic_packing_v1'/'affine_d128'/'certificate.json'
OUTPUT = ROOT/'runs'/'integer_grid_refinement_v1'
STATUS = ROOT/'validation'/'integer_grid_refinement_acceptance.json'
CASES = (('state_only',(1,),27648,128),('state_action',(1,4),27216,512))


def _case(system,baseline,name,multipliers,generate):
    directory = OUTPUT/name
    proposal_path = directory/'proposal.json'
    certificate_path = directory/'certificate.json'
    if generate:
        proposal,certificate = certify_refinement(
            system,baseline,action_multipliers=multipliers)
        if certificate is None:
            raise ValueError('Integer grid search did not produce a candidate')
        save_json(proposal_path,proposal)
        save_json(certificate_path,certificate)
    proposal = read_json(proposal_path)
    certificate = read_json(certificate_path)
    replay = verify_refinement(system,baseline,proposal,certificate)
    return {'name':name,
            'action_multipliers':list(multipliers),
            'proposal_sha256':digest(proposal_path),
            'certificate_sha256':digest(certificate_path),
            'old_upper_bound':int(replay['old_upper_bound']),
            'new_upper_bound':int(replay['new_upper_bound']),
            'action_bins':replay['action_bins'],
            'numeric_tuples_searched':replay['numeric_tuples_searched'],
            'coordinate_bins':certificate['coordinate_bins'],
            'status':replay['status']}


def main(verify=False):
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    baseline = read_json(BASELINE)
    old = verify_weighted(system,baseline)
    lower_certificate = read_json(LOWER)
    lower = verify_dynamic_packing(system,lower_certificate)
    if (old['status'] != 'certified' or old['upper_bound'] != '40824'
            or lower['status'] != 'verified' or lower['lower_bound'] != 135
            or Q(baseline['epsilon']) > Q(0.17)):
        raise ValueError('Historical upper or dynamic lower failed joint replay')
    cases = []
    for name,multipliers,target,action_bins in CASES:
        record = _case(system,baseline,name,multipliers,not verify)
        if (record['new_upper_bound'] != target
                or record['action_bins'] != action_bins
                or record['old_upper_bound'] != 40824):
            raise ValueError('Refined grid outcome changed')
        cases.append(record)
        print('replayed',name,target,flush=True)
    if cases[1]['new_upper_bound'] >= cases[0]['new_upper_bound']:
        raise ValueError('Action refinement did not improve the state-only control')
    summary = {
        'schema':'ncd.integer-grid-refinement-acceptance.v1',
        'status':'accepted',
        'model_sha256':digest(MODEL),
        'baseline_upper_sha256':digest(BASELINE),
        'dynamic_lower_sha256':digest(LOWER),
        'epsilon_lower_exact_float':str(Q(0.17)),
        'epsilon_upper_exact_rational':baseline['epsilon'],
        'previous_minimum_state_interval':[135,40824],
        'refinement_cases':cases,
        'new_minimum_state_interval':[135,27216],
        'upper_state_reduction_fraction':'1/3',
        'boundary':'numerical search is only a proposal; every upper is independently certified by exact rational replay',
    }
    if verify:
        if read_json(STATUS) != summary or read_json(OUTPUT/'summary.json') != summary:
            raise ValueError('Integer-grid acceptance replay mismatch')
    else:
        save_json(OUTPUT/'summary.json',summary)
        save_json(STATUS,summary)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    main(args.verify)
