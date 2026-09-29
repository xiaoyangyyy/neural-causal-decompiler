"""Exact co-design acceptance for state and action grid reachability."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.behavioral_cover import verify_behavioral_cover
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.dual_grid_reachability import (
    certify_dual_grid_closure,dual_grid_initial,dual_grid_output,
    dual_grid_step,verify_dual_grid_closure)
from ncd.io import digest,read_json,save_json
from ncd.reachable_two_stage import certify_two_stage

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
TWO_STAGE=ROOT/'runs'/'dual_grid_reachability_v1'/'affine_d128'/'two_stage_certificate.json'
CLOSURE=ROOT/'runs'/'dual_grid_reachability_v1'/'affine_d128'/'closure_certificate.json'
PACKING=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'packing_certificate.json'
CAPACITY=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'capacity_certificate.json'
COVER=ROOT/'runs'/'behavioral_cover_v1'/'affine_d128'/'certificate.json'
PREVIOUS=ROOT/'validation'/'correlated_reachability_acceptance.json'
PREVIOUS_CLOSURE=ROOT/'runs'/'correlated_reachability_v1'/'affine_d128'/'certificate.json'
STATUS=ROOT/'validation'/'dual_grid_reachability_acceptance.json'


def _smoke(system,recurrent,certificate):
    points=(
        tuple(Q(0) for _ in range(system.state_dim)),
        tuple(Q(1) for _ in range(system.state_dim)),
        tuple(Q(i%3,2) for i in range(system.state_dim)),
    )
    actions=(
        tuple(Q(0) for _ in range(system.action_dim)),
        tuple(Q(1) for _ in range(system.action_dim)),
        tuple(Q(1,2) for _ in range(system.action_dim)),
    )
    maximum=Q(0)
    for point in points:
        concrete=list(point)
        abstract=dual_grid_initial(system,point)
        for time in range(len(actions)+1):
            actual=value(system.observation,concrete)
            output=dual_grid_output(system,recurrent,certificate,abstract)
            maximum=max(maximum,max(abs(a-b) for a,b in zip(actual,output)))
            if maximum>Q(recurrent['epsilon']):
                raise ValueError('Dual-grid machine smoke exceeds tolerance')
            if time<len(actions):
                abstract=dual_grid_step(
                    system,recurrent,certificate,abstract,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
    return {'traces':len(points),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum)}


def main(verify=False):
    previous=read_json(PREVIOUS)
    if (previous['status']!='accepted'
            or previous['model_sha256']!=digest(MODEL)
            or previous['packing_certificate_sha256']!=digest(PACKING)
            or previous['capacity_certificate_sha256']!=digest(CAPACITY)
            or previous['cover_certificate_sha256']!=digest(COVER)
            or previous['correlated_certificate_sha256']!=digest(PREVIOUS_CLOSURE)
            or previous['new_upper_states']!=1142):
        raise ValueError('Prior same-model evidence changed')
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    lower=verify_behavioral_cover(
        system,read_json(PACKING),read_json(CAPACITY),read_json(COVER))
    if lower['packing_lower']!=162 or Q(recurrent['epsilon'])>Q(0.17):
        raise ValueError('Same-model lower/upper tolerance mismatch')
    if not verify:
        save_json(TWO_STAGE,certify_two_stage(system,recurrent))
    two_stage=read_json(TWO_STAGE)
    if not verify:
        save_json(CLOSURE,certify_dual_grid_closure(
            system,recurrent,two_stage))
    certificate=read_json(CLOSURE)
    result=verify_dual_grid_closure(system,recurrent,two_stage,certificate)
    expected={'status':'certified','action_bins':512,
              'pair_separator_count':60,
              'initial_state_count':243,'initial_candidate_edges':6480,
              'initial_retained_edges':2051,'initial_successor_count':780,
              'recurrent_candidate_edges':14696,
              'recurrent_retained_edges':4901,
              'recurrent_state_count':798,'upper_bound':1041}
    if result!=expected:
        raise ValueError('Dual-grid closure changed')
    smoke=_smoke(system,recurrent,certificate)
    summary={
        'schema':'ncd.dual-grid-reachability-acceptance.v1',
        'status':'accepted',
        'model_sha256':digest(MODEL),
        'recurrent_certificate_sha256':digest(RECURRENT),
        'two_stage_certificate_sha256':digest(TWO_STAGE),
        'closure_certificate_sha256':digest(CLOSURE),
        'packing_certificate_sha256':digest(PACKING),
        'capacity_certificate_sha256':digest(CAPACITY),
        'cover_certificate_sha256':digest(COVER),
        'epsilon_upper_exact_rational':recurrent['epsilon'],
        'epsilon_lower_exact_float':str(Q(0.17)),
        'action_bins':512,
        'initial_states':243,
        'initial_candidate_edges':6480,
        'initial_retained_edges':2051,
        'initial_successor_grid_states':780,
        'recurrent_candidate_edges':14696,
        'recurrent_retained_edges':4901,
        'closed_recurrent_states':798,
        'previous_upper_states':1142,
        'new_upper_states':1041,
        'new_minimum_state_interval':[162,1041],
        'executable_smoke':smoke,
        'boundary':'joint grid and pairwise support remain conservative',
    }
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Dual-grid acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',result['new_minimum_state_interval'],flush=True)
