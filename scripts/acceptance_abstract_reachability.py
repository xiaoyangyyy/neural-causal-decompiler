"""Exact action-box graph closure of the two-stage neural realization."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.abstract_reachability import (
    certify_abstract_closure,graph_initial,graph_output,graph_step,
    verify_abstract_closure)
from ncd.behavioral_cover import verify_behavioral_cover
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import digest,read_json,save_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_only'/'certificate.json'
TWO_STAGE=ROOT/'runs'/'reachable_two_stage_v1'/'affine_d128'/'certificate.json'
PACKING=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'packing_certificate.json'
CAPACITY=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'capacity_certificate.json'
COVER=ROOT/'runs'/'behavioral_cover_v1'/'affine_d128'/'certificate.json'
PREVIOUS=ROOT/'validation'/'reachable_two_stage_acceptance.json'
CERT=ROOT/'runs'/'abstract_reachability_v1'/'affine_d128'/'certificate.json'
STATUS=ROOT/'validation'/'abstract_reachability_acceptance.json'


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
        abstract=graph_initial(system,point)
        for time in range(len(actions)+1):
            exact=value(system.observation,concrete)
            output=graph_output(system,recurrent,certificate,abstract)
            maximum=max(maximum,max(abs(a-b) for a,b in zip(exact,output)))
            if maximum>Q(recurrent['epsilon']):
                raise ValueError('Graph machine smoke trace exceeds tolerance')
            if time<len(actions):
                abstract=graph_step(
                    system,recurrent,certificate,abstract,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
    return {'traces':len(points),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum)}


def main(verify=False):
    previous=read_json(PREVIOUS)
    if (previous['status']!='accepted'
            or previous['model_sha256']!=digest(MODEL)
            or previous['recurrent_certificate_sha256']!=digest(RECURRENT)
            or previous['two_stage_certificate_sha256']!=digest(TWO_STAGE)
            or previous['packing_certificate_sha256']!=digest(PACKING)
            or previous['capacity_certificate_sha256']!=digest(CAPACITY)
            or previous['cover_certificate_sha256']!=digest(COVER)):
        raise ValueError('Prior lower/two-stage evidence changed')
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    two_stage=read_json(TWO_STAGE)
    lower=verify_behavioral_cover(
        system,read_json(PACKING),read_json(CAPACITY),read_json(COVER))
    if lower['packing_lower']!=162 or Q(recurrent['epsilon'])>Q(0.17):
        raise ValueError('Same-model lower/upper tolerance mismatch')
    if not verify:
        save_json(CERT,certify_abstract_closure(system,recurrent,two_stage))
    certificate=read_json(CERT)
    result=verify_abstract_closure(
        system,recurrent,two_stage,certificate)
    expected={'status':'verified','initial_state_count':243,
              'initial_successor_count':1952,'recurrent_state_count':2060,
              'closure_edge_count':51304,'upper_bound':2303}
    if result!=expected:
        raise ValueError('Action-box graph closure changed')
    smoke=_smoke(system,recurrent,certificate)
    summary={
        'schema':'ncd.abstract-reachability-acceptance.v1',
        'status':'accepted',
        'model_sha256':digest(MODEL),
        'recurrent_certificate_sha256':digest(RECURRENT),
        'two_stage_certificate_sha256':digest(TWO_STAGE),
        'packing_certificate_sha256':digest(PACKING),
        'capacity_certificate_sha256':digest(CAPACITY),
        'cover_certificate_sha256':digest(COVER),
        'abstract_closure_certificate_sha256':digest(CERT),
        'epsilon_upper_exact_rational':recurrent['epsilon'],
        'epsilon_lower_exact_float':str(Q(0.17)),
        'initial_states':243,
        'initial_successor_grid_states':1952,
        'closed_recurrent_states':2060,
        'conservative_abstract_edges':51304,
        'previous_upper_states':4723,
        'new_upper_states':2303,
        'new_minimum_state_interval':[162,2303],
        'executable_smoke':smoke,
        'boundary':'action-box graph is conservative; retained states need not all be concretely reachable',
    }
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Graph closure acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',result['new_minimum_state_interval'],flush=True)
