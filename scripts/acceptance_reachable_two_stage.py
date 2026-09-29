"""Exact two-stage finite machine with reachable recurrent-state pruning."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.behavioral_cover import verify_behavioral_cover
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import digest,read_json,save_json
from ncd.reachable_two_stage import (
    certify_two_stage,verify_two_stage,two_stage_initial,
    two_stage_output,two_stage_step)

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_only'/'certificate.json'
PACKING=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'packing_certificate.json'
CAPACITY=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'capacity_certificate.json'
COVER=ROOT/'runs'/'behavioral_cover_v1'/'affine_d128'/'certificate.json'
PRIOR=ROOT/'validation'/'behavioral_cover_acceptance.json'
CERT=ROOT/'runs'/'reachable_two_stage_v1'/'affine_d128'/'certificate.json'
STATUS=ROOT/'validation'/'reachable_two_stage_acceptance.json'


def _smoke(system,recurrent):
    samples=(
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
    for point in samples:
        concrete=list(point)
        abstract=two_stage_initial(system,point)
        for time in range(len(actions)+1):
            observed=value(system.observation,concrete)
            rendered=two_stage_output(system,recurrent,abstract)
            error=max(abs(x-y) for x,y in zip(observed,rendered))
            maximum=max(maximum,error)
            if error>Q(recurrent['epsilon']):
                raise ValueError('Executable two-stage smoke trace exceeded epsilon')
            if time<len(actions):
                action=actions[time]
                abstract=two_stage_step(system,recurrent,abstract,action)
                concrete=value(system.transition,concrete+list(action))
        if abstract[0]!='recurrent':
            raise ValueError('Machine did not enter recurrent phase')
    return {'traces':len(samples),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum)}


def main(verify=False):
    prior=read_json(PRIOR)
    if (prior['status']!='accepted'
            or prior['profiles'][-1]['model_sha256']!=digest(MODEL)
            or prior['profiles'][-1]['packing_certificate_sha256']!=digest(PACKING)
            or prior['profiles'][-1]['capacity_certificate_sha256']!=digest(CAPACITY)
            or prior['profiles'][-1]['cover_certificate_sha256']!=digest(COVER)):
        raise ValueError('Prior same-model lower evidence changed')
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    if (verify_weighted(system,recurrent)['status']!='certified'
            or Q(recurrent['epsilon'])>Q(0.17)):
        raise ValueError('Recurrent exact upper is incompatible')
    lower=verify_behavioral_cover(
        system,read_json(PACKING),read_json(CAPACITY),read_json(COVER))
    if lower['packing_lower']!=162:
        raise ValueError('Existing lower proof did not replay')
    if not verify:
        save_json(CERT,certify_two_stage(system,recurrent))
    certificate=read_json(CERT)
    upper=verify_two_stage(system,recurrent,certificate)
    if upper!={'status':'verified','initial_state_count':243,
               'recurrent_state_count':4480,'upper_bound':4723}:
        raise ValueError('Two-stage exact state count changed')
    smoke=_smoke(system,recurrent)
    summary={
        'schema':'ncd.reachable-two-stage-acceptance.v1',
        'status':'accepted',
        'model_sha256':digest(MODEL),
        'recurrent_certificate_sha256':digest(RECURRENT),
        'packing_certificate_sha256':digest(PACKING),
        'capacity_certificate_sha256':digest(CAPACITY),
        'cover_certificate_sha256':digest(COVER),
        'two_stage_certificate_sha256':digest(CERT),
        'epsilon_upper_exact_rational':recurrent['epsilon'],
        'epsilon_lower_exact_float':str(Q(0.17)),
        'initial_states':243,
        'reachable_recurrent_states':4480,
        'previous_upper_states':27216,
        'new_upper_states':4723,
        'new_minimum_state_interval':[162,4723],
        'executable_smoke':smoke,
        'scope':'all unit initial states, all continuous unit action words, all finite horizons',
    }
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Two-stage acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',result['new_minimum_state_interval'],flush=True)
