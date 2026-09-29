"""Full-cube two-stage realizations for eight trained phase-crossing rings."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path

from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem,_digest
from ncd.io import digest,read_json,save_json
from ncd.reachable_two_stage import (
    certify_two_stage,two_stage_initial,two_stage_output,two_stage_step,
    verify_two_stage)
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
PRIOR=ROOT/'validation'/'trained_nonlinear_global_acceptance.json'
CERT_ROOT=ROOT/'runs'/'nonlinear_two_stage_v1'
STATUS=ROOT/'validation'/'nonlinear_two_stage_acceptance.json'
SEEDS=(6101,6102)
DIMENSIONS=(8,32,64,128)


def _smoke(system,recurrent):
    points=(tuple(Q(0) for _ in range(system.state_dim)),
            tuple(Q(1) for _ in range(system.state_dim)),
            tuple(Q(i%3,2) for i in range(system.state_dim)))
    actions=(tuple(Q(0) for _ in range(system.action_dim)),
             tuple(Q(1) for _ in range(system.action_dim)),
             tuple(Q(1,2) for _ in range(system.action_dim)))
    maximum=Q(0)
    for point in points:
        concrete=list(point)
        abstract=two_stage_initial(system,point)
        for time in range(len(actions)+1):
            actual=value(system.observation,concrete)
            output=two_stage_output(system,recurrent,abstract)
            maximum=max(maximum,max(abs(a-b) for a,b in zip(actual,output)))
            if maximum>Q(recurrent['epsilon']):
                raise ValueError('Nonlinear two-stage smoke exceeded tolerance')
            if time<len(actions):
                abstract=two_stage_step(system,recurrent,abstract,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
    return {'traces':len(points),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum)}


def main(verify=False):
    previous=read_json(PRIOR)
    study=read_json(STUDY/'summary.json')
    if (previous['state']!='verified' or not previous['all_certificates_replayed']
            or not previous['all_frozen_gates_passed']
            or previous['summary_sha256']!=digest(STUDY/'summary.json')
            or study['accepted'] is not True
            or previous['seeds']!=list(SEEDS)
            or previous['dimensions']!=list(DIMENSIONS)):
        raise ValueError('Frozen trained nonlinear study is not accepted')
    cases=[]
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            row=next(x for x in study['cases']
                     if x['seed']==seed and x['state_dim']==dimension)
            source=STUDY/f'seed_{seed}'/f'd_{dimension}'
            model=source/'system.json'
            recurrent_path=source/'certificate.json'
            record=read_json(source/'record.json')
            system=ContinuousReLUSystem.from_dict(read_json(model))
            recurrent=read_json(recurrent_path)
            if (row['accepted'] is not True
                    or row['system_sha256']!=_digest(system.to_dict())
                    or row['certificate_sha256']!=digest(recurrent_path)
                    or record['training']['system_sha256']!=row['system_sha256']
                    or record['phase_witnesses']!=row['phase_witnesses']
                    or record['lower_bound']!=81
                    or int(record['upper_bound'])!=2250):
                raise ValueError('Frozen nonlinear case changed')
            witness=phase_witnesses(system)
            if (witness!=row['phase_witnesses']
                    or any(abs(Q(x))<=Q(1,10000) for x in witness.values())):
                raise ValueError('Actual ReLU phase crossings not verified')
            lower=verify_weighted(system,recurrent)
            if (lower['status']!='certified' or lower['lower_bound']!=81
                    or lower['upper_bound']!='2250'
                    or Q(recurrent['epsilon'])!=Q(17,100)):
                raise ValueError('Original nonlinear interval mismatch')
            target=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
            if not verify:
                save_json(target,certify_two_stage(system,recurrent))
            result=verify_two_stage(system,recurrent,read_json(target))
            if result!={'status':'verified','initial_state_count':243,
                        'recurrent_state_count':216,'upper_bound':459}:
                raise ValueError('Nonlinear two-stage result changed')
            smoke=_smoke(system,recurrent)
            cases.append({
                'seed':seed,'state_dim':dimension,
                'model_sha256':digest(model),
                'recurrent_certificate_sha256':digest(recurrent_path),
                'two_stage_certificate_sha256':digest(target),
                'phase_witnesses':witness,
                'old_minimum_state_interval':[81,2250],
                'new_minimum_state_interval':[81,459],
                'initial_states':243,'recurrent_states':216,
                'executable_smoke':smoke,
            })
            print(f'verified seed={seed} d={dimension} interval=[81,459]',flush=True)
    summary={
        'schema':'ncd.nonlinear-two-stage-acceptance.v1',
        'status':'accepted',
        'previous_study_sha256':digest(STUDY/'summary.json'),
        'cases':cases,
        'seeds':list(SEEDS),'dimensions':list(DIMENSIONS),
        'all_phase_crossings_replayed':True,
        'all_weighted_and_two_stage_certificates_replayed':True,
        'same_model_minimum_state_interval':[81,459],
        'scope':'eight frozen trained phase-crossing rings, full unit initial/action cubes, all finite horizons',
        'boundary':'finite realization upper; not the globally minimal causal quotient',
    }
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Nonlinear two-stage acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',result['same_model_minimum_state_interval'],flush=True)
