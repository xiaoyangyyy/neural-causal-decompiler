"""Exact optimal initial grids for eight frozen phase-crossing rings."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.initial_grid_synthesis import (
    certify_initial_grid,verify_initial_grid,initial_grid_initial,
    initial_grid_output,initial_grid_step)
from ncd.io import digest,read_json,save_json
from ncd.reachable_two_stage import verify_two_stage
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
PRIOR=ROOT/'validation'/'nonlinear_two_stage_acceptance.json'
OLD_CERT=ROOT/'runs'/'nonlinear_two_stage_v1'
CERT_ROOT=ROOT/'runs'/'optimal_initial_grid_v1'
STATUS=ROOT/'validation'/'optimal_initial_grid_acceptance.json'
SEEDS=(6101,6102)
DIMENSIONS=(8,32,64,128)


def _smoke(system,recurrent,certificate):
    points=(tuple(Q(0) for _ in range(system.state_dim)),
            tuple(Q(1) for _ in range(system.state_dim)),
            tuple(Q(i%5,4) for i in range(system.state_dim)))
    actions=tuple(tuple(Q(a) for _ in range(system.action_dim))
                  for a in (Q(0),Q(1,128),Q(1,2),Q(127,128),Q(1)))
    maximum=Q(0)
    for point in points:
        concrete=list(point)
        abstract=initial_grid_initial(system,certificate,point)
        for time in range(len(actions)+1):
            actual=value(system.observation,concrete)
            output=initial_grid_output(system,recurrent,certificate,abstract)
            maximum=max(maximum,max(abs(a-b) for a,b in zip(actual,output)))
            if maximum>Q(recurrent['epsilon']):
                raise ValueError('Initial-grid executable smoke exceeded tolerance')
            if time<len(actions):
                abstract=initial_grid_step(system,recurrent,certificate,abstract,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
    return {'traces':len(points),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum)}


def main(verify=False):
    previous=read_json(PRIOR)
    if (previous['status']!='accepted' or len(previous['cases'])!=8
            or previous['seeds']!=list(SEEDS)
            or previous['dimensions']!=list(DIMENSIONS)
            or not previous['all_phase_crossings_replayed']
            or not previous['all_weighted_and_two_stage_certificates_replayed']):
        raise ValueError('Previous eight-case acceptance missing')
    cases=[]
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            row=next(x for x in previous['cases']
                     if x['seed']==seed and x['state_dim']==dimension)
            source=STUDY/f'seed_{seed}'/f'd_{dimension}'
            model=source/'system.json'
            recurrent_path=source/'certificate.json'
            old_proof=OLD_CERT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
            for path,key in ((model,'model_sha256'),
                             (recurrent_path,'recurrent_certificate_sha256'),
                             (old_proof,'two_stage_certificate_sha256')):
                if digest(path)!=row[key]:
                    raise ValueError('Frozen previous evidence changed')
            system=ContinuousReLUSystem.from_dict(read_json(model))
            recurrent=read_json(recurrent_path)
            witness=phase_witnesses(system)
            if (system.state_dim!=dimension or witness!=row['phase_witnesses']
                    or any(abs(Q(x))<=Q(1,10000) for x in witness.values())):
                raise ValueError('Actual nonlinear phase witnesses changed')
            lower=verify_weighted(system,recurrent)
            old=verify_two_stage(system,recurrent,read_json(old_proof))
            if (lower['status']!='certified' or lower['lower_bound']!=81
                    or lower['upper_bound']!='2250'
                    or old!={'status':'verified','initial_state_count':243,
                             'recurrent_state_count':216,'upper_bound':459}):
                raise ValueError('Original lower/upper replay mismatch')
            target=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
            if not verify:
                save_json(target,certify_initial_grid(system,recurrent))
            certificate=read_json(target)
            result=verify_initial_grid(system,recurrent,certificate)
            initial=81 if dimension==8 else 108
            expected={'status':'verified','initial_state_count':initial,
                      'exact_minimum_initial_grid_count':initial,
                      'recurrent_state_count':216,'upper_bound':initial+216}
            if (result!=expected or certificate['initial_coordinate_bins'][-1]!=1
                    or certificate['baseline_handoff_excluded_coordinates']!=([] if dimension==8 else [0])):
                raise ValueError('Exact initial-grid result changed')
            slack=min(Q(r)-Q(e) for r,e in zip(
                certificate['recurrent_relation_radii'],
                certificate['initial_to_recurrent_error_upper']))
            cases.append({'seed':seed,'state_dim':dimension,
                'model_sha256':digest(model),
                'recurrent_certificate_sha256':digest(recurrent_path),
                'previous_two_stage_certificate_sha256':digest(old_proof),
                'initial_grid_certificate_sha256':digest(target),
                'phase_witnesses':witness,
                'old_minimum_state_interval':[81,459],
                'new_minimum_state_interval':[81,initial+216],
                'initial_states':initial,'recurrent_states':216,
                'exact_minimum_initial_grid_count':initial,
                'baseline_handoff_excluded_coordinates':certificate['baseline_handoff_excluded_coordinates'],
                'initial_handoff_minimum_slack':str(slack),
                'executable_smoke':_smoke(system,recurrent,certificate)})
            print(f'verified seed={seed} d={dimension} interval=[81,{initial+216}] initial-grid-minimum={initial}',flush=True)
    summary={'schema':'ncd.optimal-initial-grid-acceptance.v1','status':'accepted',
        'previous_acceptance_sha256':digest(PRIOR),
        'seeds':list(SEEDS),'dimensions':list(DIMENSIONS),'cases':cases,
        'all_phase_crossings_replayed':True,
        'all_original_lower_and_two_stage_certificates_replayed':True,
        'all_initial_optimality_certificates_replayed':True,
        'minimum_state_intervals_by_dimension':{str(d):[81,297 if d==8 else 324] for d in DIMENSIONS},
        'scope':'eight frozen trained phase-crossing rings, full unit initial/action cubes, all finite horizons',
        'initial_optimality_scope':'positive integer coordinate grids satisfying exact output and sensitivity handoff into the fixed recurrent relation',
        'boundary':'optimal initial coordinate grid in this proof class; general finite realization minimum remains unknown'}
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Initial-grid acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',result['minimum_state_intervals_by_dimension'],flush=True)
