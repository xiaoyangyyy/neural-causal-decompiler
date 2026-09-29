"""Eight-case exact phase-polygon nonlinear finite-realization acceptance."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.initial_grid_synthesis import verify_initial_grid
from ncd.io import read_json,save_json,digest
from ncd.nonlinear_action_closure import (
    certify_nonlinear_closure,verify_nonlinear_closure,
    export_nonlinear_program,verify_nonlinear_program,
    program_initial,program_output,program_step)
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
INITIAL=ROOT/'runs'/'optimal_initial_grid_v1'
PRIOR=ROOT/'validation'/'optimal_initial_grid_acceptance.json'
CERT_ROOT=ROOT/'runs'/'nonlinear_action_closure_v1'
STATUS=ROOT/'validation'/'nonlinear_action_closure_acceptance.json'
SEEDS=(6101,6102)
DIMENSIONS=(8,32,64,128)


def _smoke(system,recurrent,initial,certificate,program):
    points=(tuple(Q(0) for _ in range(system.state_dim)),
            tuple(Q(1) for _ in range(system.state_dim)),
            tuple(Q(i%5,4) for i in range(system.state_dim)))
    actions=tuple((a,b) for a,b in ((Q(0),Q(1)),(Q(1),Q(0)),
                (Q(1,128),Q(127,128)),(Q(1,2),Q(1,2)),(Q(1),Q(1))))
    maximum=Q(0)
    for point in points:
        concrete=list(point)
        state=program_initial(program,point)
        for time in range(len(actions)+1):
            actual=value(system.observation,concrete)
            output=program_output(program,state)
            maximum=max(maximum,max(abs(a-b) for a,b in zip(actual,output)))
            if maximum>Q(recurrent['epsilon']):
                raise ValueError('Nonlinear closure runtime exceeded tolerance')
            if time<len(actions):
                state=program_step(program,state,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
    return {'traces':len(points),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum),'backend':'standalone exported program'}


def main(verify=False):
    previous=read_json(PRIOR)
    if (previous['status']!='accepted' or len(previous['cases'])!=8
            or previous['seeds']!=list(SEEDS) or previous['dimensions']!=list(DIMENSIONS)
            or not previous['all_initial_optimality_certificates_replayed']
            or not previous['all_phase_crossings_replayed']):
        raise ValueError('Previous nonlinear eight-case acceptance missing')
    cases=[]
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            row=next(r for r in previous['cases'] if r['seed']==seed and r['state_dim']==dimension)
            source=STUDY/f'seed_{seed}'/f'd_{dimension}'
            model=source/'system.json'
            recurrent_path=source/'certificate.json'
            initial_path=INITIAL/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
            for path,key in ((model,'model_sha256'),
                             (recurrent_path,'recurrent_certificate_sha256'),
                             (initial_path,'initial_grid_certificate_sha256')):
                if digest(path)!=row[key]:
                    raise ValueError('Frozen prior evidence changed')
            system=ContinuousReLUSystem.from_dict(read_json(model))
            recurrent=read_json(recurrent_path)
            initial=read_json(initial_path)
            witness=phase_witnesses(system)
            lower=verify_weighted(system,recurrent)
            old=verify_initial_grid(system,recurrent,initial)
            if (system.state_dim!=dimension or witness!=row['phase_witnesses']
                    or any(abs(Q(x))<=Q(1,10000) for x in witness.values())
                    or lower['status']!='certified' or lower['lower_bound']!=81
                    or old['upper_bound']!=row['new_minimum_state_interval'][1]):
                raise ValueError('Original nonlinear evidence replay failed')
            target=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
            if not verify:
                save_json(target,certify_nonlinear_closure(system,recurrent,initial))
            certificate=read_json(target)
            result=verify_nonlinear_closure(system,recurrent,initial,certificate)
            if (result['status']!='verified' or result['initial_state_count']!=old['initial_state_count']
                    or not 81<=result['upper_bound']<=old['upper_bound']
                    or certificate['action_phase_boundary_counts']!=[4]
                    or certificate['action_region_counts']!=[5]):
                raise ValueError('Invalid nonlinear closure result')
            program_path=target.with_name('program.json')
            if not verify:
                save_json(program_path,export_nonlinear_program(system,recurrent,initial,certificate))
            program=read_json(program_path)
            portable=verify_nonlinear_program(system,recurrent,initial,certificate,program)
            if portable!={'status':'verified','upper_bound':result['upper_bound'],'action_template_count':1}:
                raise ValueError('Standalone nonlinear program mismatch')
            statistics={key:certificate[key] for key in (
                'distinct_initial_successors','initial_candidate_edges','initial_retained_edges',
                'recurrent_candidate_edges','recurrent_retained_edges','source_centers_checked')}
            cases.append({'seed':seed,'state_dim':dimension,
                'model_sha256':digest(model),'recurrent_certificate_sha256':digest(recurrent_path),
                'initial_grid_certificate_sha256':digest(initial_path),
                'closure_certificate_sha256':digest(target),'program_sha256':digest(program_path),
                'action_template_count':portable['action_template_count'],'phase_witnesses':witness,
                'old_minimum_state_interval':row['new_minimum_state_interval'],
                'new_minimum_state_interval':[81,result['upper_bound']],
                'initial_states':result['initial_state_count'],
                'recurrent_states':result['recurrent_state_count'],
                'graph_statistics':statistics,'executable_smoke':_smoke(system,recurrent,initial,certificate,program)})
            print(f"verified seed={seed} d={dimension} interval=[81,{result['upper_bound']}] recurrent={result['recurrent_state_count']}",flush=True)
    summary={'schema':'ncd.nonlinear-action-closure-acceptance.v1','status':'accepted',
        'previous_acceptance_sha256':digest(PRIOR),'cases':cases,
        'seeds':list(SEEDS),'dimensions':list(DIMENSIONS),
        'all_phase_crossings_replayed':True,'all_original_lower_and_initial_certificates_replayed':True,
        'all_complete_nonlinear_action_graphs_replayed':True,
        'all_standalone_programs_replayed':True,
        'all_cases_strictly_improve_upper':all(r['new_minimum_state_interval'][1]<r['old_minimum_state_interval'][1] for r in cases),
        'scope':'eight frozen trained phase-crossing rings, full unit initial/action cubes, all finite horizons',
        'boundary':'closed conservative phase-polygon action graph; global machine minimum and end-to-end causal decompilation remain open'}
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Nonlinear action-closure acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',len(result['cases']),'cases; strict improvement',result['all_cases_strictly_improve_upper'],flush=True)
