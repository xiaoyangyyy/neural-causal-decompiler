"""Eight frozen fully trained mixed networks with exact joint phase programs."""
import argparse
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from ncd.io import read_json,save_json,digest
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.trained_mixed_realization import MixedTrainingConfig,run_training,train_mixed,mixed_witness
from ncd.joint_phase_initial_handoff import certify_joint_handoff,verify_joint_handoff
from ncd.nonlinear_action_closure import (certify_nonlinear_closure,verify_nonlinear_closure,
    export_nonlinear_program,verify_nonlinear_program,program_initial,program_output,program_step)
from ncd.transition_consistency_lower import certify_transition_lower,verify_transition_lower

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'validation/mixed_joint_phase_protocol.json'
STATUS=ROOT/'validation/mixed_joint_phase_acceptance.json'
TARGET=ROOT/'runs/trained_mixed_joint_v1'


def replay_training(target):
    record=read_json(target/'training.json')
    system,reproduced,parameters=train_mixed(MixedTrainingConfig(**record['config']))
    if record!=reproduced or system.to_dict()!=read_json(target/'system.json'):
        raise ValueError('Full-parameter training/model replay mismatch')
    with np.load(target/'checkpoint.npz',allow_pickle=False) as stored:
        if set(stored.files)!=set(parameters) or any(not np.array_equal(stored[k],v) for k,v in parameters.items()):
            raise ValueError('Learned local parameter checkpoint replay mismatch')
    return system,record


def smoke(system,program,epsilon):
    maximum=Q(0)
    points=([Q(0)]*system.state_dim,[Q(1)]*system.state_dim,[Q(j%5,4) for j in range(system.state_dim)])
    actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,128),Q(127,128)),(Q(1,2),Q(1,2)),(Q(1),Q(1)))
    for point in points:
        actual=list(point)
        state=program_initial(program,point)
        for time in range(len(actions)+1):
            maximum=max(maximum,max(abs(a-b) for a,b in zip(program_output(program,state),value(system.observation,actual))))
            if maximum>Q(epsilon):
                raise ValueError('Mixed standalone program exceeded tolerance')
            if time<len(actions):
                state=program_step(program,state,actions[time])
                actual=value(system.transition,actual+list(actions[time]))
    return {'traces':3,'actions_per_trace':5,'maximum_observed_error':str(maximum),'backend':'standalone program only'}


def case(protocol,seed,dimension,verify,resume=False):
    target=TARGET/f'seed_{seed}'/f'd_{dimension}'
    config=dict(protocol['training_defaults'],seed=seed,state_dim=dimension)
    training_files=('system.json','training.json','checkpoint.npz','recurrent_certificate.json')
    if not verify and not (resume and all((target/name).exists() for name in training_files)):
        run_training(MixedTrainingConfig(**config),target)
    system,training=replay_training(target)
    if training['config']!=config:
        raise ValueError('Training differs from the frozen protocol')
    if (not training['splits_disjoint'] or len(set(training['split_sha256'].values()))!=3
        or training['test']['rmse']>protocol['test_rmse_upper']
        or training['test']['max_error']>protocol['test_max_error_upper']
        or any(r['maximum_change']<=protocol['minimum_parameter_group_change']
               or r['changed_parameter_count']==0 for r in training['parameter_changes'])):
        raise ValueError('Frozen full-parameter learning gate failed')
    recurrent=read_json(target/'recurrent_certificate.json')
    original=verify_weighted(system,recurrent)
    if original['status']!='certified' or original['lower_bound']!=81:
        raise ValueError('Global recurrent relation unresolved')
    witness=mixed_witness(system,protocol['required_observed_mixed_difference'])
    for point,expected in zip(witness['four_inputs'],witness['four_scalar_outputs']):
        if value(system.transition,list(map(Q,point)))[witness['observed_coordinate']]!=Q(expected):
            raise ValueError('Mixed witness differs from original neural execution')
    if verify or (resume and (target/'mixed_witness.json').exists()):
        if read_json(target/'mixed_witness.json')!=witness:
            raise ValueError('Mixed phase witness replay mismatch')
    else:
        save_json(target/'mixed_witness.json',witness)
    if not verify and not (resume and (target/'initial_certificate.json').exists()):
        save_json(target/'initial_certificate.json',certify_joint_handoff(system,recurrent,max_regions=protocol['max_phase_regions']))
    initial=read_json(target/'initial_certificate.json')
    first=verify_joint_handoff(system,recurrent,initial)
    if first['status']!='verified' or first['minimum_initial_label_count']!=81:
        raise ValueError('Joint state/control handoff unresolved')
    if not verify and not (resume and (target/'closure_certificate.json').exists()):
        save_json(target/'closure_certificate.json',certify_nonlinear_closure(system,recurrent,initial))
    closure=read_json(target/'closure_certificate.json')
    graph=verify_nonlinear_closure(system,recurrent,initial,closure)
    if not verify and not (resume and (target/'program.json').exists()):
        save_json(target/'program.json',export_nonlinear_program(system,recurrent,initial,closure))
    program=read_json(target/'program.json')
    portable=verify_nonlinear_program(system,recurrent,initial,closure,program)
    if (portable['status']!='verified' or portable['upper_bound']!=graph['upper_bound']
            or graph['initial_state_count']!=81):
        raise ValueError('Complete mixed graph/program mismatch')
    if not verify and not (resume and (target/'lower_certificate.json').exists()):
        save_json(target/'lower_certificate.json',certify_transition_lower(system))
    lower=verify_transition_lower(system,read_json(target/'lower_certificate.json'))
    if lower['lower_bound']!=82:
        raise ValueError('Independent whole-machine lower did not close')
    return {'status':'verified','seed':seed,'state_dim':dimension,
        'model_sha256':digest(target/'system.json'),'training_sha256':digest(target/'training.json'),
        'checkpoint_sha256':digest(target/'checkpoint.npz'),
        'recurrent_certificate_sha256':digest(target/'recurrent_certificate.json'),
        'mixed_witness_sha256':digest(target/'mixed_witness.json'),
        'initial_certificate_sha256':digest(target/'initial_certificate.json'),
        'closure_certificate_sha256':digest(target/'closure_certificate.json'),
        'program_sha256':digest(target/'program.json'),
        'lower_certificate_sha256':digest(target/'lower_certificate.json'),
        'minimum_state_interval':[82,graph['upper_bound']],
        'original_weighted_upper':original['upper_bound'],
        'minimum_initial_label_count':81,'initial_states':81,
        'recurrent_states':graph['recurrent_state_count'],
        'action_template_count':portable['action_template_count'],
        'mixed_rectangle_difference':witness['mixed_rectangle_difference'],
        'mixed_observed_coordinate':witness['observed_coordinate'],
        'mixed_hidden_units_checked':initial['mixed_hidden_units_checked'],
        'joint_vertices_checked':initial['joint_vertices_checked'],
        'minimum_handoff_slack':initial['minimum_handoff_slack'],
        'test_metrics':training['test'],'parameter_changes':training['parameter_changes'],
        'training_replayed':True,'standalone_smoke':smoke(system,program,protocol['epsilon'])}


def main(verify=False,resume=False):
    protocol=read_json(PROTOCOL)
    if (protocol['status']!='frozen-before-final-training'
        or digest(ROOT/'ncd/trained_mixed_realization.py')!=protocol['training_module_sha256']):
        raise ValueError('Frozen training protocol changed')
    rows=[]
    for seed in protocol['seeds']:
        for dimension in protocol['dimensions']:
            try:
                row=case(protocol,seed,dimension,verify,resume)
                print(f"verified seed={seed} d={dimension} interval={row['minimum_state_interval']} templates={row['action_template_count']}",flush=True)
            except (ValueError,RuntimeError,FileNotFoundError) as error:
                row={'status':'unresolved','seed':seed,'state_dim':dimension,
                    'error_type':type(error).__name__,'error':str(error)}
                target=TARGET/f'seed_{seed}'/f'd_{dimension}'
                row['available_artifact_sha256']={p.name:digest(p) for p in sorted(target.glob('*')) if p.is_file()}
                print(f'unresolved seed={seed} d={dimension}: {error}',flush=True)
            rows.append(row)
    accepted=all(r['status']=='verified' for r in rows)
    summary={'schema':'ncd.mixed-joint-phase-acceptance.v1','status':'accepted' if accepted else 'partial',
        'protocol_sha256':digest(PROTOCOL),'seeds':protocol['seeds'],'dimensions':protocol['dimensions'],
        'cases':rows,'all_declared_profiles_retained':True,
        'all_full_parameter_training_replayed':accepted,'all_observed_mixed_phase_witnesses_replayed':accepted,
        'all_joint_four_dimensional_handoffs_replayed':accepted,
        'all_complete_graphs_and_standalone_programs_replayed':accepted,
        'all_independent_transition_lowers_replayed':accepted,
        'scope':'eight new fully trainable local shallow mixed networks; full initial/control cubes; every finite horizon; exact dyadic coefficient interpretation',
        'boundary':'two state supports per scalar output; synthetic targets; arbitrary dense/deep networks, global minimum and original R4/R5/R8/R9/R10 remain open'}
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Mixed acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--verify',action='store_true')
    modes.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    result=main(args.verify,args.resume)
    print(result['status'],len(result['cases']),'declared mixed-network profiles retained',flush=True)
