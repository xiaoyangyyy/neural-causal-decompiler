"""Phase-aware minimum initialization plus eight portable neural programs."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
from pathlib import Path
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json,save_json,digest
from ncd.nonlinear_action_closure import (
    certify_nonlinear_closure,verify_nonlinear_closure,
    export_nonlinear_program,verify_nonlinear_program,
    program_initial,program_output,program_step)
from ncd.phase_initial_handoff import certify_phase_handoff,verify_phase_handoff
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'runs'/'trained_nonlinear_global_v1'
OLD_INITIAL=ROOT/'runs'/'optimal_initial_grid_v1'
OLD_PROGRAM=ROOT/'runs'/'nonlinear_action_closure_v1'
PRIOR=ROOT/'validation'/'nonlinear_action_closure_acceptance.json'
CERT_ROOT=ROOT/'runs'/'phase_initial_handoff_v1'
STATUS=ROOT/'validation'/'phase_initial_handoff_acceptance.json'
SEEDS=(6101,6102)
DIMENSIONS=(8,32,64,128)


def _smoke(system,recurrent,program):
    points=(tuple(Q(0) for _ in range(system.state_dim)),
            tuple(Q(1) for _ in range(system.state_dim)),
            tuple(Q(i%5,4) for i in range(system.state_dim)))
    actions=((Q(0),Q(1)),(Q(1),Q(0)),(Q(1,128),Q(127,128)),(Q(1,2),Q(1,2)),(Q(1),Q(1)))
    maximum=Q(0)
    for point in points:
        concrete=list(point)
        state=program_initial(program,point)
        for time in range(len(actions)+1):
            actual=value(system.observation,concrete)
            output=program_output(program,state)
            maximum=max(maximum,max(abs(a-b) for a,b in zip(actual,output)))
            if maximum>Q(recurrent['epsilon']):
                raise ValueError('Phase handoff program exceeded tolerance')
            if time<len(actions):
                state=program_step(program,state,actions[time])
                concrete=value(system.transition,concrete+list(actions[time]))
    return {'traces':len(points),'actions_per_trace':len(actions),
            'maximum_observed_error':str(maximum),'backend':'standalone exported program'}


def main(verify=False):
    previous=read_json(PRIOR)
    if (previous['status']!='accepted' or len(previous['cases'])!=8
            or previous['seeds']!=list(SEEDS) or previous['dimensions']!=list(DIMENSIONS)
            or not previous['all_standalone_programs_replayed']
            or not previous['all_complete_nonlinear_action_graphs_replayed']):
        raise ValueError('Previous complete eight-case acceptance missing')
    cases=[]
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            row=next(r for r in previous['cases'] if r['seed']==seed and r['state_dim']==dimension)
            source=STUDY/f'seed_{seed}'/f'd_{dimension}'
            model=source/'system.json'
            recurrent_path=source/'certificate.json'
            old_initial_path=OLD_INITIAL/f'seed_{seed}'/f'd_{dimension}'/'certificate.json'
            old_target=OLD_PROGRAM/f'seed_{seed}'/f'd_{dimension}'
            for path,key in ((model,'model_sha256'),(recurrent_path,'recurrent_certificate_sha256'),
                (old_initial_path,'initial_grid_certificate_sha256'),
                (old_target/'certificate.json','closure_certificate_sha256'),
                (old_target/'program.json','program_sha256')):
                if digest(path)!=row[key]:
                    raise ValueError('Frozen legacy evidence changed')
            system=ContinuousReLUSystem.from_dict(read_json(model))
            recurrent=read_json(recurrent_path)
            old_initial=read_json(old_initial_path)
            old_closure=read_json(old_target/'certificate.json')
            old_program=read_json(old_target/'program.json')
            witness=phase_witnesses(system)
            lower=verify_weighted(system,recurrent)
            old=verify_nonlinear_closure(system,recurrent,old_initial,old_closure)
            portable_old=verify_nonlinear_program(system,recurrent,old_initial,old_closure,old_program)
            if (system.state_dim!=dimension or witness!=row['phase_witnesses']
                    or any(abs(Q(x))<=Q(1,10000) for x in witness.values())
                    or lower['status']!='certified' or lower['lower_bound']!=81
                    or old['upper_bound']!=row['new_minimum_state_interval'][1]
                    or portable_old['status']!='verified'):
                raise ValueError('Original nonlinear/legacy evidence replay failed')
            target=CERT_ROOT/f'seed_{seed}'/f'd_{dimension}'
            initial_path=target/'initial_certificate.json'
            closure_path=target/'closure_certificate.json'
            program_path=target/'program.json'
            if not verify:
                save_json(initial_path,certify_phase_handoff(system,recurrent))
            initial=read_json(initial_path)
            handoff=verify_phase_handoff(system,recurrent,initial)
            if handoff!={'status':'verified','initial_state_count':81,
                'recurrent_state_count':216,'upper_bound':297,'minimum_initial_label_count':81}:
                raise ValueError('Phase-aware minimum initial label result changed')
            if not verify:
                save_json(closure_path,certify_nonlinear_closure(system,recurrent,initial))
            closure=read_json(closure_path)
            new=verify_nonlinear_closure(system,recurrent,initial,closure)
            if not verify:
                save_json(program_path,export_nonlinear_program(system,recurrent,initial,closure))
            program=read_json(program_path)
            portable=verify_nonlinear_program(system,recurrent,initial,closure,program)
            if (new['status']!='verified' or new['initial_state_count']!=81
                    or not 81<=new['upper_bound']<=old['upper_bound']
                    or portable!={'status':'verified','upper_bound':new['upper_bound'],'action_template_count':1}
                    or Q(initial['minimum_handoff_slack'])<0):
                raise ValueError('Invalid phase-aware complete program')
            cases.append({'seed':seed,'state_dim':dimension,
                'model_sha256':digest(model),'recurrent_certificate_sha256':digest(recurrent_path),
                'legacy_initial_certificate_sha256':digest(old_initial_path),
                'legacy_closure_certificate_sha256':digest(old_target/'certificate.json'),
                'legacy_program_sha256':digest(old_target/'program.json'),
                'phase_initial_certificate_sha256':digest(initial_path),
                'closure_certificate_sha256':digest(closure_path),'program_sha256':digest(program_path),
                'phase_witnesses':witness,'old_minimum_state_interval':row['new_minimum_state_interval'],
                'new_minimum_state_interval':[81,new['upper_bound']],
                'initial_states':81,'minimum_initial_label_count':81,
                'recurrent_states':new['recurrent_state_count'],'action_template_count':1,
                'minimum_handoff_slack':initial['minimum_handoff_slack'],
                'state_pair_cells_checked':initial['state_pair_cells_checked'],
                'executable_smoke':_smoke(system,recurrent,program)})
            print(f"verified seed={seed} d={dimension} interval=[81,{new['upper_bound']}] initial-minimum=81 recurrent={new['recurrent_state_count']}",flush=True)
    summary={'schema':'ncd.phase-initial-handoff-acceptance.v1','status':'accepted',
        'previous_acceptance_sha256':digest(PRIOR),'cases':cases,
        'seeds':list(SEEDS),'dimensions':list(DIMENSIONS),
        'all_phase_crossings_replayed':True,'all_original_lower_certificates_replayed':True,
        'all_legacy_graphs_and_programs_replayed':True,
        'all_phase_initial_certificates_replayed':True,
        'all_initial_label_minima_proved':True,
        'all_new_closed_graphs_and_standalone_programs_replayed':True,
        'all_cases_nonincreasing_upper':all(r['new_minimum_state_interval'][1]<=r['old_minimum_state_interval'][1] for r in cases),
        'scope':'eight frozen trained phase-crossing rings; full unit initial/action cubes; all finite horizons',
        'boundary':'initial label count is exactly 81; total finite-machine minimum and original end-to-end causal requirements remain open'}
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Phase handoff acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    result=main(args.verify)
    print('verified',len(result['cases']),'phase-aware programs; legacy replay preserved',flush=True)
