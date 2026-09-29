"""Bind exact transition-consistency lowers to eight phase-aware programs."""
from pathlib import Path
import argparse
from ncd.io import read_json,save_json,digest
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.transition_consistency_lower import certify_transition_lower,verify_transition_lower

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'validation/phase_initial_handoff_acceptance.json'
STATUS=ROOT/'validation/transition_consistency_lower_acceptance.json'
STUDY=ROOT/'runs/trained_nonlinear_global_v1'
TARGET=ROOT/'runs/transition_consistency_lower_v1'
PROGRAM=ROOT/'runs/phase_initial_handoff_v1'


def main(verify=False):
    prior=read_json(PRIOR)
    pairs={(s,d) for s in (6101,6102) for d in (8,32,64,128)}
    if (prior['status']!='accepted' or len(prior['cases'])!=8
            or {(r['seed'],r['state_dim']) for r in prior['cases']}!=pairs
            or not prior['all_new_closed_graphs_and_standalone_programs_replayed']):
        raise ValueError('Eight phase-aware program proofs required')
    rows=[]
    for row in prior['cases']:
        seed,d=row['seed'],row['state_dim']
        model=STUDY/f'seed_{seed}'/f'd_{d}'/'system.json'
        program=PROGRAM/f'seed_{seed}'/f'd_{d}'/'program.json'
        if digest(model)!=row['model_sha256'] or digest(program)!=row['program_sha256']:
            raise ValueError('Frozen model or upper program changed')
        system=ContinuousReLUSystem.from_dict(read_json(model))
        certificate=TARGET/f'seed_{seed}'/f'd_{d}'/'certificate.json'
        if not verify:
            save_json(certificate,certify_transition_lower(system))
        proof=read_json(certificate)
        if verify_transition_lower(system,proof)!={'status':'verified','packing_lower_bound':81,'lower_bound':82}:
            raise ValueError('Whole-machine lower replay failed')
        rows.append({'seed':seed,'state_dim':d,'model_sha256':digest(model),
            'lower_certificate_sha256':digest(certificate),'program_sha256':digest(program),
            'previous_minimum_state_interval':row['new_minimum_state_interval'],
            'minimum_state_interval':[82,row['new_minimum_state_interval'][1]],
            'minimum_initial_label_count':81,'witness_axis':proof['witness']['observed_axis'],
            'lower_gap_margin':proof['witness']['lower_gap_margin'],
            'upper_gap_margin':proof['witness']['upper_gap_margin']})
        print(f'verified seed={seed} d={d}: whole-machine lower=82, initial-label minimum=81',flush=True)
    summary={'schema':'ncd.transition-consistency-lower-acceptance.v1','status':'accepted',
        'previous_acceptance_sha256':digest(PRIOR),'cases':rows,
        'all_transition_consistency_lowers_replayed':True,
        'lower_scope':'arbitrary deterministic state-only-output machines, continuous full initial/action cubes, times zero and one',
        'upper_scope':'phase-aware full-cube standalone programs, all finite horizons',
        'boundary':'exact global state/code minimum, general deep-network families and original end-to-end requirements remain open'}
    if verify:
        if read_json(STATUS)!=summary:
            raise ValueError('Transition lower acceptance replay mismatch')
    else:
        save_json(STATUS,summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    main(parser.parse_args().verify)
