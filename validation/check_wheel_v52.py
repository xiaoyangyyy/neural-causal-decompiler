"""Isolated-wheel replay of the 2,303-state abstract-closure finite machine."""
from __future__ import annotations
from fractions import Fraction as Q
import hashlib
from pathlib import Path
import ncd
from ncd.abstract_reachability import graph_initial,graph_output,graph_step,verify_abstract_closure
from ncd.behavioral_cover import verify_behavioral_cover
from ncd.continuous_compositional_realization import value,verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json,save_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'validation'/'wheel_v52_run'
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RECURRENT=ROOT/'runs'/'integer_grid_refinement_v1'/'state_only'/'certificate.json'
PACKING=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'packing_certificate.json'
CAPACITY=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'/'capacity_certificate.json'
COVER=ROOT/'runs'/'behavioral_cover_v1'/'affine_d128'/'certificate.json'
PROOF=ROOT/'runs'/'reachable_two_stage_v1'/'affine_d128'/'certificate.json'
GRAPH=ROOT/'runs'/'abstract_reachability_v1'/'affine_d128'/'certificate.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ncd.__version__!='0.52.0':
        raise ValueError('Installed version mismatch')
    package=Path(ncd.__file__).resolve().parent
    if not package.is_relative_to((ROOT/'validation'/'wheel_v52_env').resolve()):
        raise ValueError('Imported source instead of installed wheel')
    sources=sorted((ROOT/'ncd').glob('*.py'))
    installed=sorted(package.glob('*.py'))
    if ({p.name for p in sources}!={p.name for p in installed}
            or any(sha(p)!=sha(package/p.name) for p in sources)):
        raise ValueError('Installed modules differ from source')
    wheels=list(RUN.glob('neural_causal_decompiler-0.52.0-*.whl'))
    if len(wheels)!=1:
        raise ValueError('Expected exactly one wheel')
    acceptance=read_json(ROOT/'validation'/'abstract_reachability_acceptance.json')
    for path,key in ((MODEL,'model_sha256'),
                     (RECURRENT,'recurrent_certificate_sha256'),
                     (PACKING,'packing_certificate_sha256'),
                     (CAPACITY,'capacity_certificate_sha256'),
                     (COVER,'cover_certificate_sha256'),
                     (PROOF,'two_stage_certificate_sha256'),
                     (GRAPH,'abstract_closure_certificate_sha256')):
        if sha(path)!=acceptance[key]:
            raise ValueError('Frozen two-stage evidence changed')
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    recurrent=read_json(RECURRENT)
    lower=verify_behavioral_cover(
        system,read_json(PACKING),read_json(CAPACITY),read_json(COVER))
    graph=read_json(GRAPH)
    upper=verify_abstract_closure(system,recurrent,read_json(PROOF),graph)
    if (lower['packing_lower']!=162
            or verify_weighted(system,recurrent)['status']!='certified'
            or Q(recurrent['epsilon'])>Q(0.17)
            or upper!={'status':'verified','initial_state_count':243,
                       'initial_successor_count':1952,'recurrent_state_count':2060,
                       'closure_edge_count':51304,'upper_bound':2303}):
        raise ValueError('Installed lower/upper proof mismatch')
    point=tuple(Q(1) for _ in range(system.state_dim))
    concrete=list(point)
    state=graph_initial(system,point)
    actions=(tuple(Q(0) for _ in range(system.action_dim)),
             tuple(Q(1) for _ in range(system.action_dim)))
    for time in range(len(actions)+1):
        output=graph_output(system,recurrent,graph,state)
        actual=value(system.observation,concrete)
        if max(abs(a-b) for a,b in zip(output,actual))>Q(recurrent['epsilon']):
            raise ValueError('Installed machine smoke trace failed')
        if time<len(actions):
            state=graph_step(system,recurrent,graph,state,actions[time])
            concrete=value(system.transition,concrete+list(actions[time]))
    if state[0]!='recurrent':
        raise ValueError('Installed machine did not enter recurrent phase')
    status={
        'state':'verified','version':ncd.__version__,
        'wheel_sha256':sha(wheels[0]),'wheel_name':wheels[0].name,
        'import_file':(package/'__init__.py').relative_to(ROOT).as_posix(),
        'installed_modules_byte_identical':len(sources),
        'initial_states':243,'recurrent_states':2060,
        'conservative_abstract_edges':51304,
        'new_128_minimum_state_interval':[162,2303],
        'installed_machine_smoke':'verified',
        'tests_passed':209,
    }
    save_json(RUN/'status.json',status)
    print('verified',status['new_128_minimum_state_interval'],flush=True)


if __name__=='__main__':
    main()
