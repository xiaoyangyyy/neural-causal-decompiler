"""Phase-crossing 128D finite-machine proof and tamper rejection."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest

from ncd.affine_observability import Unresolved,_affine_network
from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json
from ncd.reachable_two_stage import (
    two_stage_initial,two_stage_output,two_stage_step,verify_two_stage)
from ncd.trained_nonlinear_realization import phase_witnesses

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs'/'trained_nonlinear_global_v1'/'seed_6101'/'d_128'
PROOF=ROOT/'runs'/'nonlinear_two_stage_v1'/'seed_6101'/'d_128'/'certificate.json'


def test_trained_phase_crossing_128d_two_stage_machine():
    system=ContinuousReLUSystem.from_dict(read_json(SOURCE/'system.json'))
    recurrent=read_json(SOURCE/'certificate.json')
    certificate=read_json(PROOF)
    witness=phase_witnesses(system)
    assert all(abs(Q(x))>Q(1,10000) for x in witness.values())
    with pytest.raises(Unresolved):
        _affine_network(system.transition)
    assert verify_two_stage(system,recurrent,certificate)=={
        'status':'verified','initial_state_count':243,
        'recurrent_state_count':216,'upper_bound':459}
    point=tuple(Q(i%2) for i in range(system.state_dim))
    concrete=list(point)
    abstract=two_stage_initial(system,point)
    actions=(tuple(Q(0) for _ in range(system.action_dim)),
             tuple(Q(1) for _ in range(system.action_dim)),
             tuple(Q(1,2) for _ in range(system.action_dim)))
    for action in actions:
        output=two_stage_output(system,recurrent,abstract)
        actual=value(system.observation,concrete)
        assert max(abs(a-b) for a,b in zip(output,actual))<=Q(recurrent['epsilon'])
        abstract=two_stage_step(system,recurrent,abstract,action)
        concrete=value(system.transition,concrete+list(action))
    forged=deepcopy(certificate)
    forged['recurrent_state_count']-=1
    with pytest.raises(ValueError,match='mismatch'):
        verify_two_stage(system,recurrent,forged)
