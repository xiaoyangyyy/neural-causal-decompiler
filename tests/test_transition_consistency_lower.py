"""Whole-machine successor obstruction and a genuine 81-state countercase."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest
from ncd.io import read_json
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.transition_consistency_lower import certify_transition_lower,verify_transition_lower

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/trained_nonlinear_global_v1/seed_6101/d_128/system.json'
CERT=ROOT/'runs/transition_consistency_lower_v1/seed_6101/d_128/certificate.json'


def test_common_successor_obstruction_beyond_static_packing():
    system=ContinuousReLUSystem.from_dict(read_json(SOURCE))
    proof=read_json(CERT)
    assert verify_transition_lower(system,proof)=={'status':'verified','packing_lower_bound':81,'lower_bound':82}
    e=Q(17,100)
    witness=proof['witness']
    assert witness['observed_axis']==0
    left=list(map(Q,witness['left_initial_observation']))
    right=list(map(Q,witness['right_initial_observation']))
    assert max(abs(a-b) for a,b in zip(left,right))==Q(319,1000)<2*e
    lo,hi=map(Q,witness['required_successor_decoder_interval'])
    assert e<lo<=hi<1-3*e
    # The two checked observation times give no pairwise packing exclusion.
    # Continuum coverage plus their common successor yields the obstruction.
    assert max(abs(Q(a)-Q(b)) for a,b in zip(witness['right_next_observation'],witness['left_next_observation']))<2*e
    assert all(Q(witness[key])>0 for key in ('lower_gap_margin','upper_gap_margin','initial_middle_label_exclusion_margin'))
    forged=deepcopy(proof)
    forged['continuum_coverage']['necessary_decoder_bands'][0][0]='0'
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_transition_lower(system,forged)


def test_do_not_claim_82_for_a_machine_with_81_states():
    payload=read_json(ROOT/'runs/trained_nonlinear_global_v1/seed_6101/d_8/system.json')
    # Constant-zero dynamics admit 81 midpoint-output states: quantize the
    # first four axes at initialization, then always enter the lowest cell.
    # The worst initial and recurrent output error is exactly 1/6 < 17/100.
    payload['transition']['weights']=[[[0.0]*10 for _ in range(8)]]
    payload['transition']['biases']=[[0.0]*8]
    assert Q(1,6)<Q(17,100)
    assert 3**4==81
    system=ContinuousReLUSystem.from_dict(payload)
    with pytest.raises(ValueError,match='No transition-consistency witness'):
        certify_transition_lower(system)
    payload['observation']['biases'][0][0]=0.01
    with pytest.raises(ValueError,match='Four direct observed'):
        certify_transition_lower(ContinuousReLUSystem.from_dict(payload))
