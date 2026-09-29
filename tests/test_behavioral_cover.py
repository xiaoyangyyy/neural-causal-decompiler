"""Matching exact all-horizon behavioral cover and packing."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest

from ncd.behavioral_cover import _covers_unit_interval,representative_states,verify_behavioral_cover
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
PACKING=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'
COVER=ROOT/'runs'/'behavioral_cover_v1'/'affine_d128'/'certificate.json'


def test_exact_cover_matches_all_horizon_packing():
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    packing=read_json(PACKING/'packing_certificate.json')
    capacity=read_json(PACKING/'capacity_certificate.json')
    cover=read_json(COVER)
    centers=representative_states(128)
    assert len(centers)==len(set(centers))==162
    assert _covers_unit_interval((Q(1,6),Q(1,2),Q(5,6)),Q(1,6))
    assert not _covers_unit_interval((Q(1,6),Q(5,6)),Q(1,6))
    assert verify_behavioral_cover(
        system,packing,capacity,cover)=={
            'status':'verified','cover_number':162,
            'representative_count':162,'packing_lower':162}
    epsilon=Q(0.17)
    assert Q(cover['initial_observed_error_upper'])<=epsilon
    assert Q(cover['first_step_output0_error_upper'])<=epsilon
    assert Q(cover['future_output0_error_upper'])<=epsilon
    forged=deepcopy(cover)
    forged['exact_all_horizon_behavioral_cover_number']=161
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_behavioral_cover(system,packing,capacity,forged)
