"""Analytic maximum and matching exact one-step witness."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest

from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import read_json
from ncd.one_step_packing_capacity import (
    six_pair_ring_points, verify_one_step_capacity)

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
RUN=ROOT/'runs'/'one_step_capacity_v1'/'affine_d128'


def test_exact_capacity_matches_162_point_certificate():
    system=ContinuousReLUSystem.from_dict(read_json(MODEL))
    packing=read_json(RUN/'packing_certificate.json')
    capacity=read_json(RUN/'capacity_certificate.json')
    assert len(six_pair_ring_points(128))==162
    result=verify_one_step_capacity(system,packing,capacity)
    assert result=={'status':'verified','exact_one_step_packing_capacity':162,
                    'exact_all_horizon_packing_capacity':162,
                    'lower_witness_points':162,'lower_witness_pairs':13041}
    assert Q(capacity['row0_next_output_span_bound'])<=2*Q(0.17)*2
    assert Q(capacity['future_first_output_bound'])<=2*Q(0.17)
    assert Q(capacity['future_hidden_bound'])<=Q(capacity['hidden_gain'])
    assert all(Q(x)<=2*Q(0.17)
               for x in capacity['other_next_output_span_bounds'])
    forged=deepcopy(capacity)
    forged['exact_one_step_packing_capacity']=163
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_one_step_capacity(system,packing,forged)
