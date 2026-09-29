"""Exact lower proof for every weighted uniform coordinate grid."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import pytest

from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import sensitivity
from ncd.io import read_json
from ncd.weighted_grid_optimality import (
    _absolute_rows, certify_weighted_grid_optimality, verify_weighted_grid_optimality)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'runs'/'certified_continuous_scale_seed4701'/'profiles'/'profile_003'/'system.json'
UPPER = ROOT/'runs'/'integer_grid_refinement_v1'/'state_action'/'certificate.json'
CERT = ROOT/'runs'/'weighted_grid_optimality_v1'/'certificate.json'


def test_complete_weighted_grid_class_proof_replays():
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    upper = read_json(UPPER)
    certificate = read_json(CERT)
    result = verify_weighted_grid_optimality(system,upper,certificate)
    assert result == {'status':'verified','class_minimum_states':27216,
                      'horizon':10,'exclusion_cases':125}
    assert certificate['unrestricted_pair_minimum_product'] == 63
    assert certificate['least_exclusion_case'] == [15,8]
    assert Q(certificate['least_exclusion_margin']) > 0
    with pytest.raises(ValueError,match='Extra-axis branch'):
        certify_weighted_grid_optimality(system,upper,horizon=8)
    forged = deepcopy(certificate)
    forged['certified_class_minimum_states'] = 27215
    with pytest.raises(ValueError,match='replay mismatch'):
        verify_weighted_grid_optimality(system,upper,forged)

def test_exact_influence_rows_match_original_weighted_verifier():
    system = ContinuousReLUSystem.from_dict(read_json(MODEL))
    transition = _absolute_rows(system.transition)
    for axis in (0, 1, 2, 3, 126, 127):
        basis = [Q(0)] * (system.state_dim + system.action_dim)
        basis[axis] = Q(1)
        original = sensitivity(system.transition,basis)
        assert original == [row.get(axis,Q(0)) for row in transition]
    observation = _absolute_rows(system.observation)
    for axis in (0, 1, 2, 3):
        basis = [Q(0)] * system.state_dim
        basis[axis] = Q(1)
        original = sensitivity(system.observation,basis)
        assert original == [row.get(axis,Q(0)) for row in observation]
