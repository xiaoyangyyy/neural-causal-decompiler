"""Exact all-horizon affine neural quotient and negative controls."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import pytest

from ncd.affine_observability import (
    _affine_network, certify_quotient, verify_quotient, run_case, verify_case)
from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]


def _profile(d):
    index = {8: 0, 32: 1, 64: 2, 128: 3}[d]
    path = ROOT / "runs" / "certified_continuous_scale_seed4701" / "profiles" / f"profile_{index:03d}" / "system.json"
    return ContinuousReLUSystem.from_dict(read_json(path))


@pytest.mark.parametrize("dimension", (8, 32, 64, 128))
def test_frozen_cyclic_affine_network_is_exactly_full_observable(dimension):
    system = _profile(dimension)
    certificate = certify_quotient(system)
    assert certificate["status"] == "certified"
    assert certificate["quotient_dim"] == dimension
    assert certificate["proof"] == "cyclic-triangular-delayed-observation"
    assert len(certificate["ring_edges"]) == dimension
    witness = certificate["delayed_witness"]
    assert witness["source_coordinate"] == 4
    assert witness["horizon"] == dimension - 4
    assert witness["first_distinction_horizon"] == dimension - 4
    assert witness["initially_unobserved"]
    assert Q(witness["exact_response_difference"]) != 0
    assert verify_quotient(system, certificate)["quotient_dim"] == dimension


def test_delayed_observation_of_initially_hidden_coordinate():
    system = _profile(8)
    transition, _, _ = _affine_network(system.transition)
    observation, _, _ = _affine_network(system.observation)
    A = tuple(row[:8] for row in transition)
    delta = [Q(int(i == 4)) for i in range(8)]
    for horizon in range(5):
        response = [sum((a * x for a, x in zip(row, delta)), Q(0))
                    for row in observation]
        assert (any(response) if horizon == 4 else not any(response))
        delta = [sum((a * x for a, x in zip(row, delta)), Q(0))
                 for row in A]


def test_one_effective_ring_edge_ablation_exposes_four_dimensional_quotient():
    system = _profile(128)
    modified = deepcopy(system.to_dict())
    hidden_index = 127
    weight = modified["transition"]["weights"][1][0][hidden_index]
    assert weight != 0
    modified["transition"]["weights"][0].append([0.0] * 133)
    modified["transition"]["biases"][0].append(1.0)
    for i, row in enumerate(modified["transition"]["weights"][1]):
        row.append(weight if i == 0 else 0.0)
    modified["transition"]["weights"][1][0][hidden_index] = 0.0
    ablated = ContinuousReLUSystem.from_dict(modified)
    before, before_bias, _ = _affine_network(system.transition)
    after, after_bias, _ = _affine_network(ablated.transition)
    assert before_bias == after_bias
    assert [(i, j) for i in range(128) for j in range(133)
            if before[i][j] != after[i][j]] == [(0, 127)]
    certificate = certify_quotient(ablated)
    assert certificate["status"] == "certified"
    assert certificate["quotient_dim"] == 4
    assert certificate["proof"] == "closed-coordinate-projection"
    assert certificate["coordinates"] == [0, 1, 2, 3]
    assert verify_quotient(ablated, certificate)["quotient_dim"] == 4


def test_noncoordinate_sum_quotient_and_zero_dimensional_constant():
    transition = ReLUMLP(((
        (.2, .2, .1), (.1, .1, .2)),), ((.1, .2),))
    observation = ReLUMLP((((.5, .5),),), ((0.,),))
    system = ContinuousReLUSystem(2, 1, transition, observation)
    certificate = certify_quotient(system)
    assert certificate["status"] == "certified"
    assert certificate["proof"] == "exact-observability-row-space"
    assert certificate["quotient_dim"] == 1
    assert certificate["quotient"]["basis"] == [["1", "1"]]
    assert certificate["quotient"]["pivots"] == [0]
    assert verify_quotient(system, certificate)["quotient_dim"] == 1
    constant = ContinuousReLUSystem(
        2, 1, transition, ReLUMLP((((0., 0.),),), ((.7,),)))
    zero = certify_quotient(constant)
    assert zero["proof"] == "constant-observation"
    assert zero["quotient_dim"] == 0


def test_phase_changing_network_stays_unresolved():
    path = ROOT / "runs" / "functional_support_global_v1" / "controls" / "narrow_tent" / "system.json"
    system = ContinuousReLUSystem.from_dict(read_json(path))
    certificate = certify_quotient(system)
    assert certificate["status"] == "unresolved"
    assert "phase changes" in certificate["reason"]


def test_certificate_tampering_is_rejected(tmp_path):
    system = _profile(8)
    result = run_case(system, tmp_path)
    assert result["quotient_dim"] == 8
    certificate = read_json(tmp_path / "certificate.json")
    changed = deepcopy(certificate)
    changed["quotient_dim"] = 4
    save_json(tmp_path / "certificate.json", changed)
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_case(system, tmp_path)
