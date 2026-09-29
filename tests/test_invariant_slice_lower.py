"""Invariant-slice lower bound from independent controlled coordinates."""
from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import pytest

from ncd.continuous_compositional_realization import value
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.invariant_slice_lower import (
    certify_slice_lower, verify_slice_lower, run_case, verify_case)
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "runs" / "certified_shifted_realization_seed14701" / "profiles"


def _systems():
    scalar = ContinuousReLUSystem.from_dict(
        read_json(PROFILES / "profile_000" / "system.json"))
    product = ContinuousReLUSystem.from_dict(
        read_json(PROFILES / "profile_001" / "system.json"))
    return scalar, product


def test_four_exact_invariant_slices_raise_2d_lower_bound(tmp_path):
    scalar, product = _systems()
    result = run_case(scalar, product, tmp_path)
    assert result == {"status": "certified-lower-bound",
                      "lower_bound": 28, "slice_count": 4}
    certificate = read_json(tmp_path / "certificate.json")
    assert certificate["scalar_lower_bound"] == 7
    assert certificate["lower_bound"] == 4 * 7
    epsilon = Q(certificate["epsilon_exact_float"])
    centers = [Q(item["second_state"]) for item in certificate["slices"]]
    for i, item in enumerate(certificate["slices"]):
        action = Q(item["fixed_second_action"])
        assert 0 <= action <= 1
        for first in (Q(0), Q(1, 2), Q(1)):
            assert value(product.transition,
                         [first, centers[i], Q(1, 2), action])[1] == centers[i]
        if i:
            assert centers[i] - centers[i - 1] > 2 * epsilon
    assert verify_case(scalar, product, tmp_path) == result


def test_coupling_or_observation_change_fails_closed():
    scalar, product = _systems()
    changed = deepcopy(product.to_dict())
    changed["transition"]["weights"][-1][0][1] += 0.01
    coupled = ContinuousReLUSystem.from_dict(changed)
    assert certify_slice_lower(scalar, coupled)["status"] == "unresolved"
    changed = deepcopy(product.to_dict())
    changed["observation"]["weights"][-1][0][1] += 0.01
    mixed_observation = ContinuousReLUSystem.from_dict(changed)
    assert certify_slice_lower(scalar, mixed_observation)["status"] == "unresolved"


def test_certificate_tampering_is_rejected(tmp_path):
    scalar, product = _systems()
    run_case(scalar, product, tmp_path)
    certificate = read_json(tmp_path / "certificate.json")
    changed = deepcopy(certificate)
    changed["slices"][2]["fixed_second_action"] = "0"
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_slice_lower(scalar, product, changed)
    changed = deepcopy(certificate)
    changed["lower_bound"] = 35
    save_json(tmp_path / "certificate.json", changed)
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_case(scalar, product, tmp_path)
