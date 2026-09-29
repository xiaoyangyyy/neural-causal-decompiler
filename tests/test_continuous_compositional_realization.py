"""Meaningful checks for the succinct global simulation proof."""
from copy import deepcopy
from pathlib import Path

import pytest

from ncd.continuous_separation import ContinuousReLUSystem
from ncd.continuous_compositional_realization import (
    certify_compositional, machine_initial, machine_output, machine_step,
    verify_compositional, verify_frozen_scale_models,
)
from ncd.io import read_json


SOURCE = Path(__file__).resolve().parents[1] / "runs" / "certified_continuous_scale_seed4701"


def _trained_eight():
    return ContinuousReLUSystem.from_dict(
        read_json(SOURCE / "profiles" / "profile_000" / "system.json"))


def test_full_cube_trained_model_and_executable_machine():
    system = _trained_eight()
    certificate = certify_compositional(system)
    assert certificate["status"] == "certified"
    assert certificate["lower_bound"] == 81
    assert certificate["upper_bound"] == str(14 ** 8)
    state = machine_initial((0.35,) * 8, 14)
    concrete = (0.35,) * 8
    for step in range(25):
        action = (0.1 + step % 5 * 0.17, 0.15 + step % 3 * 0.2)
        actual = system.observation(concrete)
        abstract = machine_output(system, state, 14)
        assert max(abs(float(x) - y) for x, y in zip(abstract, actual)) <= 0.17
        state = machine_step(system, state, action, 14, 64)
        concrete = system.step(concrete, action)
    assert verify_compositional(system, certificate)["status"] == "certified"


def test_tampering_and_failed_closure_are_rejected():
    system = _trained_eight()
    certificate = certify_compositional(system)
    changed = deepcopy(certificate)
    changed["transition_error_upper"][0] = "0"
    with pytest.raises(ValueError, match="does not match"):
        verify_compositional(system, changed)
    changed = deepcopy(certificate)
    changed["upper_bound"] = "1"
    with pytest.raises(ValueError, match="does not match"):
        verify_compositional(system, changed)
    unresolved = certify_compositional(system, radius="0.10", epsilon="0.17")
    assert unresolved["status"] == "unresolved"
    assert unresolved["upper_bound"] is None


def test_all_frozen_profiles_replay_if_artifact_exists():
    output = Path(__file__).resolve().parents[1] / "runs" / "compositional_scale_seed4701"
    if output.exists():
        assert len(verify_frozen_scale_models(SOURCE, output)["records"]) == 4




def test_weighted_one_bin_collapse_and_tampering():
    from ncd.continuous_compositional_realization import (
        certify_weighted, verify_weighted, weighted_machine_initial,
        weighted_machine_output, weighted_machine_step, verify_weighted_scale_models,
    )
    system = _trained_eight()
    bins = (12, 6, 6, 6, 1, 1, 1, 12)
    radii = ("0.17",) * 4 + ("1.01",) * 3 + ("0.17",)
    certificate = certify_weighted(system, bins, radii, action_bins=128)
    assert certificate["status"] == "certified"
    assert certificate["lower_bound"] == 81
    assert certificate["upper_bound"] == "31104"
    concrete = (0.35,) * 8
    abstract = weighted_machine_initial(concrete, bins)
    for step in range(25):
        action = (0.1 + step % 5 * 0.17, 0.15 + step % 3 * 0.2)
        expected = system.observation(concrete)
        actual = weighted_machine_output(system, abstract, bins)
        assert max(abs(float(x) - y) for x, y in zip(actual, expected)) <= 0.17
        abstract = weighted_machine_step(system, abstract, action, bins, 128)
        concrete = system.step(concrete, action)
    changed = deepcopy(certificate)
    changed["coordinate_bins"][4] = 2
    with pytest.raises(ValueError, match="does not match"):
        verify_weighted(system, changed)
    modified_model = deepcopy(system.to_dict())
    modified_model["transition"]["weights"][-1][0][0] += 0.001
    modified_system = ContinuousReLUSystem.from_dict(modified_model)
    with pytest.raises(ValueError, match="does not match"):
        verify_weighted(modified_system, certificate)
    output = Path(__file__).resolve().parents[1] / "runs" / "weighted_compositional_scale_seed4701"
    assert len(verify_weighted_scale_models(SOURCE, output)["records"]) == 4

