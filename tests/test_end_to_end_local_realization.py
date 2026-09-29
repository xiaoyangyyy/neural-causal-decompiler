"""Trainable hidden-direction regression and tamper checks."""
from copy import deepcopy
from fractions import Fraction

import pytest

from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.end_to_end_local_realization import (
    LocalLearningConfig, run_case, verify_case,
)
from ncd.io import read_json, save_json


def test_learned_hidden_directions_are_nonlinear_and_certified(tmp_path):
    config = LocalLearningConfig(
        seed=7101, state_dim=8, train_samples=256,
        selection_samples=64, test_samples=128,
        epochs=600, rollout_cases=8, rollout_horizon=4)
    case = tmp_path / "case"
    record = run_case(config, case)
    training = record["training"]
    assert record["certificate_status"] == "certified"
    assert [record["lower_bound"], int(record["upper_bound"])] == [81, 2250]
    assert training["splits_disjoint"]
    assert training["test"]["rmse"] < 0.0001
    assert training["mean_hidden_direction_movement"] > 0.01
    assert training["mean_hidden_bias_movement"] > 0.01
    assert all(abs(Fraction(x)) > Fraction(1, 10000)
               for x in record["phase_witnesses"].values())
    assert verify_case(case) == record


def test_learned_frozen_network_and_training_tampering_rejected(tmp_path):
    config = LocalLearningConfig(
        seed=7102, state_dim=8, train_samples=256,
        selection_samples=64, test_samples=128,
        epochs=120, rollout_cases=8, rollout_horizon=4)
    case = tmp_path / "case"
    run_case(config, case)
    model = read_json(case / "system.json")
    certificate = read_json(case / "certificate.json")
    model["transition"]["weights"][-1][0][0] += 0.001
    with pytest.raises(ValueError, match="does not match"):
        verify_weighted(ContinuousReLUSystem.from_dict(model), certificate)
    training = read_json(case / "training.json")
    training["test"]["rmse"] = 0.0
    save_json(case / "training.json", training)
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_case(case)

