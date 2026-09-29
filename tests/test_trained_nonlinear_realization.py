"""Held-out nonlinear training and exact global-certificate regression checks."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from ncd.continuous_compositional_realization import verify_weighted
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.trained_nonlinear_realization import (
    NonlinearTrainingConfig, phase_witnesses, run_case, verify_case,
)
from ncd.io import read_json


def test_trained_phase_crossings_global_certificate_and_replay(tmp_path):
    config = NonlinearTrainingConfig(
        seed=6101, state_dim=8, train_samples=256,
        selection_samples=64, test_samples=128,
        rollout_cases=8, rollout_horizon=4)
    case = tmp_path / "case"
    record = run_case(config, case)
    assert record["certificate_status"] == "certified"
    assert [record["lower_bound"], int(record["upper_bound"])] == [81, 2250]
    assert record["training"]["splits_disjoint"]
    assert record["training"]["test"]["rmse"] < 0.001
    assert all(abs(Fraction(x)) > Fraction(1, 10000)
               for x in record["phase_witnesses"].values())
    assert verify_case(case) == record

    model = ContinuousReLUSystem.from_dict(read_json(case / "system.json"))
    state = np.full(8, 0.4)
    demand = model.step(state, (1.0, 0.0))
    signal = model.step(state, (0.0, 1.0))
    assert demand[0] > signal[0] + 0.1
    assert phase_witnesses(model) == record["phase_witnesses"]


def test_nonlinear_frozen_network_tampering_rejected(tmp_path):
    config = NonlinearTrainingConfig(
        seed=6102, state_dim=8, train_samples=256,
        selection_samples=64, test_samples=128,
        rollout_cases=8, rollout_horizon=4)
    case = tmp_path / "case"
    run_case(config, case)
    model_dict = read_json(case / "system.json")
    certificate = read_json(case / "certificate.json")
    model_dict["transition"]["weights"][-1][0][0] += 0.001
    changed = ContinuousReLUSystem.from_dict(model_dict)
    with pytest.raises(ValueError, match="does not match"):
        verify_weighted(changed, certificate)
    stored = read_json(case / "training.json")
    stored["test"]["rmse"] = 0.0
    from ncd.io import save_json
    save_json(case / "training.json", stored)
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_case(case)

