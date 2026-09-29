import copy

import numpy as np
import pytest

from ncd.continuous_scale import (
    ContinuousScaleConfig,
    run_continuous_scale,
    run_scale_profile,
    train_scaled_dynamics,
    verify_continuous_scale,
)
from ncd.continuous_separation import (
    certified_separation,
    relational_response_distance_bounds,
    response_distance_bounds,
    verify_separation_certificate,
)


def test_relational_bound_preserves_shared_controls_and_is_tighter():
    system, training = train_scaled_dynamics(8, 2, 128, 4701)
    assert training["fit_max_error"] < 1e-9
    profile = run_scale_profile(ContinuousScaleConfig.quick(), 0)
    near = profile["pairs"]["near"]
    low = np.zeros((3, 2))
    high = np.ones((3, 2))
    _, relational_upper = relational_response_distance_bounds(
        system, near["left"], near["right"], low, high)
    _, independent_upper = response_distance_bounds(
        system, near["left"], near["right"], low, high)
    assert relational_upper < 0.02
    assert independent_upper > relational_upper


def test_scaled_certificate_statuses_and_tamper_detection():
    config = ContinuousScaleConfig.quick()
    profile = run_scale_profile(config, 0)
    assert profile["results"]["near"]["relational"]["status"] == "certified-within-epsilon"
    assert profile["results"]["near"]["independent"]["status"] == "unresolved"
    assert profile["results"]["far"]["relational"]["status"] == "separated"
    system, _ = train_scaled_dynamics(8, 2, config.training_samples, config.seed)
    certificate = profile["results"]["near"]["relational"]
    assert verify_separation_certificate(system, certificate)["status"] == "certified-within-epsilon"
    tampered = copy.deepcopy(certificate)
    tampered["upper_bound"] = 0.0
    with pytest.raises(ValueError):
        verify_separation_certificate(system, tampered)


def test_continuous_scale_workflow_replays(tmp_path):
    output = tmp_path / "scale"
    summary = run_continuous_scale(output, ContinuousScaleConfig.quick())
    assert summary["profiles"] == 2
    assert summary["relational_statuses"] == {
        "separated": 2, "certified-within-epsilon": 2, "unresolved": 0}
    assert summary["independent_statuses"]["unresolved"] == 2
    replay = verify_continuous_scale(output)
    assert replay["status"] == "verified"
    assert replay["certificates_verified"] == 8
