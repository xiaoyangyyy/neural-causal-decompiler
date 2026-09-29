import copy

import pytest

from ncd.continuous_nonlinear import (
    NonlinearContinuousConfig,
    run_continuous_nonlinear,
    run_nonlinear_profile,
    train_phase_crossing_dynamics,
    verify_continuous_nonlinear,
)
from ncd.continuous_separation import verify_separation_certificate


METHODS = (
    "hybrid_best",
    "hybrid_widest",
    "independent_best",
    "independent_widest",
)


def test_phase_crossing_training_is_exact_and_crosses_every_gate():
    _, training = train_phase_crossing_dynamics(16, 2, 256, 5701)
    assert training["fit_max_error"] < 1e-9
    assert training["phase_crossing_gates"] == 3
    assert all(low < 0 < high
               for low, high in training["certified_gate_preactivation_intervals"])


def test_nonlinear_ablation_closes_near_and_far_certificates():
    config = NonlinearContinuousConfig.quick()
    profile = run_nonlinear_profile(config, 0)
    near = profile["results"]["near"]
    far = profile["results"]["far"]
    assert all(near[method]["status"] == "certified-within-epsilon" for method in METHODS)
    assert all(far[method]["status"] == "separated" for method in METHODS)
    assert near["hybrid_best"]["leaves"] < near["hybrid_widest"]["leaves"]
    assert near["hybrid_widest"]["leaves"] < near["independent_widest"]["leaves"]
    methods = near["leaf_methods"]["hybrid_widest"]
    assert methods["relational-stable"] > 0
    assert methods["independent-ibp"] > 0

    system, _ = train_phase_crossing_dynamics(
        16, 2, config.training_samples, config.seed)
    certificate = near["hybrid_widest"]
    assert verify_separation_certificate(system, certificate)["status"] ==         "certified-within-epsilon"
    tampered = copy.deepcopy(certificate)
    tampered["upper_bound"] = 0.0
    with pytest.raises(ValueError):
        verify_separation_certificate(system, tampered)


def test_continuous_nonlinear_workflow_replays_and_detects_tampering(tmp_path):
    output = tmp_path / "nonlinear"
    summary = run_continuous_nonlinear(output, NonlinearContinuousConfig.quick())
    assert summary["profiles"] == 1
    assert summary["all_training_domains_certify_phase_crossings"]
    replay = verify_continuous_nonlinear(output)
    assert replay["status"] == "verified"
    assert replay["certificates_verified"] == 8

    profile = output / "profiles" / "profile_000" / "profile.json"
    profile.write_text(profile.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_continuous_nonlinear(output)