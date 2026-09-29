import copy

import numpy as np
import pytest

from ncd.continuous_regions import (
    ContinuousRegionConfig,
    certified_region_separation,
    region_response_distance_bounds,
    run_continuous_regions,
    run_region_profile,
    verify_continuous_regions,
    verify_region_separation_certificate,
)
from ncd.continuous_scale import train_scaled_dynamics
from ncd.continuous_separation import benchmark_continuous_system


def test_region_interval_contains_dense_state_and_action_grid():
    system = benchmark_continuous_system()
    lower, upper = region_response_distance_bounds(
        system, (0.20,), (0.21,), (0.24,), (0.25,),
        np.array([[0.0]]), np.array([[1.0]]))
    distances = []
    for left in np.linspace(0.20, 0.21, 5):
        for right in np.linspace(0.24, 0.25, 5):
            for action in np.linspace(0.0, 1.0, 101):
                word = np.array([[action]])
                distances.append(float(np.max(np.abs(
                    system.response((left,), word)
                    - system.response((right,), word)))))
    assert lower <= min(distances)
    assert max(distances) <= upper


def test_region_three_way_status_and_tamper_detection():
    system = benchmark_continuous_system()
    within = certified_region_separation(
        system, (0.20,), (0.205,), (0.245,), (0.25,),
        (0.0,), (1.0,), 1, 0.03, max_leaves=32,
        split_strategy="best-bound")
    separated = certified_region_separation(
        system, (0.09,), (0.11,), (0.89,), (0.91,),
        (0.0,), (1.0,), 1, 0.3, max_leaves=32,
        split_strategy="best-bound")
    unresolved = certified_region_separation(
        system, (0.20,), (0.24,), (0.21,), (0.25,),
        (0.0,), (1.0,), 1, 0.01, max_leaves=32,
        split_strategy="best-bound")
    assert within["status"] == "certified-within-epsilon"
    assert separated["status"] == "robustly-separated"
    assert unresolved["status"] == "unresolved"
    assert verify_region_separation_certificate(system, within)["status"] ==         "certified-within-epsilon"
    tampered = copy.deepcopy(within)
    tampered["upper_bound"] = 0.0
    with pytest.raises(ValueError):
        verify_region_separation_certificate(system, tampered)


def test_scaled_region_relational_bound_closes_when_independent_does_not():
    profile = run_region_profile(ContinuousRegionConfig.quick(), 0)
    near = profile["results"]["near"]
    far = profile["results"]["far"]
    assert near["relational"]["status"] == "certified-within-epsilon"
    assert near["relational"]["leaves"] == 1
    assert near["independent"]["status"] == "unresolved"
    assert far["relational"]["status"] == "robustly-separated"
    assert far["relational"]["witness_bound_method"] == "relational-stable"
    system, _ = train_scaled_dynamics(8, 2, 256, 8701)
    assert verify_region_separation_certificate(
        system, far["relational"])["status"] == "robustly-separated"


def test_continuous_region_workflow_replays_and_detects_tampering(tmp_path):
    output = tmp_path / "regions"
    summary = run_continuous_regions(output, ContinuousRegionConfig.quick())
    assert summary["statuses"]["relational"] == {
        "robustly-separated": 1,
        "certified-within-epsilon": 1,
        "unresolved": 0,
    }
    assert summary["statuses"]["independent"]["unresolved"] == 1
    replay = verify_continuous_regions(output)
    assert replay["status"] == "verified"
    assert replay["certificates_verified"] == 4

    profile = output / "profiles" / "profile_000" / "profile.json"
    profile.write_text(profile.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_continuous_regions(output)