import copy

import numpy as np
import pytest

from ncd.continuous_separation import (
    ContinuousSeparationConfig,
    benchmark_continuous_system,
    certified_incompatibility_lower_bound,
    certify_continuous_horizon_one_realization,
    certified_separation,
    response_distance_bounds,
    run_continuous_separation,
    verify_continuous_separation,
    verify_incompatibility_lower_bound,
    verify_continuous_realization,
    verify_separation_certificate,
)


def test_interval_bound_contains_dense_intervention_search():
    system = benchmark_continuous_system()
    low, high = response_distance_bounds(system, (0.2,), (0.25,), np.array([[0.0]]), np.array([[1.0]]))
    values = []
    for action in np.linspace(0.0, 1.0, 1001):
        word = np.array([[action]])
        distance = float(np.max(np.abs(system.response((0.2,), word) - system.response((0.25,), word))))
        values.append(distance)
    assert low <= max(values) <= high


def test_three_way_status_and_tamper_detection():
    system = benchmark_continuous_system()
    separate = certified_separation(system, (0.1,), (0.9,), (0.0,), (1.0,), 1, 0.3, 32)
    within = certified_separation(system, (0.2,), (0.25,), (0.0,), (1.0,), 1, 0.03, 32)
    unresolved = certified_separation(system, (0.2,), (0.25,), (0.0,), (1.0,), 1, 0.03, 4)
    assert separate["status"] == "separated"
    assert within["status"] == "certified-within-epsilon"
    assert unresolved["status"] == "unresolved"
    assert verify_separation_certificate(system, within)["leaves_verified"] == 16
    tampered = copy.deepcopy(within)
    tampered["tree"]["children"][0]["low"][0][0] = 0.01
    with pytest.raises(ValueError):
        verify_separation_certificate(system, tampered)


def test_continuous_workflow_replays(tmp_path):
    output = tmp_path / "continuous"
    summary = run_continuous_separation(output, ContinuousSeparationConfig.quick())
    assert summary["statuses"] == {
        "separated": 1,
        "certified-within-epsilon": 2,
        "unresolved": 1,
    }
    replay = verify_continuous_separation(output)
    assert replay["status"] == "verified"
    assert replay["certificates_verified"] == 4
    assert replay["incompatibility_lower_bound"]["lower_bound"] == 3
    assert replay["continuous_realization"]["minimal"]
    assert replay["continuous_realization"]["lower_bound"] == 2
    assert replay["continuous_realization"]["upper_bound"] == 2


def test_continuous_incompatibility_graph_certifies_chromatic_lower_bound():
    system = benchmark_continuous_system()
    certificate = certified_incompatibility_lower_bound(
        system, ((0.1,), (0.2,), (0.25,), (0.9,)), (0.0,), (1.0,), 1, 0.03, 32)
    result = verify_incompatibility_lower_bound(system, certificate)
    assert result == {"status": "verified", "states": 4, "pairs": 6, "edges": 5,
                      "lower_bound": 3}


def test_horizon_one_continuous_realization_closes_and_rejects_bad_upper():
    system = benchmark_continuous_system()
    lower = certified_incompatibility_lower_bound(
        system, ((0.1,), (0.2,), (0.25,), (0.9,)), (0.0,), (1.0,), 1, 0.3, 32)
    candidate = {
        "outputs": [[0.23], [0.82]], "encoding": [0, 0, 0, 1],
        "transition_trees": [
            {"kind": "leaf", "low": [0.0], "high": [1.0], "target": 0},
            {"kind": "split", "low": [0.0], "high": [1.0], "dimension": 0,
             "value": 0.1875, "children": [
                 {"kind": "leaf", "low": [0.0], "high": [0.1875], "target": 0},
                 {"kind": "leaf", "low": [0.1875], "high": [1.0], "target": 1}]},
        ],
    }
    certificate = certify_continuous_horizon_one_realization(system, lower, candidate)
    assert verify_continuous_realization(system, certificate)["minimal"]
    tampered = copy.deepcopy(certificate)
    tampered["candidate"]["outputs"][0][0] = 0.0
    with pytest.raises(ValueError):
        verify_continuous_realization(system, tampered)
