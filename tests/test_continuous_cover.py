import copy

import numpy as np
import pytest

from ncd.continuous_regions import certified_region_separation
from ncd.continuous_separation import certified_separation

from ncd.continuous_cover import (
    ContinuousCoverConfig,
    benchmark_cover_system,
    certified_behavioral_cover,
    run_continuous_covers,
    verify_behavioral_cover,
    verify_continuous_covers,
)


def test_cover_benchmark_network_matches_declared_dynamics():
    system = benchmark_cover_system(2)
    state = np.array([0.2, 0.7])
    actions = np.array([[0.4, 0.1], [0.8, 0.3]])
    response = system.response(state, actions)
    expected = [state]
    current = state
    for action in actions:
        current = 0.5 * current + 0.4 * action
        expected.append(current)
    assert np.allclose(response, np.asarray(expected), atol=1e-14, rtol=0.0)


def test_global_unit_cube_cover_closes_in_one_and_two_dimensions():
    for dimension, expected in ((1, 5), (2, 25)):
        system = benchmark_cover_system(dimension)
        certificate = certified_behavioral_cover(system, 5, 3, 0.101)
        result = verify_behavioral_cover(system, certificate)
        assert result["minimal"]
        assert result["lower_bound"] == expected
        assert result["upper_bound"] == expected

    tampered = copy.deepcopy(certificate)
    tampered["cells"][0]["high"][0] = 0.19
    with pytest.raises(ValueError):
        verify_behavioral_cover(system, tampered)


def test_continuous_cover_workflow_replays_and_detects_tampering(tmp_path):
    output = tmp_path / "cover"
    summary = run_continuous_covers(output, ContinuousCoverConfig.quick())
    assert summary["all_minimal"]
    assert summary["largest_cover"] == 5
    assert summary["total_cells_verified"] == 5
    assert summary["total_packing_pairs_verified"] == 10
    replay = verify_continuous_covers(output)
    assert replay["status"] == "verified"
    assert replay["certificates_verified"] == 15

    profile = output / "profiles" / "profile_000" / "profile.json"
    profile.write_text(profile.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_continuous_covers(output)

def test_cover_rejects_valid_subproofs_on_narrower_action_domain():
    system = benchmark_cover_system(1)
    certificate = certified_behavioral_cover(system, 5, 3, 0.101)
    cell = certificate["cells"][0]
    restricted_cell = certified_region_separation(
        system, cell["low"], cell["high"], cell["representative"],
        cell["representative"], (0.0,), (0.5,), 3, 0.101 / 2.0,
        max_leaves=1, bound_method="relational-stable")
    assert restricted_cell["status"] == "certified-within-epsilon"
    tampered_cell = copy.deepcopy(certificate)
    tampered_cell["cells"][0]["certificate"] = restricted_cell
    with pytest.raises(ValueError, match="Cell proof"):
        verify_behavioral_cover(system, tampered_cell)

    pair = certificate["packing_pair_certificates"][0]
    restricted_pair = certified_separation(
        system, certificate["packing_points"][pair["left"]],
        certificate["packing_points"][pair["right"]],
        (0.0,), (0.5,), 3, 0.101, max_leaves=1,
        bound_method="relational-stable")
    assert restricted_pair["status"] == "separated"
    tampered_pair = copy.deepcopy(certificate)
    tampered_pair["packing_pair_certificates"][0]["certificate"] = restricted_pair
    with pytest.raises(ValueError, match="Packing proof"):
        verify_behavioral_cover(system, tampered_pair)