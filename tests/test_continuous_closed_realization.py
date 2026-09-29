import copy
import numpy as np
import pytest

from ncd.continuous_cover import benchmark_cover_system
from ncd.continuous_closed_realization import (
    ClosedRealizationConfig, certified_closed_realization, realization_output,
    realize_initial, realize_step, run_closed_realizations,
    verify_closed_realization, verify_closed_realizations,
)


def test_infinite_horizon_certificate_closes_transition_relation_in_1d_and_2d():
    for dimension, lower, upper, pairs in ((1, 5, 10, 10), (2, 25, 100, 300)):
        system = benchmark_cover_system(dimension)
        certificate = certified_closed_realization(system)
        result = verify_closed_realization(system, certificate)
        assert result["horizon"] == "unbounded"
        assert (result["lower_bound"], result["upper_bound"]) == (lower, upper)
        assert result["packing_pairs_verified"] == pairs
        assert result["transition_segments_verified"] == 50
        assert not result["minimal"]
        assert max(segment["next_error_upper"]
                   for entry in certificate["axis_transitions"]
                   for segment in entry["segments"]) < certificate["epsilon"]


def test_executable_realization_tracks_long_continuous_control_words():
    system = benchmark_cover_system(2)
    certificate = certified_closed_realization(system)
    rng = np.random.default_rng(301)
    starts = [np.array([0.0, 1.0]), np.array([1.0, 0.0]), rng.random(2)]
    cuts = [segment["action_high"] for segment in
            certificate["axis_transitions"][4]["segments"][:-1]]
    actions = [np.array([cut, cut]) for cut in cuts]
    actions += [rng.random(2) for _ in range(1000)]
    for start in starts:
        concrete = start.copy()
        abstract = realize_initial(certificate, tuple(start))
        for action in actions:
            output = np.array(realization_output(certificate, abstract))
            assert np.max(np.abs(system.observation(concrete) - output)) <= 0.101
            concrete = system.step(concrete, action)
            abstract = realize_step(certificate, abstract, tuple(action))


def test_realization_verifier_rejects_partition_and_bound_tampering():
    system = benchmark_cover_system(1)
    certificate = certified_closed_realization(system)
    tampered = copy.deepcopy(certificate)
    tampered["axis_transitions"][0]["segments"][0]["action_high"] = 0.0
    with pytest.raises(ValueError, match="partition"):
        verify_closed_realization(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["axis_transitions"][0]["segments"][0]["next_error_upper"] = 0.0
    with pytest.raises(ValueError, match="partition"):
        verify_closed_realization(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["packing_pairs"].pop()
    with pytest.raises(ValueError, match="Packing"):
        verify_closed_realization(system, tampered)


def test_closed_realization_workflow_replays_and_checks_manifest(tmp_path):
    output = tmp_path / "closed"
    summary = run_closed_realizations(output, ClosedRealizationConfig.quick())
    assert summary["largest_realization"] == 10
    assert summary["total_transition_segments_verified"] == 50
    replay = verify_closed_realizations(output)
    assert replay["status"] == "verified"
    assert replay["transition_segments_verified"] == 50
    profile = output / "profiles" / "profile_000" / "profile.json"
    profile.write_text(profile.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_closed_realizations(output)

