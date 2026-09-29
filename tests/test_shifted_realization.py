import copy
import numpy as np
import pytest

from ncd.continuous_cover import benchmark_cover_system
from ncd.shifted_realization import (
    ShiftedRealizationConfig, certified_shifted_realization,
    run_shifted_realizations, shifted_initial, shifted_output, shifted_step,
    verify_shifted_realization, verify_shifted_realizations,
)


def test_shifted_grid_certifies_nine_and_eighty_one_states():
    config = ShiftedRealizationConfig()
    for dimension, upper, pairs in ((1, 9, 10), (2, 81, 300)):
        system = benchmark_cover_system(dimension)
        certificate = certified_shifted_realization(system, config)
        result = verify_shifted_realization(system, certificate)
        assert result["upper_bound"] == upper
        assert result["horizon"] == "unbounded"
        assert result["action_segments_verified"] == 42
        assert result["packing_pairs_verified"] == pairs
        assert max(s["state_error_upper"]
                   for e in certificate["axis_transitions"]
                   for s in e["segments"]) < config.relation_radius


def test_nine_state_machine_tracks_long_continuous_action_sequences():
    system = benchmark_cover_system(1)
    certificate = certified_shifted_realization(system, ShiftedRealizationConfig())
    rng = np.random.default_rng(907)
    cuts = [s["action_high"] for s in
            certificate["axis_transitions"][4]["segments"][:-1]]
    actions = [0.0, 1.0, *cuts, *rng.random(1000).tolist()]
    for start in (0.0, 0.15, 0.55, 0.95, 1.0):
        concrete = np.array([start])
        abstract = shifted_initial(certificate, (start,))
        for action in actions:
            output = np.asarray(shifted_output(certificate, abstract))
            assert np.max(np.abs(system.observation(concrete) - output)) <= 0.101
            concrete = system.step(concrete, [action])
            abstract = shifted_step(certificate, abstract, (action,))


def test_shifted_checker_rejects_action_partition_and_observation_tampering():
    system = benchmark_cover_system(1)
    certificate = certified_shifted_realization(system, ShiftedRealizationConfig())
    tampered = copy.deepcopy(certificate)
    tampered["axis_transitions"][0]["segments"][0]["action_high"] = 0.0
    with pytest.raises(ValueError, match="partition"):
        verify_shifted_realization(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["axis_observations"][0]["output_error_upper"] = 0.0
    with pytest.raises(ValueError, match="observation"):
        verify_shifted_realization(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["axis_initial"][0]["high"] = 0.12
    with pytest.raises(ValueError, match="initial"):
        verify_shifted_realization(system, tampered)


def test_shifted_workflow_replays_and_detects_manifest_tampering(tmp_path):
    output = tmp_path / "shifted"
    summary = run_shifted_realizations(output, ShiftedRealizationConfig.quick())
    assert summary["largest_realization"] == 9
    replay = verify_shifted_realizations(output)
    assert replay["largest_realization"] == 9
    profile = output / "profiles" / "profile_000" / "profile.json"
    profile.write_text(profile.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_shifted_realizations(output)

