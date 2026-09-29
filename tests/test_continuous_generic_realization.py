from ncd.continuous_separation import ContinuousReLUSystem
import copy
import numpy as np
import pytest

from ncd.continuous_generic_realization import (
    GridRealizationConfig, benchmark_coupled_system,
    certified_grid_realization, grid_initial, grid_output, grid_step,
    run_generic_grid, verify_generic_grid, verify_grid_realization,
)


def test_coupled_phase_crossing_model_and_full_inductive_certificate():
    system = benchmark_coupled_system()
    left = system.step([0.8, 0.2], [0.3, 0.4])
    swapped = system.step([0.8, 0.4], [0.3, 0.4])
    assert left[0] != swapped[0]
    above = system.step([0.6, 0.4], [0.3, 0.4])
    below = system.step([0.4, 0.6], [0.3, 0.4])
    assert np.isclose(above[0], 0.1 + 0.25 * 0.6 + 0.1 * 0.4 + 0.2 * 0.3 + 0.05 * 0.2)
    assert np.isclose(below[0], 0.1 + 0.25 * 0.4 + 0.1 * 0.6 + 0.2 * 0.3)
    config = GridRealizationConfig()
    certificate = certified_grid_realization(system, config)
    result = verify_grid_realization(system, certificate)
    assert result["status"] == "certified"
    assert (result["lower_bound"], result["upper_bound"]) == (25, 100)
    assert result["transition_boxes_verified"] == 10000
    assert result["worst_transition_bound"] < config.relation_radius


def test_executable_coupled_realization_tracks_continuous_actions():
    system = benchmark_coupled_system()
    certificate = certified_grid_realization(system, GridRealizationConfig())
    rng = np.random.default_rng(531)
    actions = [np.array([x, y]) for x in (0.0, 0.5, 1.0)
               for y in (0.0, 0.5, 1.0)]
    actions += [rng.random(2) for _ in range(500)]
    for start in (np.array([0.0, 1.0]), np.array([1.0, 0.0]), rng.random(2)):
        concrete = start.copy()
        abstract = grid_initial(certificate, tuple(start))
        for action in actions:
            output = np.asarray(grid_output(system, certificate, abstract))
            assert np.max(np.abs(system.observation(concrete) - output)) <= 0.12
            concrete = system.step(concrete, action)
            abstract = grid_step(certificate, abstract, tuple(action))


def test_checker_rejects_locally_tampered_transition_and_unresolved_is_explicit():
    system = benchmark_coupled_system()
    certificate = certified_grid_realization(system, GridRealizationConfig())
    tampered = copy.deepcopy(certificate)
    tampered["transitions"][0]["target"] = 1
    with pytest.raises(ValueError, match="transition"):
        verify_grid_realization(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["relation_output"][0]["output_error_upper"] = 0.0
    with pytest.raises(ValueError, match="Observation"):
        verify_grid_realization(system, tampered)
    unresolved = certified_grid_realization(
        system, GridRealizationConfig(relation_radius=0.05))
    assert unresolved["status"] == "unresolved"
    assert unresolved["upper_bound"] is None
    assert verify_grid_realization(system, unresolved)["status"] == "unresolved"
    with pytest.raises(ValueError, match="No certified"):
        grid_initial(unresolved, (0.1, 0.2))


def test_generic_workflow_replays_and_checks_manifest(tmp_path):
    output = tmp_path / "coupled"
    summary = run_generic_grid(output, benchmark_coupled_system(), GridRealizationConfig())
    assert summary["status"] == "certified"
    assert summary["transition_boxes_verified"] == 10000
    assert verify_generic_grid(output)["status"] == "certified"
    certificate = output / "certificate.json"
    certificate.write_text(certificate.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_generic_grid(output)


def test_generic_certificate_accepts_a_distinct_supplied_network():
    value = benchmark_coupled_system().to_dict()
    value["transition"]["weights"][1][0][1] = 0.08
    system = ContinuousReLUSystem.from_dict(value)
    certificate = certified_grid_realization(system, GridRealizationConfig())
    assert certificate["status"] == "certified"
    assert certificate["system_sha256"] != certified_grid_realization(
        benchmark_coupled_system(), GridRealizationConfig())["system_sha256"]
    assert verify_grid_realization(system, certificate)["upper_bound"] == 100