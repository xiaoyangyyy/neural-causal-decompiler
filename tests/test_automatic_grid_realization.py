"""Automatic grid search and exact certificate regression checks."""
from copy import deepcopy
from pathlib import Path

import pytest

from ncd.automatic_grid_realization import propose_grid, run_case, verify_case
from ncd.continuous_generic_realization import benchmark_coupled_system
from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.io import read_json, save_json


def test_coupled_phase_crossing_grid_is_automatically_certified(tmp_path):
    system = benchmark_coupled_system()
    proposal = propose_grid(system, epsilon="0.12")
    assert proposal["status"] == "candidate"
    assert proposal["coordinate_bins"] == [8, 8]
    case = tmp_path / "coupled"
    result = run_case(system, case, epsilon="0.12", packing_axes=2)
    assert result["certificate_status"] == "certified"
    assert result["upper_bound"] == "64"
    assert verify_case(system, case) == result
    changed = read_json(case / "proposal.json")
    changed["coordinate_bins"][0] = 9
    save_json(case / "proposal.json", changed)
    with pytest.raises(ValueError, match="search replay mismatch"):
        verify_case(system, case)


def test_noncontractive_network_fails_closed():
    transition = ReLUMLP(
        weights=(((1.1, 0.0),),),
        biases=((0.0,),))
    observation = ReLUMLP(
        weights=(((1.0,),),),
        biases=((0.0,),))
    system = ContinuousReLUSystem(1, 1, transition, observation, ("control",))
    proposal = propose_grid(system, epsilon="0.17")
    assert proposal["status"] == "unresolved"
    assert "not contractive" in proposal["reason"]


def test_frozen_learned_model_uses_no_handwritten_bin_pattern(tmp_path):
    source = Path(__file__).resolve().parents[1] / (
        "runs/learned_local_global_v1/seed_7201/d_8/system.json")
    system = ContinuousReLUSystem.from_dict(read_json(source))
    case = tmp_path / "learned"
    result = run_case(system, case)
    assert result["certificate_status"] == "certified"
    assert result["lower_bound"] == 81
    assert int(result["upper_bound"]) < 2250
    changed_model = deepcopy(system.to_dict())
    changed_model["transition"]["weights"][-1][0][0] += 0.001
    with pytest.raises(ValueError, match="search replay mismatch"):
        verify_case(ContinuousReLUSystem.from_dict(changed_model), case)

