import copy
from fractions import Fraction
import pytest

from ncd.continuous_cover import benchmark_cover_system
from ncd.continuous_closed_realization import certified_closed_realization
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.transition_overlap_lower import (
    TransitionLowerConfig, benchmark_transition_lower_system,
    certified_transition_lower, run_transition_lower,
    verify_transition_lower, verify_transition_lower_run,
)


def test_transition_overlap_excludes_five_states_beyond_pairwise_packing():
    system = benchmark_transition_lower_system()
    lower = certified_transition_lower(system, TransitionLowerConfig())
    result = verify_transition_lower(system, lower)
    upper = certified_closed_realization(system)
    assert result["lower_bound"] == 6
    assert result["transition_aware_exclusions"] == 1
    assert upper["lower_bound"] == 5
    assert upper["upper_bound"] == 10
    assert lower["excluded_sizes"][-1]["reason"] == "deterministic-transition-overlap"
    epsilon = Fraction(lower["epsilon_exact_float"])
    lam = Fraction(lower["lambda"])
    beta = Fraction(lower["beta"])
    assert 5 * 2 * epsilon - 1 < lam / 5
    assert beta + lam / 5 > 2 * epsilon


def test_transition_lower_rejects_tampered_inequalities_and_nonidentity_output():
    system = benchmark_cover_system(1)
    certificate = certified_transition_lower(system, TransitionLowerConfig())
    tampered = copy.deepcopy(certificate)
    tampered["excluded_sizes"][-1]["forced_target_overlap"] = "1/20"
    with pytest.raises(ValueError, match="inequality"):
        verify_transition_lower(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["lower_bound"] = 10
    with pytest.raises(ValueError, match="count"):
        verify_transition_lower(system, tampered)
    value = system.to_dict()
    value["observation"]["weights"][1][0][0] = 0.9
    wrong = ContinuousReLUSystem.from_dict(value)
    with pytest.raises(ValueError, match="identity"):
        certified_transition_lower(wrong, TransitionLowerConfig())


def test_transition_lower_accepts_distinct_stable_affine_relu_system():
    value = benchmark_cover_system(1).to_dict()
    value["transition"]["weights"][1][0][1] = 0.25
    value["transition"]["biases"][1][0] = -0.75
    system = ContinuousReLUSystem.from_dict(value)
    certificate = certified_transition_lower(system, TransitionLowerConfig())
    assert certificate["beta"] != certified_transition_lower(
        benchmark_cover_system(1), TransitionLowerConfig())["beta"]
    assert verify_transition_lower(system, certificate)["lower_bound"] >= 5


def test_transition_lower_workflow_replays_and_checks_manifest(tmp_path):
    output = tmp_path / "lower"
    result = run_transition_lower(
        output, benchmark_cover_system(1), TransitionLowerConfig())
    assert result["lower_bound"] == 6
    assert verify_transition_lower_run(output) == result
    certificate = output / "certificate.json"
    certificate.write_text(certificate.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_transition_lower_run(output)

