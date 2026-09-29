import copy
from fractions import Fraction
import pytest

from ncd.continuous_closed_realization import certified_closed_realization
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.multiswitch_lower import (
    MultiSwitchConfig, benchmark_multiswitch_system,
    certified_multiswitch_lower, run_multiswitch_lower,
    verify_multiswitch_lower, verify_multiswitch_run,
)
from ncd.transition_overlap_lower import (
    TransitionLowerConfig, certified_transition_lower,
)


def test_multiswitch_certificate_excludes_six_states_and_strengthens_old_bound():
    system = benchmark_multiswitch_system()
    certificate = certified_multiswitch_lower(system, MultiSwitchConfig())
    verified = verify_multiswitch_lower(system, certificate)
    old = certified_transition_lower(system, TransitionLowerConfig())
    upper = certified_closed_realization(system)
    assert old["lower_bound"] == 6
    assert verified == {
        "status": "verified", "lower_bound": 7, "excluded_sizes": 6,
        "multi_switch_exclusions": 2, "arithmetic": "exact-rational",
    }
    assert upper["upper_bound"] == 10
    assert certificate["first_unresolved_size"] == 7
    six = certificate["excluded_sizes"][5]
    assert six["required_target_count"] == 4
    assert Fraction(six["forced_total_overlap"]) == Fraction(1, 4)
    assert Fraction(six["overlap_budget"]) < Fraction(1, 4)


def test_multiswitch_checker_rejects_count_overlap_and_scope_tampering():
    system = benchmark_multiswitch_system()
    certificate = certified_multiswitch_lower(system, MultiSwitchConfig())
    for key, value in (
        ("required_target_count", 3),
        ("forced_total_overlap", "1/8"),
        ("reason", "output-cover-length"),
    ):
        tampered = copy.deepcopy(certificate)
        tampered["excluded_sizes"][5][key] = value
        with pytest.raises(ValueError, match="inequality"):
            verify_multiswitch_lower(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["first_unresolved_size"] = 8
    with pytest.raises(ValueError, match="unresolved"):
        verify_multiswitch_lower(system, tampered)
    tampered = copy.deepcopy(certificate)
    tampered["scope"] = "sampled actions"
    with pytest.raises(ValueError, match="scope"):
        verify_multiswitch_lower(system, tampered)


def test_multiswitch_proof_is_bound_to_the_actual_affine_network():
    value = benchmark_multiswitch_system().to_dict()
    value["transition"]["weights"][1][0][1] = 0.25
    value["transition"]["biases"][1][0] = -0.75
    altered = ContinuousReLUSystem.from_dict(value)
    certificate = certified_multiswitch_lower(altered, MultiSwitchConfig())
    assert certificate["beta"] == "1/4"
    assert verify_multiswitch_lower(altered, certificate)["lower_bound"] >= 5
    with pytest.raises(ValueError, match="system"):
        verify_multiswitch_lower(benchmark_multiswitch_system(), certificate)


def test_multiswitch_workflow_replays_and_detects_artifact_tampering(tmp_path):
    output = tmp_path / "lower"
    result = run_multiswitch_lower(
        output, benchmark_multiswitch_system(), MultiSwitchConfig())
    assert result["lower_bound"] == 7
    assert verify_multiswitch_run(output) == result
    certificate = output / "certificate.json"
    certificate.write_text(certificate.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_multiswitch_run(output)

