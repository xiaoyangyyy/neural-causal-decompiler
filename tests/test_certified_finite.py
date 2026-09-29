import copy
import json
import pytest

from ncd.certified_finite import (
    FiniteInterventionalSystem,
    ResponseOracle,
    certificate_from_recovery,
    compare_partition,
    oracle_minimize,
    recover_from_responses,
    verify_certificate,
)
from ncd.certified_experiment import (
    CertifiedFiniteConfig,
    generate_split_system,
    run_certified_finite,
    verify_certified_finite,
)
from ncd.io import digest, save_json
from ncd.finite_neural import FiniteNeuralTransducer
from ncd.approximate_finite import approximate_minimize, verify_approximate_certificate


def tiny_system():
    # s1 and s3 use different coordinates but have the same future behavior.
    return FiniteInterventionalSystem(
        ("s0", "s1", "s2", "s3"),
        ("left", "right"),
        ((1, 2), (1, 2), (2, 1), (3, 2)),
        (0, 1, 1, 1),
        ((0.0, 0.0), (10.0, 0.0), (0.1, 0.0), (-10.0, 0.0)),
    )


def test_exact_quotient_has_independent_upper_and_lower_certificate():
    system = tiny_system()
    result = oracle_minimize(system)
    assert result["blocks"] == [["s0"], ["s1", "s2", "s3"]]
    checked = verify_certificate(system, result["certificate"])
    assert checked["minimal"]
    assert checked["lower_bound"] == checked["upper_bound"] == 2
    assert checked["homomorphism_checks"] == 8


def test_relu_transducer_is_the_enumerated_ground_truth():
    source = tiny_system()
    neural = FiniteNeuralTransducer.compile(source)
    restored = FiniteNeuralTransducer.from_dict(neural.to_dict())
    enumerated = restored.enumerate_system()
    assert enumerated.transitions == source.transitions
    assert enumerated.outputs == source.outputs
    assert enumerated.embeddings[2] == (0.0, 0.0, 1.0, 0.0)


def test_response_only_recovery_matches_oracle_and_budget_keeps_unknowns():
    system = generate_split_system(11, abstract_states=4, duplicates=3, actions=2)
    truth = oracle_minimize(system)
    recovered = recover_from_responses(ResponseOracle(system), strategy="active")
    assert recovered["certified_complete"]
    assert recovered["candidate"] is not None
    assert compare_partition(truth["blocks"], recovered["blocks"])["exact"]

    certificate = certificate_from_recovery(system.identity(), recovered)
    assert verify_certificate(system, certificate)["minimal"]

    limited = recover_from_responses(
        ResponseOracle(system), strategy="active", query_budget=system.state_count
    )
    assert not limited["certified_complete"]
    assert limited["pair_status"]["unresolved"] > 0
    assert limited["pair_status"]["merged_certified"] == 0


def test_certificate_rejects_upper_and_lower_tampering():
    system = tiny_system()
    certificate = oracle_minimize(system)["certificate"]
    bad_upper = copy.deepcopy(certificate)
    bad_upper["candidate"]["transitions"][0][0] = 0
    with pytest.raises(ValueError, match="transition homomorphism"):
        verify_certificate(system, bad_upper)

    bad_lower = copy.deepcopy(certificate)
    bad_lower["lower"]["witnesses"][0]["left_response"] = "forged"
    with pytest.raises(ValueError, match="incorrect response"):
        verify_certificate(system, bad_lower)


def test_finite_approximate_bounds_close_and_upper_is_global():
    system = FiniteInterventionalSystem(
        ("a", "b", "c", "d"),
        ("hold",),
        ((0,), (1,), (2,), (3,)),
        (0.0, 0.2, 3.0, 3.2),
    )
    certificate = approximate_minimize(system, 0.11)
    checked = verify_approximate_certificate(system, certificate)
    assert checked["minimal"]
    assert checked["lower_bound"] == checked["upper_bound"] == 2
    assert checked["maximum_output_error"] <= 0.11

    forged = copy.deepcopy(certificate)
    forged["upper"]["candidate"]["outputs"][0] = 1.0
    with pytest.raises(ValueError, match="exceeds epsilon"):
        verify_approximate_certificate(system, forged)


def test_quick_workflow_replays_and_detects_resigned_metric_tampering(tmp_path):
    output = tmp_path / "certified"
    summary = run_certified_finite(output, CertifiedFiniteConfig.quick(seed=37))
    assert summary["active_exact_cases"] == summary["cases"] == 3
    assert summary["all_active_certificates_verified"]
    assert summary["limited_budget"]["unresolved_pairs"] > 0
    assert verify_certified_finite(output)["status"] == "verified"

    changed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    changed["active_exact_cases"] = 0
    save_json(output / "summary.json", changed)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["summary.json"] = digest(output / "summary.json")
    save_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Summary metrics"):
        verify_certified_finite(output)
