from copy import deepcopy
from pathlib import Path
from fractions import Fraction as Q
import pytest
from finite_graph_chernoff.boundary import certify, verify, power_compare


def test_original_96_rows_now_satisfy_single_event_recovery_gate():
    c = certify()
    r = verify(c)
    assert r["requested_recovery_success_probability_gate"] == "proved"
    assert r["old_second_moment_gate"] == "unresolved-by-this-bound"
    assert r["uniform_error_upper"]["rational_power_base"] == "8/9"
    assert r["uniform_error_upper"]["integer_exponent"] == 48
    assert Q(r["uniform_error_upper"]["short_exact_fraction"]) < Q(1, 100)
    assert r["recovery_estimator_unchanged"] and not r["original_objective_achieved"]


def test_six_events_use_stronger_tail_and_declared_correction():
    c = certify(family_size=6)
    r = verify(c)
    assert c["uniform_error_upper"]["gate"] == "unresolved-by-this-bound"
    assert r["requested_recovery_success_probability_gate"] == "proved"
    assert r["best_supported_bound"] == "mills_cauchy_schwarz"
    assert Q(r["best_short_exact_fraction"]) * 6 < Q(1, 100)
    assert verify(certify(family_size=100))["requested_recovery_success_probability_gate"] == "unresolved-by-this-bound"


def test_odd_rows_drop_only_the_beneficial_half_power():
    c = certify(samples=97)
    assert c["uniform_error_upper"]["odd_half_power_dropped_conservatively"]
    assert c["uniform_error_upper"]["integer_exponent"] == 48
    assert verify(c)["requested_recovery_success_probability_gate"] == "proved"
    assert verify(certify(samples=1))["uniform_error_upper"]["short_exact_fraction"] == "1"


def test_known_negative_coefficient_changes_no_probability_bound():
    a, b = certify(), certify(a="-1/2")
    assert a["uniform_error_upper"] == b["uniform_error_upper"]
    assert verify(b)["requested_recovery_success_probability_gate"] == "proved"


def test_exact_compare_includes_equality_without_float_log():
    assert power_compare(Q(2, 3), 3, Q(8, 27))
    assert not power_compare(Q(2, 3), 3, Q(8, 27) - Q(1, 10**30))


def test_large_arithmetic_budget_retains_unresolved_gate():
    c = certify(a="1/2", samples=1000000)
    r = verify(c)
    assert not c["uniform_error_upper"]["comparison_supported"] and not c["mills_cauchy_schwarz_upper"]["comparison_supported"]
    assert r["requested_recovery_success_probability_gate"] == "unresolved-arithmetic-budget"
    assert c["arithmetic_budget_exhaustion_does_not_refute_recovery"]


@pytest.mark.parametrize("change", ["tilt", "variance", "power", "gate", "noise", "scope"])
def test_mgf_premises_tail_and_scope_cannot_be_forged(change):
    c = certify()
    if change == "tilt":
        c["tilt"] = "1/2"
    elif change == "variance":
        c["gaussian_mgf_denominator"] = "5/4"
    elif change == "power":
        c["uniform_error_upper"]["integer_exponent"] = 96
    elif change == "gate":
        c = certify(family_size=100)
        c["requested_recovery_success_probability_gate"] = "proved"
    elif change == "noise":
        c["base_certificate"]["noise_law"] = "Correlated exogenous noise"
    else:
        c["original_claim_closed"] = True
    with pytest.raises(ValueError):
        verify(c)


def test_inverse_chisquare_premise_and_radical_bound_are_checked():
    c = certify()
    assert Q(c["pi_lower_rational"]) > 3
    c["mills_cauchy_schwarz_upper"]["sqrt_3n_minus6_integer_lower"] = 17
    with pytest.raises(ValueError):
        verify(c)
    assert verify(certify(samples=2))["mills_cauchy_schwarz_upper"]["not_used_reason"]


def test_bundle_dependency_sources_are_required(tmp_path):
    import hashlib, json, finite_graph_proof, finite_graph_chernoff
    from finite_graph_chernoff.__main__ import prove
    root = tmp_path / "case"
    (root / "validation").mkdir(parents=True)
    sources = {}
    for package in (finite_graph_proof, finite_graph_chernoff):
        for p in Path(package.__file__).parent.glob("*.py"):
            sources[package.__name__ + "/" + p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    del sources["finite_graph_proof/boundary.py"]
    config = {"schema": "ncd.finite-chernoff-protocol.v1", "output": "runs/proof", "cases": {"six": {"a": "1/2", "samples": 96, "delta": "1/100", "family_size": 6}}, "source_sha256": sources, "original_objective_achieved": False}
    path = root / "validation/protocol.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="source binding"):
        prove(path)


def test_rehashed_false_recovery_gate_is_rejected(tmp_path):
    import hashlib, json, finite_graph_proof, finite_graph_chernoff
    from finite_graph_chernoff.__main__ import prove, verify_bundle
    root = tmp_path / "case"
    (root / "validation").mkdir(parents=True)
    sources = {}
    for package in (finite_graph_proof, finite_graph_chernoff):
        for p in Path(package.__file__).parent.glob("*.py"):
            sources[package.__name__ + "/" + p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    config = {"schema": "ncd.finite-chernoff-protocol.v1", "output": "runs/proof", "cases": {"hundred": {"a": "1/2", "samples": 96, "delta": "1/100", "family_size": 100}}, "source_sha256": sources, "original_objective_achieved": False}
    path = root / "validation/protocol.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = prove(path)
    assert result["cases"]["hundred"]["requested_recovery_success_probability_gate"] == "unresolved-by-this-bound"
    cert_path = root / "runs/proof/certificates.json"
    certificates = json.loads(cert_path.read_text())
    certificates["hundred"]["requested_recovery_success_probability_gate"] = "proved"
    cert_path.write_text(json.dumps(certificates), encoding="utf-8")
    manifest_path = root / "runs/proof/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["certificates.json"] = hashlib.sha256(cert_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_bundle(manifest_path)
