from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import pytest
from finite_graph_proof.boundary import certify, verify
from finite_graph_proof.__main__ import prove, verify_bundle


def test_distinct_population_laws_and_finite_sample_overlap():
    c = certify()
    assert c["models"][0]["covariance"] != c["models"][1]["covariance"]
    assert c["models"][0]["graph"] != c["models"][1]["graph"]
    assert c["overlap"]["maximum_residual_quadratic"] == "17/128"
    assert c["overlap"]["minimax_error_lower"] == {"multiplier": "1/2", "base": "1/576", "exponent": 96}
    v = verify(c)
    assert v["all_estimators_covered"] and v["population_family_identifiable"]
    assert v["zero_error_finite_sample_equivalence_class_recovery"] == "refuted"
    assert not v["original_objective_achieved"]


def test_sample96_does_not_prove_or_refute_statistical_gate():
    c = certify()
    v = verify(c)
    assert v["uniform_error_upper"] == "5/53"
    assert v["requested_confidence_gate"] == "unresolved-by-this-bound"
    assert "Statistical recovery at declared nonzero error" in c["not_refuted"]


def test_sufficient_bound_and_multiple_event_correction():
    assert verify(certify(samples=990))["requested_confidence_gate"] == "proved"
    assert verify(certify(samples=989))["requested_confidence_gate"] == "unresolved-by-this-bound"
    six = certify(samples=5990, family_size=6)
    assert verify(six)["uniform_error_upper"] == "1/600"
    assert six["estimator"]["sufficient_samples"] == 5990
    assert verify(certify(samples=990, family_size=6))["requested_confidence_gate"] == "unresolved-by-this-bound"


def test_negative_coefficient_uses_known_sign():
    c = certify(a="-1/2")
    assert c["estimator"]["row_means"] == ["-1/2", "1/2"]
    assert verify(c)["uniform_error_upper"] == "5/53"


def test_vanishing_separation_cannot_keep_same_sample_guarantee():
    c = certify(a="1/100")
    assert c["estimator"]["sufficient_samples"] > 1000000
    assert verify(c)["requested_confidence_gate"] == "unresolved-by-this-bound"


@pytest.mark.parametrize("change", ["covariance", "moment", "graph", "confidence", "population", "complete"])
def test_forged_theorem_rejected(change):
    c = certify()
    if change == "covariance":
        c["models"][0]["covariance"][0][0] = "2"
    elif change == "moment":
        c["estimator"]["row_variance"] = "1"
    elif change == "graph":
        c["models"][0]["graph"][1][1] = 1
    elif change == "confidence":
        c["estimator"]["requested_confidence_gate"] = "proved"
    elif change == "population":
        c["population"] = "All Gaussian directions identifiable"
    else:
        c["original_objective_achieved"] = True
    with pytest.raises(ValueError):
        verify(c)


def test_degenerate_or_non_iid_contract_is_not_silently_claimed():
    with pytest.raises(ValueError):
        certify(a="0")
    with pytest.raises(ValueError):
        certify(samples=True)
    c = certify()
    c["noise_law"] = "Correlated noise and repeated same observation"
    with pytest.raises(ValueError):
        verify(c)


def test_candidate_sets_and_original_worlds_remain_outside_claim():
    c = certify()
    assert "Candidate sets that retain both different equivalence classes" in c["not_refuted"]
    assert "Graph recovery for all original 3/5/8-node worlds" in c["not_proved"]
    assert not c["original_claim_closed"]


def _fixture(tmp_path):
    root = tmp_path / "case"
    (root / "validation").mkdir(parents=True)
    import finite_graph_proof
    package = Path(finite_graph_proof.__file__).parent
    config = {"schema": "ncd.finite-graph-protocol.v1", "output": "runs/theorem", "cases": {"finite96": {"a": "0.5", "samples": 96, "delta": "1/100", "family_size": 1}}, "source_sha256": {"finite_graph_proof/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")}, "original_objective_achieved": False}
    path = root / "validation/protocol.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, root / "runs/theorem"


def test_bundle_replay_keeps_parameters_and_refuses_overwrite(tmp_path):
    config, output = _fixture(tmp_path)
    result = prove(config)
    assert prove(config, resume=True) == result
    assert verify_bundle(output / "manifest.json") == result
    with pytest.raises(FileExistsError):
        prove(config)


def test_rehashed_scope_forgery_still_rejected(tmp_path):
    config, output = _fixture(tmp_path)
    prove(config)
    certificates = json.loads((output / "certificates.json").read_text())
    certificates["finite96"]["original_claim_closed"] = True
    (output / "certificates.json").write_text(json.dumps(certificates), encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["certificates.json"] = hashlib.sha256((output / "certificates.json").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_bundle(output / "manifest.json")


def test_installed_source_binding_is_complete(tmp_path):
    config, output = _fixture(tmp_path)
    data = json.loads(config.read_text())
    del data["source_sha256"]["finite_graph_proof/boundary.py"]
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="source binding"):
        prove(config)
