from copy import deepcopy
from pathlib import Path
import hashlib, json
import pytest
from joint_noise_proof.boundary import certify, verify
from joint_noise_proof.__main__ import prove, verify_bundle


def test_full_support_joint_counterexample_with_zero_marginal_errors():
    c = certify()
    r = verify(c)
    assert c["reference"]["graph"] == c["realization"]["graph"]
    assert c["reference"]["equations"] == c["realization"]["equations"]
    assert c["positive_definite_leading_minors"] == [["1", "1", "1"], ["1", "1", "3/4"]]
    assert r["coordinate_marginal_w1"] == ["0"] * 3
    assert r["joint_w1_l1_strict_lower"] == "7/25"
    assert r["conclusion"] == "refuted" and r["both_joint_gaussians_full_support"]
    assert not r["independent_noise_propagation_refuted"]
    assert not r["actual_learned_residuals_certified"]
    assert not r["original_objective_achieved"]


@pytest.mark.parametrize("change", ["covariance", "singular", "marginal", "lower", "radical", "equation", "joint_law", "scope"])
def test_counterexample_and_conditions_cannot_be_forged(change):
    c = certify()
    if change == "covariance":
        c["realization"]["noise_covariance"][1][2] = "0"
        c["realization"]["noise_covariance"][2][1] = "0"
    elif change == "singular":
        c["realization"]["noise_covariance"][1][2] = "1"
        c["realization"]["noise_covariance"][2][1] = "1"
    elif change == "marginal":
        c["realization"]["noise_covariance"][1][1] = "2"
    elif change == "lower":
        c["rational_bounds"]["joint_w1_l1_lower"] = "1"
    elif change == "radical":
        c["rational_bounds"]["sqrt2_lower"] = "3/2"
    elif change == "equation":
        c["realization"]["equations"][2] = "X2=U2"
    elif change == "joint_law":
        c["realization"]["noise"] = "independent coordinates"
    else:
        c["original_claim_closed"] = True
    with pytest.raises(ValueError):
        verify(c)


def test_portable_bundle_replays_and_retains_previous_artifacts(tmp_path):
    import joint_noise_proof
    package = Path(joint_noise_proof.__file__).parent
    root = tmp_path / "case"
    (root / "validation").mkdir(parents=True)
    c = {"schema": "ncd.joint-noise-protocol.v1", "output": "runs/counterexample", "cases": {"joint": {}}, "source_sha256": {"joint_noise_proof/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")}, "original_objective_achieved": False}
    path = root / "validation/protocol.json"
    path.write_text(json.dumps(c), encoding="utf-8")
    result = prove(path)
    assert verify_bundle(root / "runs/counterexample/manifest.json") == result
    assert prove(path, resume=True) == result
    with pytest.raises(FileExistsError):
        prove(path)
    certificate_path = root / "runs/counterexample/certificates.json"
    certificates = json.loads(certificate_path.read_text())
    certificates["joint"]["original_objective_achieved"] = True
    certificate_path.write_text(json.dumps(certificates), encoding="utf-8")
    manifest_path = root / "runs/counterexample/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["certificates.json"] = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_bundle(manifest_path)
