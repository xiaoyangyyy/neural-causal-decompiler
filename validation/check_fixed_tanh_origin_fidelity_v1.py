"""Independent replay of a one-point frozen neural fidelity refutation."""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.frozen_mechanism_proof import export_mechanism, verify_mechanism

PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
CERTIFICATE = ROOT / "validation/fixed_tanh_origin_fidelity_v1.json"
OUTPUT = ROOT / "validation/fixed_tanh_origin_fidelity_verification_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
GENERATOR_SHA256 = "5e258387bbfd7cfeac39c7d02f5320a81a1668b85630eee4df8e3f36379313d5"
BACKEND_SHA256 = {
    "ncd/frozen_mechanism_proof.py": "af508660639be8a7a3cb112cbc78975c900e7daceba7d9cd6ff31aea5ec7cafc",
    "ncd/proof_intervals.py": "8838c1141070672ca69775520d81f87dc56a93678e3d818c75235ed2c3981f21",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(certificate=CERTIFICATE, package=PACKAGE):
    package = Path(package)
    if digest(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original archive manifest changed")
    if digest(ROOT / "validation/fixed_tanh_origin_fidelity_v1.py") != GENERATOR_SHA256:
        raise ValueError("Origin generator source changed")
    manifest = load(package / "manifest.json")
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Wrong archive schema")
    for name, expected in manifest["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Archived payload changed")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Archived source changed")
    for name, expected in BACKEND_SHA256.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Proof backend changed")
    world = load(package / "world.json")
    scm = load(package / "model/structured/explicit_scm.json")
    result = load(package / "result.json")
    if (world["split"] != "test_id" or result["world_id"] != world["world_id"]
            or result["status"] != "posthoc-development-only"
            or result["same_world_across_splits"] is not True
            or result["original_claim_closed"] is not False
            or scm["source_graph"] != [[0, 1, 1], [0, 0, 0], [0, 1, 0]]):
        raise ValueError("Archived candidate scope changed")
    cert = load(certificate)
    expected_fields = {
        "schema", "status", "bundle_manifest_sha256", "proof_backend_sha256",
        "generator_source_sha256", "world_id", "model_sha256", "scm_sha256",
        "network", "program", "proof",
        "point_is_in_original_declared_domain_proved", "network_semantics",
        "device_rounding_certified", "teacher_oracle_equations_used",
        "candidate_specific_only", "original_claim_closed", "original_objective_achieved",
    }
    if set(cert) != expected_fields:
        raise ValueError("Origin certificate fields changed")
    if (cert["schema"] != "ncd.fixed-tanh-origin-fidelity.v1"
            or cert["status"] != "verified-point-counterexample"
            or cert["bundle_manifest_sha256"] != MANIFEST_SHA256
            or cert["proof_backend_sha256"] != BACKEND_SHA256
            or cert["generator_source_sha256"] != GENERATOR_SHA256
            or cert["world_id"] != world["world_id"]
            or cert["model_sha256"] != digest(package / "model/baseline/mechanism_1.pt")
            or cert["scm_sha256"] != digest(package / "model/structured/explicit_scm.json")
            or cert["network"] != export_mechanism(package / "model/baseline/mechanism_1.pt")
            or cert["network"]["parents"] != [0, 2]
            or cert["program"] != scm["equations"][1]
            or cert["point_is_in_original_declared_domain_proved"] is not False
            or cert["network_semantics"] != "exact stored float32 parameters, real arithmetic and rigorous rational tanh enclosure"
            or cert["device_rounding_certified"] is not False
            or cert["teacher_oracle_equations_used"] is not False
            or cert["candidate_specific_only"] is not True
            or cert["original_claim_closed"] is not False
            or cert["original_objective_achieved"] is not False):
        raise ValueError("Origin certificate scope or dependencies changed")
    proof = cert["proof"]
    if (proof["domain"] != [["0", "0"]] * 3 or proof["epsilon"] != "1/100"
            or proof["normalizer"] != cert["network"]["output_scale"]
            or proof["status"] != "refuted"
            or len(proof["tree"]) != 1
            or proof["counterexample"]["point"] != ["0", "0", "0"]
            or Fraction(proof["counterexample"]["error"][0]) <= Fraction(1, 100)):
        raise ValueError("Origin witness contract changed")
    replay = verify_mechanism(cert["network"], cert["program"], proof)
    if replay["conclusion"] != "refuted" or replay["boxes_checked"] != 1:
        raise ValueError("Independent interval replay failed")
    return {
        "schema": "ncd.fixed-tanh-origin-fidelity-verification.v1",
        "status": "independently-verified-point-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "generator_source_sha256": GENERATOR_SHA256,
        "verifier_source_sha256": digest(__file__),
        "certificate_sha256": digest(certificate),
        "normalized_error_interval": proof["counterexample"]["error"],
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous origin verification")
        OUTPUT.write_text(json.dumps(verify(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    receipt = verify()
    if load(OUTPUT) != receipt:
        raise ValueError("Origin receipt mismatch")
    print(json.dumps(receipt, sort_keys=True))
