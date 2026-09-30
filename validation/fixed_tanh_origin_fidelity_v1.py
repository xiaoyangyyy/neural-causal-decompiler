"""One-leaf interval counterexample for an archived frozen neural mechanism."""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.frozen_mechanism_proof import (
    export_mechanism, certify_mechanism, verify_mechanism,
)

PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
OUTPUT = ROOT / "validation/fixed_tanh_origin_fidelity_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
BACKEND_SHA256 = {
    "ncd/frozen_mechanism_proof.py": "af508660639be8a7a3cb112cbc78975c900e7daceba7d9cd6ff31aea5ec7cafc",
    "ncd/proof_intervals.py": "8838c1141070672ca69775520d81f87dc56a93678e3d818c75235ed2c3981f21",
}
DOMAIN = [["0", "0"], ["0", "0"], ["0", "0"]]
MODEL = "model/baseline/mechanism_1.pt"
SCM = "model/structured/explicit_scm.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_archive(package):
    if digest(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original archive manifest changed")
    manifest = read(package / "manifest.json")
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
    world, scm, result = (read(package / x) for x in
        ("world.json", SCM, "result.json"))
    if (world["split"] != "test_id" or result["world_id"] != world["world_id"]
            or result["status"] != "posthoc-development-only"
            or result["same_world_across_splits"] is not True
            or result["original_claim_closed"] is not False
            or scm["source_graph"] != [[0, 1, 1], [0, 0, 0], [0, 1, 0]]):
        raise ValueError("Archived candidate scope changed")
    return world, scm


def compute(package=PACKAGE):
    package = Path(package)
    world, scm = check_archive(package)
    network = export_mechanism(package / MODEL)
    if network["parents"] != [0, 2]:
        raise ValueError("Wrong frozen mechanism parents")
    program = scm["equations"][1]
    proof = certify_mechanism(network, program, DOMAIN, "1/100",
                              max_boxes=1, seconds=30)
    if (proof["status"] != "refuted" or len(proof["tree"]) != 1
            or proof["counterexample"]["point"] != ["0", "0", "0"]
            or Fraction(proof["counterexample"]["error"][0]) <= Fraction(1, 100)):
        raise ValueError("Origin is not a strict one-leaf counterexample")
    if verify_mechanism(network, program, proof)["conclusion"] != "refuted":
        raise ValueError("Formal proof replay failed")
    return {
        "schema": "ncd.fixed-tanh-origin-fidelity.v1",
        "status": "verified-point-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "proof_backend_sha256": BACKEND_SHA256,
        "generator_source_sha256": digest(__file__),
        "world_id": world["world_id"],
        "model_sha256": digest(package / MODEL),
        "scm_sha256": digest(package / SCM),
        "network": network,
        "program": program,
        "proof": proof,
        "point_is_in_original_declared_domain_proved": False,
        "network_semantics": "exact stored float32 parameters, real arithmetic and rigorous rational tanh enclosure",
        "device_rounding_certified": False,
        "teacher_oracle_equations_used": False,
        "candidate_specific_only": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(path=OUTPUT, package=PACKAGE):
    actual = read(path)
    expected = compute(package)
    if actual != expected:
        raise ValueError("Origin fidelity certificate mismatch")
    return {
        "status": actual["status"],
        "certificate_sha256": digest(path),
        "normalized_error_lower": actual["proof"]["counterexample"]["error"][0],
        "original_claim_closed": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous origin certificate")
        OUTPUT.write_text(json.dumps(compute(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(verify(), sort_keys=True))
