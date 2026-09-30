"""Independent rational replay for the archived fixed Gaussian noise candidate."""
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
CERTIFICATE = ROOT / "validation/fixed_gaussian_noise_transport_v1.json"
PRIMARY = ROOT / "validation/fixed_gaussian_noise_transport_v1.py"
RECEIPT = ROOT / "validation/fixed_gaussian_noise_transport_verification_v1.json"
ORIGINAL_MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
PRIMARY_SOURCE_SHA256 = "d76655413f483eeedfc51af17412022a516f4a28d14cf8795e83262202adc3da"
CERTIFICATE_FIELDS = {
    "schema", "status", "bundle_manifest_sha256", "world_sha256",
    "candidate_result_sha256", "checker_source_sha256", "world_id",
    "true_noise_scale_binary_rational", "candidate_mean_binary_rational",
    "candidate_scale_binary_rational", "coordinate_wasserstein_l1_upper",
    "joint_wasserstein_l1_upper", "coupling", "ideal_law_only",
    "candidate_joint_product_law_declared",
    "candidate_residual_joint_independence_proved",
    "true_law_oracle_metadata_used", "mechanism_error_bound_proved",
    "intervention_distribution_bound_proved",
    "independent_worlds_for_inference", "original_claim_closed",
    "original_objective_achieved",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_binary(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("Expected stored finite binary64")
    return Fraction(*value.as_integer_ratio())


def check(certificate_path=CERTIFICATE, package=PACKAGE):
    certificate_path = Path(certificate_path)
    package = Path(package)
    certificate = read(certificate_path)
    manifest_path = package / "manifest.json"
    if digest(manifest_path) != ORIGINAL_MANIFEST_SHA256:
        raise ValueError("Original fixed candidate manifest changed")
    manifest = read(manifest_path)
    if (digest(PRIMARY) != PRIMARY_SOURCE_SHA256
            or certificate.get("checker_source_sha256") != PRIMARY_SOURCE_SHA256
            or certificate.get("bundle_manifest_sha256") != ORIGINAL_MANIFEST_SHA256):
        raise ValueError("Fixed certificate source or manifest binding changed")
    if (type(certificate) is not dict or set(certificate) != CERTIFICATE_FIELDS
            or certificate["schema"] != "ncd.fixed-gaussian-noise-transport.v1"
            or certificate["status"] != "verified-fixed-candidate-noise-bound"
            or certificate["coupling"]
            != "shared independent standard Gaussian coordinates"
            or certificate["ideal_law_only"] is not True
            or certificate["candidate_joint_product_law_declared"] is not True
            or certificate["candidate_residual_joint_independence_proved"] is not False
            or certificate["true_law_oracle_metadata_used"] is not True
            or certificate["mechanism_error_bound_proved"] is not False
            or certificate["intervention_distribution_bound_proved"] is not False
            or certificate["independent_worlds_for_inference"] != 0
            or certificate["original_claim_closed"] is not False
            or certificate["original_objective_achieved"] is not False):
        raise ValueError("Fixed noise transport claim was widened")
    for name, expected_hash in manifest["files"].items():
        if digest(package / name) != expected_hash:
            raise ValueError("Original candidate file changed: " + name)
    for name, expected_hash in manifest["source_files"].items():
        if digest(ROOT / name) != expected_hash:
            raise ValueError("Original candidate source changed: " + name)
    world_path = package / "world.json"
    result_path = package / "result.json"
    if (certificate["world_sha256"] != digest(world_path)
            or certificate["candidate_result_sha256"] != digest(result_path)):
        raise ValueError("Original world or candidate hash changed")
    world = read(world_path)
    result = read(result_path)
    if (world["noise_family"] != "gaussian"
            or world["root_shift"] is not False
            or world["scales"] != [1., 1., 1.]
            or result["world_id"] != world["world_id"]
            or certificate["world_id"] != world["world_id"]
            or result["candidate"]["status"] != "candidate-only"
            or result["candidate"]["joint_noise_independence_proved"] is not False
            or result["candidate"]["true_noise_family_proved"] is not False
            or result["same_world_across_splits"] is not True
            or result["independent_worlds_for_inference"] != 0):
        raise ValueError("Original Gaussian candidate scope changed")
    true_scale = exact_binary(world["noise_scale"])
    candidates = result["candidate"]["node_models"]
    if true_scale <= 0 or len(candidates) != 3:
        raise ValueError("Wrong Gaussian dimensions")
    means = []
    scales = []
    bounds = []
    for item in candidates:
        if set(item) != {"family", "loc", "scale"} or item["family"] != "gaussian":
            raise ValueError("Non-Gaussian node candidate")
        mean = exact_binary(item["loc"])
        scale = exact_binary(item["scale"])
        if scale <= 0:
            raise ValueError("Invalid candidate scale")
        means.append(mean)
        scales.append(scale)
        bounds.append(abs(mean) + abs(true_scale - scale))
    if (certificate["true_noise_scale_binary_rational"] != str(true_scale)
            or certificate["candidate_mean_binary_rational"] != list(map(str, means))
            or certificate["candidate_scale_binary_rational"] != list(map(str, scales))
            or certificate["coordinate_wasserstein_l1_upper"] != list(map(str, bounds))
            or certificate["joint_wasserstein_l1_upper"]
            != str(sum(bounds, Fraction(0)))):
        raise ValueError("Gaussian transport arithmetic mismatch")
    return {
        "schema": "ncd.fixed-gaussian-noise-transport-verification.v1",
        "status": "verified-fixed-candidate-noise-bound",
        "original_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "primary_source_sha256": PRIMARY_SOURCE_SHA256,
        "independent_checker_sha256": digest(__file__),
        "certificate_sha256": digest(certificate_path),
        "joint_wasserstein_l1_upper": certificate["joint_wasserstein_l1_upper"],
        "same_world_across_splits": True,
        "intervention_distribution_bound_proved": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def replay_receipt():
    expected = check()
    if read(RECEIPT) != expected:
        raise ValueError("Independent fixed Gaussian noise receipt changed")
    return expected


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if RECEIPT.exists():
            raise FileExistsError("Retain prior fixed Gaussian noise receipt")
        result = check()
        RECEIPT.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    elif len(sys.argv) == 1:
        result = replay_receipt()
    else:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(result, sort_keys=True))
