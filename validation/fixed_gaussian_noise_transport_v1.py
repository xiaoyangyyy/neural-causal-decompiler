"""Exact ideal-law noise transport bound for one frozen posthoc SCM candidate."""
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
OUTPUT = ROOT / "validation/fixed_gaussian_noise_transport_v1.json"
EXPECTED_MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
EXPECTED_SOURCE = {
    "validation/run_continuous_noise_diagnostic_v1.py",
    "validation/continuous_noise_candidate_v1.py",
    "ncd/mechanisms.py", "ncd/multiverse.py", "ncd/cdir.py",
    "ncd/worlds.py", "ncd/graphs.py", "ncd/statistics.py",
    "ncd/model.py", "ncd/io.py",
}
EXPECTED_FILES = {
    "world.json", "model/structured/explicit_scm.json",
    "model/baseline/mechanism_0.pt", "model/baseline/mechanism_1.pt",
    "model/baseline/mechanism_2.pt", "result.json",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binary_rational(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(name + " must be a finite stored binary64 number")
    return Fraction(*value.as_integer_ratio())


def compute(package=PACKAGE):
    package = Path(package)
    manifest_path = package / "manifest.json"
    if digest(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Wrong frozen noise candidate manifest")
    manifest = read(manifest_path)
    if (manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1"
            or set(manifest["files"]) != EXPECTED_FILES
            or set(manifest["source_files"]) != EXPECTED_SOURCE):
        raise ValueError("Frozen noise bundle manifest changed")
    for name, expected in manifest["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Frozen noise bundle file changed: " + name)
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen noise source changed: " + name)
    world = read(package / "world.json")
    result = read(package / "result.json")
    if (world["noise_family"] != "gaussian" or world["root_shift"] is not False
            or world["scales"] != [1., 1., 1.]
            or result["world_id"] != world["world_id"]
            or result["status"] != "posthoc-development-only"
            or result["candidate"]["status"] != "candidate-only"
            or result["candidate"]["joint_noise_independence_proved"] is not False
            or result["candidate"]["true_noise_family_proved"] is not False
            or result["original_claim_closed"] is not False):
        raise ValueError("Frozen noise comparison has different scope")
    candidates = result["candidate"]["node_models"]
    if len(candidates) != 3 or any(
        set(item) != {"family", "loc", "scale"}
        or item["family"] != "gaussian" for item in candidates
    ):
        raise ValueError("All three candidate laws must be Gaussian")
    true_scale = binary_rational(world["noise_scale"], "true noise scale")
    if true_scale <= 0:
        raise ValueError("Invalid true Gaussian scale")
    means = []
    scales = []
    coordinate = []
    for item in candidates:
        mean = binary_rational(item["loc"], "candidate Gaussian mean")
        scale = binary_rational(item["scale"], "candidate Gaussian scale")
        if scale <= 0:
            raise ValueError("Invalid candidate Gaussian scale")
        means.append(mean)
        scales.append(scale)
        coordinate.append(abs(mean) + abs(scale - true_scale))
    return {
        "schema": "ncd.fixed-gaussian-noise-transport.v1",
        "status": "verified-fixed-candidate-noise-bound",
        "bundle_manifest_sha256": digest(manifest_path),
        "world_sha256": digest(package / "world.json"),
        "candidate_result_sha256": digest(package / "result.json"),
        "checker_source_sha256": digest(__file__),
        "world_id": world["world_id"],
        "true_noise_scale_binary_rational": str(true_scale),
        "candidate_mean_binary_rational": list(map(str, means)),
        "candidate_scale_binary_rational": list(map(str, scales)),
        "coordinate_wasserstein_l1_upper": list(map(str, coordinate)),
        "joint_wasserstein_l1_upper": str(sum(coordinate, Fraction(0))),
        "coupling": "shared independent standard Gaussian coordinates",
        "ideal_law_only": True,
        "candidate_joint_product_law_declared": True,
        "candidate_residual_joint_independence_proved": False,
        "true_law_oracle_metadata_used": True,
        "mechanism_error_bound_proved": False,
        "intervention_distribution_bound_proved": False,
        "independent_worlds_for_inference": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(certificate_path=OUTPUT, package=PACKAGE):
    actual = read(certificate_path)
    expected = compute(package)
    if actual != expected:
        raise ValueError("Fixed Gaussian transport certificate mismatch")
    return {
        "status": "verified-fixed-candidate-noise-bound",
        "certificate_sha256": digest(certificate_path),
        "joint_wasserstein_l1_upper": actual["joint_wasserstein_l1_upper"],
        "intervention_distribution_bound_proved": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous fixed noise transport certificate")
        certificate = compute()
        OUTPUT.write_text(
            json.dumps(certificate, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
        result = verify()
    elif len(sys.argv) == 1:
        result = verify()
    else:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(result, sort_keys=True))
