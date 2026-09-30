"""Independent fixed-world intervention witness verifier; does not import generator."""
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
CERTIFICATE = ROOT / "validation/fixed_scm_intervention_obstruction_v1.json"
OUTPUT = ROOT / "validation/fixed_scm_intervention_obstruction_verification_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
GENERATOR_SHA256 = "951be337527ff3f2e0be4a24488b276e9d4bb34112bc4a274ac626874dca1c3a"


def hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def q(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("Nonfinite or non-binary64 number")
    return Fraction(Decimal.from_float(value))


def degree(node):
    op = node["op"]
    if op == "constant":
        return 0
    if op == "var":
        if node["index"] not in (0, 2):
            raise ValueError("Wrong parent")
        return 1
    if op not in ("add", "mul") or len(node["args"]) < 2:
        raise ValueError("Unsupported candidate operator")
    parts = [degree(child) for child in node["args"]]
    result = max(parts) if op == "add" else sum(parts)
    if result > 2:
        raise ValueError("Ray has unsupported degree")
    return result


def value(node, t):
    op = node["op"]
    if op == "constant":
        return q(node["value"])
    if op == "var":
        if node["index"] not in (0, 2):
            raise ValueError("Wrong parent")
        return t
    args = [value(child, t) for child in node["args"]]
    if op == "add":
        return sum(args, Fraction(0))
    if op == "mul":
        result = Fraction(1)
        for arg in args:
            result *= arg
        return result
    raise ValueError("Unsupported candidate operator")


def verify(certificate=CERTIFICATE, package=PACKAGE):
    package = Path(package)
    if hash_file(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Wrong original archive manifest")
    if hash_file(ROOT / "validation/fixed_scm_intervention_obstruction_v1.py") != GENERATOR_SHA256:
        raise ValueError("Generator source changed")
    manifest = load(package / "manifest.json")
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Wrong archive")
    for name, expected in manifest["files"].items():
        if hash_file(package / name) != expected:
            raise ValueError("Archived payload changed")
    for name, expected in manifest["source_files"].items():
        if hash_file(ROOT / name) != expected:
            raise ValueError("Archived code changed")
    world = load(package / "world.json")
    scm = load(package / "model/structured/explicit_scm.json")
    result = load(package / "result.json")
    graph = [[0, 1, 1], [0, 0, 0], [0, 1, 0]]
    if (world["graph"] != graph or scm["source_graph"] != graph
            or scm["effective_graph"] != graph or world["family"] != "linear_gaussian"
            or world["noise_family"] != "gaussian"
            or world["root_shift"] is not False or world["scales"] != [1.0] * 3
            or world["split"] != "test_id" or result["world_id"] != world["world_id"]
            or result["same_world_across_splits"] is not True
            or result["status"] != "posthoc-development-only"
            or result["candidate"]["status"] != "candidate-only"
            or result["candidate"]["joint_noise_independence_proved"] is not False
            or result["original_claim_closed"] is not False):
        raise ValueError("Wrong frozen scope")
    models = result["candidate"]["node_models"]
    if len(models) != 3 or any(model["family"] != "gaussian" or q(model["scale"]) <= 0
                               for model in models):
        raise ValueError("Wrong ideal Gaussian noise model")
    if q(world["noise_scale"]) <= 0:
        raise ValueError("Wrong true noise scale")
    equation = scm["equations"][1]
    if degree(equation) != 2:
        raise ValueError("Candidate ray is not quadratic")
    truth = world["equations"][1]
    if len(truth) != 2 or {tuple(x["parents"]) for x in truth} != {(0,), (2,)}:
        raise ValueError("Wrong true parent equation")
    if any(x["operator"] != "linear" for x in truth):
        raise ValueError("True child not linear")
    true_slope = sum((q(x["coefficient"]) for x in truth), Fraction(0))
    loc = q(models[1]["loc"])
    def gap(t):
        return value(equation, t) + loc - true_slope * t
    left, middle, right = gap(Fraction(-1)), gap(Fraction(0)), gap(Fraction(1))
    a = (right + left - 2 * middle) / 2
    b = (right - left) / 2
    if not a or abs(right) <= Fraction(1, 100):
        raise ValueError("No verified quadratic intervention witness")
    cert = load(certificate)
    exact_fields = {
        "schema": "ncd.fixed-scm-intervention-obstruction.v1",
        "status": "verified-fixed-candidate-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "world_sha256": hash_file(package / "world.json"),
        "candidate_scm_sha256": hash_file(package / "model/structured/explicit_scm.json"),
        "candidate_result_sha256": hash_file(package / "result.json"),
        "checker_source_sha256": GENERATOR_SHA256,
        "world_id": world["world_id"],
        "scope": "archived test_id world; declared ideal Gaussian candidate noise; all-real parent interventions",
        "delta_mean_on_do_x0_x2_equal_t": {"t2": str(a), "t": str(b), "constant": str(middle)},
        "witness_intervention": {"0": "1", "2": "1"},
        "witness_child_mean_gap": str(right),
        "witness_joint_w1_l1_lower": str(abs(right)),
        "witness_joint_w1_l1_exceeds_1_over_100": True,
        "all_real_joint_w1_l1_supremum_infinite": True,
        "all_real_mechanism_absolute_error_supremum_infinite": True,
        "candidate_noise_law_ideal_only": True,
        "true_world_oracle_metadata_used": True,
        "independent_worlds_for_inference": 0,
        "all_candidates_refuted": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    if cert != exact_fields:
        raise ValueError("Independent certificate replay mismatch")
    return {
        "schema": "ncd.fixed-scm-intervention-obstruction-verification.v1",
        "status": "independently-verified-fixed-candidate-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "generator_source_sha256": GENERATOR_SHA256,
        "verifier_source_sha256": hash_file(__file__),
        "certificate_sha256": hash_file(certificate),
        "witness_joint_w1_l1_lower": str(abs(right)),
        "all_real_joint_w1_l1_supremum_infinite": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous verification")
        OUTPUT.write_text(json.dumps(verify(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    expected = verify()
    if load(OUTPUT) != expected:
        raise ValueError("Independent verification receipt mismatch")
    print(json.dumps(expected, sort_keys=True))
