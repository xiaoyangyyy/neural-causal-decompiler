"""Exact all-real intervention obstruction for one archived symbolic SCM."""
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
OUTPUT = ROOT / "validation/fixed_scm_intervention_obstruction_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rational(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("Expected a finite JSON binary64 number")
    return Fraction(*value.as_integer_ratio())


def polynomial(node):
    """Expand the archived child expression in (x0,x2) over Q."""
    op = node["op"]
    if op == "constant":
        return {(0, 0): rational(node["value"])}
    if op == "var":
        index = node["index"]
        if index not in (0, 2):
            raise ValueError("Unexpected candidate parent")
        return {(int(index == 0), int(index == 2)): Fraction(1)}
    if op not in ("add", "mul") or len(node["args"]) < 2:
        raise ValueError("Unexpected symbolic operator")
    terms = [polynomial(item) for item in node["args"]]
    out = {(0, 0): Fraction(0)} if op == "add" else {(0, 0): Fraction(1)}
    for term in terms:
        if op == "add":
            for key, value in term.items():
                out[key] = out.get(key, Fraction(0)) + value
        else:
            product = {}
            for a, av in out.items():
                for b, bv in term.items():
                    key = (a[0] + b[0], a[1] + b[1])
                    if sum(key) > 2:
                        raise ValueError("Candidate degree exceeds certified degree")
                    product[key] = product.get(key, Fraction(0)) + av * bv
            out = product
    return {key: value for key, value in out.items() if value}


def compute(package=PACKAGE):
    package = Path(package)
    if digest(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original bundle manifest identity changed")
    manifest = read(package / "manifest.json")
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Wrong archive schema")
    for name, expected in manifest["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Archived file changed: " + name)
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Archived source changed: " + name)

    world = read(package / "world.json")
    scm = read(package / "model/structured/explicit_scm.json")
    result = read(package / "result.json")
    graph = [[0, 1, 1], [0, 0, 0], [0, 1, 0]]
    if (world["graph"] != graph or scm["source_graph"] != graph
            or scm["effective_graph"] != graph
            or world["family"] != "linear_gaussian"
            or world["noise_family"] != "gaussian"
            or world["root_shift"] is not False
            or world["scales"] != [1.0, 1.0, 1.0]
            or world["split"] != "test_id"
            or result["world_id"] != world["world_id"]
            or result["status"] != "posthoc-development-only"
            or result["same_world_across_splits"] is not True
            or result["candidate"]["status"] != "candidate-only"
            or result["candidate"]["joint_noise_independence_proved"] is not False
            or result["original_claim_closed"] is not False):
        raise ValueError("Frozen candidate or truth scope changed")
    models = result["candidate"]["node_models"]
    if len(models) != 3 or any(item["family"] != "gaussian" for item in models):
        raise ValueError("Expected three ideal Gaussian candidate laws")
    if not rational(world["noise_scale"]) > 0 or not rational(models[1]["scale"]) > 0:
        raise ValueError("Invalid Gaussian scale")
    true = {(0, 0): Fraction(0)}
    if len(world["equations"][1]) != 2:
        raise ValueError("Expected two true linear parent terms")
    for term in world["equations"][1]:
        if term["operator"] != "linear" or len(term["parents"]) != 1:
            raise ValueError("True child mechanism is not linear")
        parent = term["parents"][0]
        if parent not in (0, 2):
            raise ValueError("Unexpected true parent")
        key = (int(parent == 0), int(parent == 2))
        true[key] = true.get(key, Fraction(0)) + rational(term["coefficient"])
    if set(true) != {(0, 0), (1, 0), (0, 1)}:
        raise ValueError("Unexpected true linear equation")
    candidate = polynomial(scm["equations"][1])
    delta = {key: candidate.get(key, Fraction(0)) - true.get(key, Fraction(0))
             for key in set(candidate) | set(true)}
    delta[(0, 0)] += rational(models[1]["loc"])
    delta = {key: value for key, value in delta.items() if value}
    allowed = {(0, 0), (1, 0), (0, 1), (1, 1)}
    if not set(delta) <= allowed or (1, 1) not in delta:
        raise ValueError("Expected nonzero x0*x2 mismatch")
    a = delta[(1, 1)]
    b = delta.get((1, 0), Fraction(0)) + delta.get((0, 1), Fraction(0))
    c = delta.get((0, 0), Fraction(0))
    witness = a + b + c
    if abs(witness) <= Fraction(1, 100):
        raise ValueError("The fixed do(0=1,2=1) witness is not separated")
    return {
        "schema": "ncd.fixed-scm-intervention-obstruction.v1",
        "status": "verified-fixed-candidate-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "world_sha256": digest(package / "world.json"),
        "candidate_scm_sha256": digest(package / "model/structured/explicit_scm.json"),
        "candidate_result_sha256": digest(package / "result.json"),
        "checker_source_sha256": digest(__file__),
        "world_id": world["world_id"],
        "scope": "archived test_id world; declared ideal Gaussian candidate noise; all-real parent interventions",
        "delta_mean_on_do_x0_x2_equal_t": {"t2": str(a), "t": str(b), "constant": str(c)},
        "witness_intervention": {"0": "1", "2": "1"},
        "witness_child_mean_gap": str(witness),
        "witness_joint_w1_l1_lower": str(abs(witness)),
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


def verify(path=OUTPUT, package=PACKAGE):
    actual = read(path)
    expected = compute(package)
    if actual != expected:
        raise ValueError("Candidate obstruction certificate mismatch")
    return {"status": expected["status"], "certificate_sha256": digest(path),
            "witness_joint_w1_l1_lower": expected["witness_joint_w1_l1_lower"],
            "original_claim_closed": False}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain the previous certificate")
        OUTPUT.write_text(json.dumps(compute(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(verify(), sort_keys=True))
