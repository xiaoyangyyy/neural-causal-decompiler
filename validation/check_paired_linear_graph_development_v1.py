"""Independent algebraic replay of exact paired-do graph identification."""
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
CERTIFICATE = ROOT / "validation/paired_linear_graph_development_v1.json"
OUTPUT = ROOT / "validation/paired_linear_graph_development_verification_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
ESTIMATOR_SHA256 = "ae9efba51dcbb5568a2e296e396dc7cdb8020f8a491e6c6da5710401696c36f5"
GENERATOR_SHA256 = "77de3b178301d58faa2d3e1cb6b8938d4065e6660515eb56ad7a96c939bc9f31"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def multiply(a, b):
    n = len(a)
    return [[sum((a[i][k] * b[k][j] for k in range(n)), Q(0))
             for j in range(n)] for i in range(n)]


def verify(certificate=CERTIFICATE, package=PACKAGE):
    package = Path(package)
    if digest(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original archive manifest changed")
    if (digest(ROOT / "ncd/paired_linear_graph.py") != ESTIMATOR_SHA256
            or digest(ROOT / "validation/paired_linear_graph_development_v1.py") != GENERATOR_SHA256):
        raise ValueError("Estimator or generator source changed")
    manifest = read(package / "manifest.json")
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Wrong archive schema")
    for name, expected in manifest["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Archived payload changed")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Archived source changed")
    world = read(package / "world.json")
    if (world["family"] != "linear_gaussian" or world["noise_family"] != "gaussian"
            or world["root_shift"] is not False or world["scales"] != [1.0] * 3
            or world["split"] != "test_id"
            or world["graph"] != [[0, 1, 1], [0, 0, 0], [0, 1, 0]]):
        raise ValueError("Wrong frozen linear world")
    cert = read(certificate)
    required = {
        "schema", "status", "bundle_manifest_sha256", "world_sha256",
        "estimator_source_sha256", "generator_source_sha256", "world_id",
        "exogenous_witness", "intervention_levels", "plus_responses",
        "minus_responses", "estimate", "oracle_graph_used_only_for_evaluation",
        "paired_exogenous_assumption", "independent_intervention_samples_certified",
        "floating_inference_rounding_certified", "one_archived_development_world_only",
        "original_claim_closed", "original_objective_achieved",
    }
    if set(cert) != required:
        raise ValueError("Paired graph certificate fields changed")
    if (cert["schema"] != "ncd.paired-linear-graph-development.v1"
            or cert["status"] != "verified-scoped-exact-recovery"
            or cert["bundle_manifest_sha256"] != MANIFEST_SHA256
            or cert["world_sha256"] != digest(package / "world.json")
            or cert["estimator_source_sha256"] != ESTIMATOR_SHA256
            or cert["generator_source_sha256"] != GENERATOR_SHA256
            or cert["world_id"] != world["world_id"]
            or cert["exogenous_witness"] != ["1/7", "-1/11", "2/13"]
            or cert["intervention_levels"] != [["1", "-1"]] * 3
            or cert["oracle_graph_used_only_for_evaluation"] is not True
            or cert["paired_exogenous_assumption"] is not True
            or cert["independent_intervention_samples_certified"] is not False
            or cert["floating_inference_rounding_certified"] is not False
            or cert["one_archived_development_world_only"] is not True
            or cert["original_claim_closed"] is not False
            or cert["original_objective_achieved"] is not False):
        raise ValueError("Paired graph scope changed")
    if (len(cert["plus_responses"]) != 3 or len(cert["minus_responses"]) != 3
            or any(len(row) != 3 for row in cert["plus_responses"] + cert["minus_responses"])):
        raise ValueError("Incomplete intervention response grid")
    noise = [Q(value) for value in cert["exogenous_witness"]]
    direct = [[Q(0) for _ in range(3)] for _ in range(3)]
    for target, terms in enumerate(world["equations"]):
        for term in terms:
            if term["operator"] != "linear" or len(term["parents"]) != 1:
                raise ValueError("Nonlinear true mechanism")
            source = term["parents"][0]
            direct[target][source] += Q(float(term["coefficient"]))
    for source in range(3):
        for rows, level in ((cert["plus_responses"], Q(1)),
                            (cert["minus_responses"], Q(-1))):
            response = [Q(value) for value in rows[source]]
            if response[source] != level:
                raise ValueError("Intervened coordinate wrong")
            for target in range(3):
                if target == source:
                    continue
                expected = noise[target] + sum(
                    (direct[target][parent] * response[parent] for parent in range(3)),
                    Q(0),
                )
                if response[target] != expected:
                    raise ValueError("Intervention response violates true equation or shared noise")
    total = [[(Q(cert["plus_responses"][source][target])
               - Q(cert["minus_responses"][source][target])) / 2
              for source in range(3)] for target in range(3)]
    identity = [[Q(int(i == j)) for j in range(3)] for i in range(3)]
    i_minus_direct = [[identity[i][j] - direct[i][j] for j in range(3)]
                      for i in range(3)]
    if multiply(total, i_minus_direct) != identity:
        raise ValueError("Total-effect identity does not follow from responses")
    estimate = cert["estimate"]
    if set(estimate) != {
        "total_effect", "inverse_total_effect", "direct_effect", "graph",
        "paired_exogenous_required", "truth_graph_read_by_estimator",
    } or estimate["paired_exogenous_required"] is not True or estimate["truth_graph_read_by_estimator"] is not False:
        raise ValueError("Estimator claim changed")
    if (estimate["total_effect"] != [[str(v) for v in row] for row in total]
            or estimate["inverse_total_effect"] != [[str(v) for v in row] for row in i_minus_direct]
            or estimate["direct_effect"] != [[str(v) for v in row] for row in direct]
            or estimate["graph"] != world["graph"]):
        raise ValueError("Exact graph reconstruction mismatch")
    return {
        "schema": "ncd.paired-linear-graph-development-verification.v1",
        "status": "independently-verified-scoped-exact-recovery",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "estimator_source_sha256": ESTIMATOR_SHA256,
        "generator_source_sha256": GENERATOR_SHA256,
        "verifier_source_sha256": digest(__file__),
        "certificate_sha256": digest(certificate),
        "graph": estimate["graph"],
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous graph verification")
        OUTPUT.write_text(json.dumps(verify(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    result = verify()
    if read(OUTPUT) != result:
        raise ValueError("Paired graph replay receipt mismatch")
    print(json.dumps(result, sort_keys=True))
