"""Development certificate: paired-do total-effect inversion on one archived world."""
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.paired_linear_graph import recover_graph

PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
OUTPUT = ROOT / "validation/paired_linear_graph_development_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
NOISE = ["1/7", "-1/11", "2/13"]
LEVELS = [["1", "-1"]] * 3


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def topo(graph):
    n = len(graph)
    remaining = set(range(n))
    order = []
    while remaining:
        available = [j for j in sorted(remaining)
                     if all(not graph[i][j] for i in remaining)]
        if not available:
            raise ValueError("World graph is cyclic")
        order.extend(available)
        remaining.difference_update(available)
    return order


def exact_world_response(world, interventions, noise):
    graph = world["graph"]
    result = [Q(0)] * len(graph)
    scales = [Q(float(s)) for s in world["scales"]]
    for node in topo(graph):
        if node in interventions:
            result[node] = Q(interventions[node]) / scales[node]
            continue
        value = noise[node]
        for term in world["equations"][node]:
            if term["operator"] != "linear" or len(term["parents"]) != 1:
                raise ValueError("World is not linear")
            value += Q(float(term["coefficient"])) * result[term["parents"][0]]
        result[node] = value
    return [str(value * scales[node]) for node, value in enumerate(result)]


def compute(package=PACKAGE):
    package = Path(package)
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
    world = read(package / "world.json")
    if (world["family"] != "linear_gaussian" or world["noise_family"] != "gaussian"
            or world["root_shift"] is not False or world["scales"] != [1.0] * 3
            or world["graph"] != [[0, 1, 1], [0, 0, 0], [0, 1, 0]]
            or world["split"] != "test_id"):
        raise ValueError("Wrong frozen linear-Gaussian world")
    u = [Q(value) for value in NOISE]
    plus = [exact_world_response(world, {i: Q(1)}, u) for i in range(3)]
    minus = [exact_world_response(world, {i: Q(-1)}, u) for i in range(3)]
    # The estimator receives only response vectors and intervention levels.
    estimate = recover_graph(plus, minus, LEVELS)
    if estimate["graph"] != world["graph"]:
        raise ValueError("Recovered graph differs from this oracle world")
    true_direct = [[Q(0) for _ in range(3)] for _ in range(3)]
    for target, terms in enumerate(world["equations"]):
        for term in terms:
            true_direct[target][term["parents"][0]] += Q(float(term["coefficient"]))
    if estimate["direct_effect"] != [
        [str(value) for value in row] for row in true_direct
    ]:
        raise ValueError("Recovered direct coefficients differ from truth")
    return {
        "schema": "ncd.paired-linear-graph-development.v1",
        "status": "verified-scoped-exact-recovery",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "world_sha256": digest(package / "world.json"),
        "estimator_source_sha256": digest(ROOT / "ncd/paired_linear_graph.py"),
        "generator_source_sha256": digest(__file__),
        "world_id": world["world_id"],
        "exogenous_witness": NOISE,
        "intervention_levels": LEVELS,
        "plus_responses": plus,
        "minus_responses": minus,
        "estimate": estimate,
        "oracle_graph_used_only_for_evaluation": True,
        "paired_exogenous_assumption": True,
        "independent_intervention_samples_certified": False,
        "floating_inference_rounding_certified": False,
        "one_archived_development_world_only": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(path=OUTPUT, package=PACKAGE):
    actual = read(path)
    if actual != compute(package):
        raise ValueError("Paired linear graph certificate mismatch")
    return {"status": actual["status"], "certificate_sha256": digest(path),
            "graph": actual["estimate"]["graph"], "original_claim_closed": False}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous paired graph certificate")
        OUTPUT.write_text(json.dumps(compute(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(verify(), sort_keys=True))
