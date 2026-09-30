"""Posthoc development diagnostic: choose frozen mechanisms by inferred parent role."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "runs/interventional_mechanism_dev_comparison_v1.json"
VERIFICATION = ROOT / "validation/interventional_mechanism_dev_result_verification_v1.json"
GRAPH = ROOT / "runs/interventional_mechanism_dev_v2/candidate/graph_prediction.json"
OUTPUT = ROOT / "validation/interventional_mechanism_hybrid_dev_v1.json"
RESULT_SHA256 = "a95722c8411296646c8018f0602e06a736943359e9168c41e2012c50c20899e1"
VERIFICATION_SHA256 = "1775295eb04585761a8188004751626e5cc571151a027deeb225d5eefada1b7c"
GRAPH_SHA256 = "d7d541c5fb0b1d8ca7d0f38ba2e79415011eb05bee326a2f8a9de85f0f583966"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_roles(graph):
    """No truth or evaluation metrics enter this frozen role rule."""
    n = len(graph)
    if n != 3 or any(len(row) != n for row in graph):
        raise ValueError("Expected three-node inferred graph")
    if any(type(graph[i][j]) is not int or graph[i][j] not in (0, 1)
           for i in range(n) for j in range(n)):
        raise ValueError("Invalid graph entries")
    if any(graph[i][i] for i in range(n)):
        raise ValueError("Inferred graph has self edge")
    return ["mixed" if any(graph[parent][target] for parent in range(n))
            else "control" for target in range(n)]


def compose(original, roles):
    arms = {name: original[name]["conditions"] for name in ("mixed", "control")}
    if len(roles) != 3 or set(roles) - set(arms):
        raise ValueError("Invalid role selection")
    if len(arms["mixed"]) != 9 or len(arms["control"]) != 9:
        raise ValueError("Expected nine common intervention conditions")
    conditions = []
    by_mask = {}
    executed = []
    for mixed, control in zip(arms["mixed"], arms["control"]):
        if mixed["interventions"] != control["interventions"]:
            raise ValueError("Arms use different intervention masks")
        mask = mixed["interventions"]
        if len(mixed["nodes"]) != 3 or len(control["nodes"]) != 3:
            raise ValueError("Incomplete node metrics")
        rows = []
        for node in range(3):
            left, right = mixed["nodes"][node], control["nodes"][node]
            if left["node"] != node or right["node"] != node or left["executed"] is not right["executed"]:
                raise ValueError("Different target or branch coverage")
            if left["executed"] and not math.isclose(
                left["true_mean"], right["true_mean"], rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("Arms use different true outcomes")
            chosen = dict((left if roles[node] == "mixed" else right))
            chosen["source_arm"] = roles[node]
            rows.append(chosen)
            if chosen["executed"]:
                executed.append(chosen)
        conditions.append({"interventions": mask, "nodes": rows})
        by_mask[tuple(sorted((int(k), v) for k, v in mask.items()))] = rows
    normalizers = original["common_normalizers"]
    if len(normalizers) != 3 or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in normalizers
    ):
        raise ValueError("Invalid frozen common normalizers")
    contrasts = []
    for source in range(3):
        negative = by_mask[((source, -1.0),)]
        positive = by_mask[((source, 1.0),)]
        for target in range(3):
            if target == source:
                continue
            predicted = positive[target]["predicted_mean"] - negative[target]["predicted_mean"]
            truth = positive[target]["true_mean"] - negative[target]["true_mean"]
            contrasts.append({
                "source": source, "target": target,
                "predicted_contrast": predicted, "true_contrast": truth,
                "normalized_abs_error": abs(predicted - truth) / normalizers[target],
            })
    return {
        "conditions": conditions,
        "paired_contrasts": contrasts,
        "max_executed_normalized_mae": max(row["normalized_mae"] for row in executed),
        "max_executed_normalized_mse": max(row["normalized_mse"] for row in executed),
        "max_paired_contrast_normalized_abs_error": max(
            row["normalized_abs_error"] for row in contrasts),
        "unique_exogenous_draws": 512,
        "paired_intervention_outcomes_are_independent_worlds": False,
    }


def compute():
    if (digest(RESULT) != RESULT_SHA256 or digest(VERIFICATION) != VERIFICATION_SHA256
            or digest(GRAPH) != GRAPH_SHA256):
        raise ValueError("A frozen development input changed")
    graph_record = read(GRAPH)
    if graph_record["oracle_graph_used"] is not False:
        raise ValueError("Inferred graph used oracle truth")
    graph = graph_record["inferred_graph"]
    roles = choose_roles(graph)
    verification = read(VERIFICATION)
    if (verification["status"] != "verified-development-only"
            or verification["verification"]["status"] != "verified-development-only"
            or verification["verification"]["result_sha256"] != RESULT_SHA256
            or verification["original_claim_closed"] is not False):
        raise ValueError("Source result lacks independent replay")
    source = read(RESULT)
    if (source["status"] != "development-only"
            or source["inferred_graph"] != graph
            or source["mixed"]["unique_exogenous_draws"] != 512
            or source["control"]["unique_exogenous_draws"] != 512
            or source["original_claim_closed"] is not False):
        raise ValueError("Wrong frozen source result")
    metrics = compose(source, roles)
    return {
        "schema": "ncd.interventional-mechanism-hybrid-dev.v1",
        "status": "posthoc-development-only",
        "source_result_sha256": RESULT_SHA256,
        "source_verification_sha256": VERIFICATION_SHA256,
        "inferred_graph_sha256": GRAPH_SHA256,
        "generator_source_sha256": digest(__file__),
        "inferred_graph": graph,
        "node_role_rule": "indegree zero: observational control; positive indegree: mixed do training",
        "selected_model_arm_by_node": roles,
        "metrics": metrics,
        "local_normalized_mse_threshold": 0.01,
        "passes_local_max_normalized_mse_on_this_world": metrics["max_executed_normalized_mse"] < 0.01,
        "rule_selected_after_development_metrics_seen": True,
        "same_world_posthoc_selection_and_evaluation": True,
        "independent_confirmation_worlds": 0,
        "full_scm_rollout_evaluated": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(path=OUTPUT):
    actual = read(path)
    if actual != compute():
        raise ValueError("Hybrid development certificate mismatch")
    return {"status": actual["status"], "certificate_sha256": digest(path),
            "max_executed_normalized_mse": actual["metrics"]["max_executed_normalized_mse"],
            "original_claim_closed": False}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous hybrid certificate")
        OUTPUT.write_text(json.dumps(compute(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(verify(), sort_keys=True))
