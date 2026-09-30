"""Freeze a fresh independent-do linear-graph confirmation before new worlds run."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation/independent_do_linear_confirmation_protocol_v1.json"
PRIOR = ROOT / "validation/interventional_role_confirmation_protocol_v1.json"
DIAGNOSTIC = ROOT / "validation/independent_do_linear_graph_diagnostic_v1.json"
SOURCES = (
    "ncd/independent_do_graph.py", "ncd/multiverse.py", "ncd/worlds.py",
    "ncd/graphs.py", "ncd/model.py", "ncd/graph_model.py",
    "ncd/active_intervention_graph.py",
    "validation/confirm_independent_do_linear_graph_v1.py",
    "validation/verify_independent_do_linear_graph_v1.py",
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("Existing frozen protocol retained")
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if (prior["graph_teacher_sha256"] != digest(ROOT / prior["graph_teacher"])
            or diagnostic["linear_gaussian_worlds"] != 60
            or diagnostic["status"] != "posthoc-archived-independent-do-diagnostic"
            or diagnostic["thresholds_exploratory_after_source_results_seen"] != [.05,.1,.2]):
        raise ValueError("Development source or teacher changed")
    plan = {
        "schema": "ncd.independent-do-linear-confirmation-plan.v1",
        "status": "frozen-before-confirmation",
        "seeds": [8401,8402], "nodes": [3,5,8],
        "environments": ["test_id","test_function","test_noise","test_scale","test_intervention"],
        "linear_indices": [0,5],
        "declared_worlds": 60,
        "discovery_rows": 96,
        "mechanism_observation_rows": 512,
        "do_group_total_rows": 384,
        "threshold": .2,
        "relative_gate": {"candidate_min_exact": 30,
                          "candidate_minus_teacher_min": 20,
                          "each_node_size_min_exact": 10,
                          "paired_sign_p_max": "1/100"},
        "graph_teacher": prior["graph_teacher"],
        "graph_teacher_sha256": prior["graph_teacher_sha256"],
        "source_diagnostic_sha256": digest(DIAGNOSTIC),
        "source_diagnostic_selected_after_results_seen": True,
        "fresh_confirmation_worlds_not_evaluated_when_frozen": True,
        "candidate_has_384_independent_do_rows_teacher_has_96_observation_rows": True,
        "max_stage_seconds": 43200,
        "artifact_budget_bytes": 8*1024**3,
        "memory_bytes": 8*1024**3,
        "training_threads": 1,
        "output": "runs/independent_do_linear_confirmation_v1",
        "source_sha256": {p: digest(ROOT / p) for p in SOURCES},
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    OUTPUT.write_text(json.dumps(plan, sort_keys=True, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps({"status": plan["status"], "worlds": plan["declared_worlds"],
                      "threshold": plan["threshold"], "protocol_sha256": digest(OUTPUT)},
                     sort_keys=True))


if __name__ == "__main__":
    main()