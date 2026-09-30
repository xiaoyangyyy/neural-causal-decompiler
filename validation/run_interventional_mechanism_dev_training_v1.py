"""Train development mechanisms from candidate-only batches and inferred DAG.

This runner never opens truth_only/world.json. Its result is development-only
and cannot close an original causal recovery claim.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import numpy as np

from interventional_mechanism_fit_v1 import (
    ObservationBatch, fit, prepare, save_checkpoint,
)
from ncd.graphs import topological_order


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")


def load(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["schema"] != "ncd.interventional-mechanism-dev-training-plan.v1":
        raise ValueError("Unexpected training plan")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Training source hash mismatch: " + name)
    preflight_path = ROOT / plan["preflight_receipt"]
    if digest(preflight_path) != plan["preflight_receipt_sha256"]:
        raise ValueError("Preflight receipt hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight["status"] != "candidate-ready" or preflight["seed_conflict"]:
        raise ValueError("Development preflight not ready")
    candidate = ROOT / plan["candidate_directory"]
    for name, expected in preflight["candidate_files_sha256"].items():
        if digest(candidate / name) != expected:
            raise ValueError("Candidate input hash mismatch: " + name)
    graph = json.loads((candidate / "graph_prediction.json").read_text(encoding="utf-8"))
    batches = json.loads((candidate / "batches.json").read_text(encoding="utf-8"))
    if graph["oracle_graph_used"] is not False:
        raise ValueError("Oracle graph forbidden")
    if (batches["true_equations_used_for_training"] is not False
            or batches["oracle_graph_used"] is not False):
        raise ValueError("Truth metadata forbidden in candidate inputs")
    inferred = np.asarray(graph["inferred_graph"], dtype=bool)
    if inferred.shape != (3, 3):
        raise ValueError("Expected frozen three-node inferred graph")
    topological_order(inferred)
    matrices = np.load(candidate / "mechanism_observations.npz",
                       allow_pickle=False)
    expected_keys = {item["matrix_key"] for item in batches["batches"]}
    if set(matrices.files) != expected_keys:
        raise ValueError("Observation matrix key mismatch")
    fitting, validation = [], []
    for item in batches["batches"]:
        data = matrices[item["matrix_key"]]
        if len(data) != item["rows"]:
            raise ValueError("Observation row count mismatch")
        entry = ObservationBatch(
            item["source"], data, tuple(item["row_ids"]),
            {int(k): float(v) for k, v in item["interventions"].items()},
        )
        if item["split"] == "fit":
            fitting.append(entry)
        elif item["split"] == "validation":
            validation.append(entry)
        else:
            raise ValueError("Unknown mechanism split")
    if sum(len(b.data) for b in fitting + validation) != 512:
        raise ValueError("Mechanism observation budget mismatch")
    return plan, preflight, inferred, fitting, validation


def run(plan_path, train=False):
    started = monotonic()
    plan, preflight, graph, fitting, validation = load(plan_path)
    diagnostics = []
    for target in range(3):
        parents = tuple(map(int, np.flatnonzero(graph[:, target])))
        _, _, record = prepare(fitting, validation, target, parents, 3)
        diagnostics.append(record)
    if not train:
        return {
            "schema": "ncd.interventional-mechanism-dev-training-precheck.v1",
            "status": "validated-without-training",
            "preflight_sha256": digest(ROOT / plan["preflight_receipt"]),
            "targets": diagnostics,
            "original_claim_closed": False,
            "original_objective_achieved": False,
        }
    output = ROOT / plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    prior_result = output / "training_result.json"
    if prior_result.exists():
        previous = json.loads(prior_result.read_text(encoding="utf-8"))
        if (previous.get("plan_sha256") != digest(plan_path)
                or previous.get("preflight_sha256") != digest(ROOT / plan["preflight_receipt"])
                or len(previous.get("completed", [])) != 3):
            raise ValueError("Existing training result does not match frozen inputs")
        for item in previous["completed"]:
            target = item["target"]
            if (digest(output / f"mechanism_{target}.pt") != item["checkpoint_sha256"]
                    or digest(output / f"mechanism_{target}.json") != item["record_sha256"]):
                raise ValueError("Existing training checkpoint mismatch")
        return previous
    completed = []
    for target in range(3):
        if monotonic() - started > plan["stage_seconds"]:
            raise TimeoutError("Development training stage exceeded budget")
        parents = tuple(map(int, np.flatnonzero(graph[:, target])))
        checkpoint = output / f"mechanism_{target}.pt"
        record_path = output / f"mechanism_{target}.json"
        if checkpoint.exists() or record_path.exists():
            if not checkpoint.exists() or not record_path.exists():
                raise ValueError("Partial node checkpoint requires manual audit")
            saved = json.loads(record_path.read_text(encoding="utf-8"))
            if (saved.get("checkpoint_sha256") != digest(checkpoint)
                    or saved.get("seed") != plan["seed"] + target
                    or saved.get("epochs") != plan["epochs"]
                    or saved.get("inferred_parents") != list(parents)):
                raise ValueError("Existing node checkpoint does not match frozen inputs")
        else:
            model, record = fit(
                fitting, validation, target, parents, 3,
                seed=plan["seed"] + target,
                epochs=plan["epochs"], width=plan["width"])
            saved = save_checkpoint(model, record, checkpoint)
            save_json(record_path, saved)
        used_bytes = sum(file.stat().st_size for file in output.rglob("*") if file.is_file())
        if used_bytes > plan["artifact_budget_bytes"]:
            raise ValueError("Training artifact budget exceeded")
        completed.append({
            "target": target,
            "checkpoint_sha256": digest(checkpoint),
            "record_sha256": digest(record_path),
        })
    result = {
        "schema": "ncd.interventional-mechanism-dev-training-result.v1",
        "status": "trained-development-only",
        "plan_sha256": digest(plan_path),
        "preflight_sha256": digest(ROOT / plan["preflight_receipt"]),
        "inferred_graph": graph.astype(int).tolist(),
        "completed": completed,
        "elapsed_seconds": monotonic() - started,
        "true_graph_used": False,
        "true_equations_used": False,
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    save_json(output / "training_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="validation/interventional_mechanism_dev_training_protocol_v1.json")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(ROOT / args.plan, train=args.train), sort_keys=True))


if __name__ == "__main__":
    main()