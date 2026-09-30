"""Independent dev-world evaluator for do-trained and observational mechanisms.

Only this evaluator reads the true world. The same 512 exogenous draws are
reused across interventions; paired outcomes are not independent worlds.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ncd.mechanisms import load_mechanism, neural_values
from ncd.multiverse import GraphWorld


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def paired_seed(world_id, source):
    raw = sha256((world_id + ":" + source).encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def true_deterministic(world, observed, target):
    scales = np.asarray(world.scales or (1.,) * world.nodes, dtype=float)
    internal = observed / scales
    value = sum((term.evaluate(internal) for term in world.equations[target]),
                start=np.zeros(len(observed), dtype=float))
    return np.broadcast_to(value * scales[target], (len(observed),))


def intervention_masks(nodes):
    masks = [{}]
    masks.extend({target: value}
                 for target in range(nodes) for value in (-1., 1.))
    if nodes >= 3:
        masks.extend(({0: 1., 1: -1.}, {0: -1., 2: 1.}))
    return masks


def evaluate_predictors(world, predictors, exogenous, normalizers):
    if len(predictors) != world.nodes:
        raise ValueError("One predictor is required per node")
    u = np.asarray(exogenous, dtype=float)
    if u.shape[1:] != (world.nodes,) or not np.isfinite(u).all():
        raise ValueError("Wrong held-out exogenous shape")
    scale = np.asarray(normalizers, dtype=float)
    if scale.shape != (world.nodes,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid common normalizer")
    rows = []
    cached = {}
    for mask in intervention_masks(world.nodes):
        observed = world.sample(
            interventions=mask, exogenous=u, samples=len(u))
        node_rows = []
        for target, predictor in enumerate(predictors):
            if target in mask:
                node_rows.append({
                    "node": target, "executed": False,
                    "reason": "structural_equation_cut_by_do",
                })
                continue
            truth = true_deterministic(world, observed, target)
            predicted = np.asarray(predictor(observed), dtype=float)
            if predicted.shape != (len(u),) or not np.isfinite(predicted).all():
                raise ValueError("Invalid mechanism prediction")
            error = (predicted - truth) / scale[target]
            node_rows.append({
                "node": target, "executed": True,
                "normalized_mae": float(np.mean(np.abs(error))),
                "normalized_mse": float(np.mean(error * error)),
                "predicted_mean": float(np.mean(predicted)),
                "true_mean": float(np.mean(truth)),
            })
        key = tuple(sorted(mask.items()))
        cached[key] = node_rows
        rows.append({
            "interventions": {str(k): v for k, v in mask.items()},
            "nodes": node_rows,
        })
    contrasts = []
    for source in range(world.nodes):
        minus = cached[((source, -1.),)]
        plus = cached[((source, 1.),)]
        for target in range(world.nodes):
            if target == source:
                continue
            predicted = plus[target]["predicted_mean"] - minus[target]["predicted_mean"]
            truth = plus[target]["true_mean"] - minus[target]["true_mean"]
            contrasts.append({
                "source": source, "target": target,
                "normalized_abs_error": float(abs(predicted - truth) / scale[target]),
                "predicted_contrast": predicted,
                "true_contrast": truth,
            })
    executed = [node for row in rows for node in row["nodes"] if node["executed"]]
    return {
        "conditions": rows,
        "paired_contrasts": contrasts,
        "max_executed_normalized_mae": max(
            node["normalized_mae"] for node in executed),
        "max_executed_normalized_mse": max(
            node["normalized_mse"] for node in executed),
        "max_paired_contrast_normalized_abs_error": max(
            row["normalized_abs_error"] for row in contrasts),
        "unique_exogenous_draws": len(u),
        "paired_intervention_outcomes_are_independent_worlds": False,
    }


def verified_predictors(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    result_path = ROOT / plan["output"] / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result["schema"] != "ncd.interventional-mechanism-dev-training-result.v1"
            or result["status"] != "trained-development-only"
            or result["plan_sha256"] != digest(plan_path)
            or result["preflight_sha256"] != digest(ROOT / plan["preflight_receipt"])
            or result["true_graph_used"] is not False
            or result["true_equations_used"] is not False
            or len(result["completed"]) != 3):
        raise ValueError("Training result not eligible for independent evaluation")
    preflight = json.loads(
        (ROOT / plan["preflight_receipt"]).read_text(encoding="utf-8"))
    candidate = ROOT / plan["candidate_directory"]
    for name, expected in preflight["candidate_files_sha256"].items():
        if digest(candidate / name) != expected:
            raise ValueError("Training candidate input changed: " + name)
    graph = np.asarray(result["inferred_graph"], dtype=bool)
    observed_graph = json.loads(
        (candidate / "graph_prediction.json").read_text(encoding="utf-8"))
    if (observed_graph["oracle_graph_used"] is not False
            or observed_graph["inferred_graph"] != result["inferred_graph"]):
        raise ValueError("Inferred graph changed after training")
    predictors = []
    for target, item in enumerate(result["completed"]):
        if item["target"] != target:
            raise ValueError("Training target order mismatch")
        checkpoint = ROOT / plan["output"] / f"mechanism_{target}.pt"
        record_path = ROOT / plan["output"] / f"mechanism_{target}.json"
        if (digest(checkpoint) != item["checkpoint_sha256"]
                or digest(record_path) != item["record_sha256"]):
            raise ValueError("Trained checkpoint or record hash mismatch")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        parents = tuple(map(int, np.flatnonzero(graph[:, target])))
        if (record["inferred_parents"] != list(parents)
                or record["checkpoint_sha256"] != item["checkpoint_sha256"]):
            raise ValueError("Trained parent binding mismatch")
        model = load_mechanism(checkpoint)
        if tuple(model.parents) != parents:
            raise ValueError("Checkpoint uses different parents")
        predictors.append(lambda observed, model=model: neural_values(model, observed))
    return predictors, result


def observed_row_hashes(matrices):
    return {
        sha256(np.asarray(row, dtype="<f8").tobytes()).digest()
        for matrix in matrices for row in matrix
    }


def training_observation_hashes(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    archive_path = ROOT / plan["candidate_directory"] / "mechanism_observations.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        return observed_row_hashes(archive[name] for name in archive.files)


def run(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "ncd.interventional-mechanism-dev-evaluation-plan.v1":
        raise ValueError("Wrong evaluation protocol")
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Evaluation source hash mismatch: " + name)
    if protocol["evaluation_rows"] != 512:
        raise ValueError("Held-out evaluation budget changed")
    if (protocol["original_normalized_mse_threshold"] != .01
            or protocol["relative_improvement_rule"]
            != "strict_max_executed_normalized_mae_reduction"):
        raise ValueError("Evaluation thresholds changed")
    world_path = ROOT / protocol["truth_world"]
    if digest(world_path) != protocol["truth_world_sha256"]:
        raise ValueError("True development world changed")
    world = GraphWorld.from_dict(
        json.loads(world_path.read_text(encoding="utf-8")))
    if world.identity != protocol["world_id"]:
        raise ValueError("Wrong development world")
    mixed_plan = ROOT / protocol["mixed_training_plan"]
    control_plan = ROOT / protocol["control_training_plan"]
    if (digest(mixed_plan) != protocol["mixed_training_plan_sha256"]
            or digest(control_plan) != protocol["control_training_plan_sha256"]):
        raise ValueError("Training plan changed")
    mixed, mixed_result = verified_predictors(mixed_plan)
    control, control_result = verified_predictors(control_plan)
    if mixed_result["inferred_graph"] != control_result["inferred_graph"]:
        raise ValueError("Candidate and control graphs differ")
    torch.set_num_threads(1)
    discovery_path = ROOT / protocol["discovery"]
    if digest(discovery_path) != protocol["discovery_sha256"]:
        raise ValueError("Frozen common normalizer source changed")
    with np.load(discovery_path, allow_pickle=False) as archive:
        discovery = archive["observations"]
    normalizers = np.maximum(np.std(discovery, axis=0), .05)
    seed = paired_seed(world.identity, protocol["evaluation_source"])
    _, exogenous = world.sample(
        seed=seed, samples=512, return_exogenous=True)
    fit_hashes = (training_observation_hashes(mixed_plan)
                  | training_observation_hashes(control_plan))
    eval_hashes = observed_row_hashes(
        world.sample(interventions=mask, exogenous=exogenous, samples=512)
        for mask in intervention_masks(world.nodes))
    if fit_hashes & eval_hashes:
        raise ValueError("Training/evaluation observation row leakage")
    mixed_metrics = evaluate_predictors(world, mixed, exogenous, normalizers)
    control_metrics = evaluate_predictors(world, control, exogenous, normalizers)
    result = {
        "schema": "ncd.interventional-mechanism-dev-evaluation.v1",
        "status": "development-only",
        "protocol_sha256": digest(protocol_path),
        "world_sha256": digest(world_path),
        "mixed_training_result_sha256": digest(ROOT / json.loads(
            mixed_plan.read_text(encoding="utf-8"))["output"] / "training_result.json"),
        "control_training_result_sha256": digest(ROOT / json.loads(
            control_plan.read_text(encoding="utf-8"))["output"] / "training_result.json"),
        "inferred_graph": mixed_result["inferred_graph"],
        "true_graph": np.asarray(world.graph, dtype=int).tolist(),
        "source_graph_exact": bool(np.array_equal(
            mixed_result["inferred_graph"], np.asarray(world.graph, dtype=int))),
        "common_normalizers": normalizers.tolist(),
        "evaluation_source": protocol["evaluation_source"],
        "evaluation_seed": seed,
        "training_evaluation_exact_row_overlap": 0,
        "mixed": mixed_metrics,
        "control": control_metrics,
        "mixed_meets_0_01_max_normalized_mse": (
            mixed_metrics["max_executed_normalized_mse"]
            <= protocol["original_normalized_mse_threshold"]),
        "mixed_reduces_max_normalized_mae": (
            mixed_metrics["max_executed_normalized_mae"]
            < control_metrics["max_executed_normalized_mae"]),
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    output = ROOT / protocol["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    path = (sys.argv[1] if len(sys.argv) == 2 else
            "validation/interventional_mechanism_dev_evaluation_protocol_v1.json")
    outcome = run(ROOT / path)
    print(json.dumps({
        "status": outcome["status"],
        "mixed_max_nmse": outcome["mixed"]["max_executed_normalized_mse"],
        "control_max_nmse": outcome["control"]["max_executed_normalized_mse"],
        "original_claim_closed": False,
    }, sort_keys=True))