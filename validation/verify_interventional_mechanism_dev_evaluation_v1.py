"""Read-only independent replay of the fixed development do comparison.

This checker recomputes outcomes and metrics without calling the evaluation
runner. It does not establish independent-world or original R0-R13 closure.
"""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import numpy as np
import torch

from ncd.mechanisms import load_mechanism, neural_values
from ncd.multiverse import GraphWorld
from verify_interventional_mechanism_dev_preflight_v2 import verify as verify_mixed_data
from verify_interventional_mechanism_dev_control_v1 import verify as verify_control_data


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_for(world_id, source):
    raw = sha256((world_id + ":" + source).encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def row_hashes(matrices):
    return {
        sha256(np.asarray(row, dtype="<f8").tobytes()).digest()
        for matrix in matrices for row in matrix
    }


def masks():
    return [
        {}, {0: -1.}, {0: 1.}, {1: -1.}, {1: 1.},
        {2: -1.}, {2: 1.}, {0: 1., 1: -1.}, {0: -1., 2: 1.},
    ]


def independent_metrics(world, predictors, exogenous, normalizers):
    if world.nodes != 3 or len(predictors) != 3:
        raise ValueError("Three predictor functions and nodes required")
    u = np.asarray(exogenous, dtype=float)
    normalizers = np.asarray(normalizers, dtype=float)
    if (u.ndim != 2 or u.shape[1] != 3 or not np.isfinite(u).all()
            or normalizers.shape != (3,) or not np.isfinite(normalizers).all()
            or np.any(normalizers <= 0)):
        raise ValueError("Invalid evaluation inputs")
    scales = np.asarray(world.scales or (1.,) * 3, dtype=float)
    conditions = []
    by_mask = {}
    all_executed = []
    for intervention in masks():
        observed = world.sample(
            interventions=intervention, exogenous=u, samples=len(u))
        nodes = []
        for target, predictor in enumerate(predictors):
            if target in intervention:
                nodes.append({
                    "node": target, "executed": False,
                    "reason": "structural_equation_cut_by_do",
                })
                continue
            truth = observed[:, target] - scales[target] * u[:, target]
            predicted = np.asarray(predictor(observed), dtype=float)
            if predicted.shape != (len(u),) or not np.isfinite(predicted).all():
                raise ValueError("Invalid trained mechanism prediction")
            error = (predicted - truth) / normalizers[target]
            node = {
                "node": target, "executed": True,
                "normalized_mae": float(np.mean(np.abs(error))),
                "normalized_mse": float(np.mean(error * error)),
                "predicted_mean": float(np.mean(predicted)),
                "true_mean": float(np.mean(truth)),
            }
            nodes.append(node)
            all_executed.append(node)
        by_mask[tuple(sorted(intervention.items()))] = nodes
        conditions.append({
            "interventions": {str(k): v for k, v in intervention.items()},
            "nodes": nodes,
        })
    contrasts = []
    for source in range(3):
        negative = by_mask[((source, -1.),)]
        positive = by_mask[((source, 1.),)]
        for target in range(3):
            if target == source:
                continue
            predicted = (positive[target]["predicted_mean"]
                         - negative[target]["predicted_mean"])
            truth = positive[target]["true_mean"] - negative[target]["true_mean"]
            contrasts.append({
                "source": source, "target": target,
                "normalized_abs_error": float(
                    abs(predicted - truth) / normalizers[target]),
                "predicted_contrast": predicted,
                "true_contrast": truth,
            })
    return {
        "conditions": conditions,
        "paired_contrasts": contrasts,
        "max_executed_normalized_mae": max(
            row["normalized_mae"] for row in all_executed),
        "max_executed_normalized_mse": max(
            row["normalized_mse"] for row in all_executed),
        "max_paired_contrast_normalized_abs_error": max(
            row["normalized_abs_error"] for row in contrasts),
        "unique_exogenous_draws": len(u),
        "paired_intervention_outcomes_are_independent_worlds": False,
    }


def same(expected, observed, path="result"):
    if type(expected) is not type(observed):
        raise ValueError(path + " type changed")
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            raise ValueError(path + " fields changed")
        for name in expected:
            same(expected[name], observed[name], path + "." + str(name))
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            raise ValueError(path + " length changed")
        for index, (left, right) in enumerate(zip(expected, observed)):
            same(left, right, path + "[" + str(index) + "]")
    elif isinstance(expected, float):
        if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(path + " numeric value changed")
    elif expected != observed:
        raise ValueError(path + " value changed")


def trained_predictors(plan_path):
    plan = read(plan_path)
    if (plan["schema"] != "ncd.interventional-mechanism-dev-training-plan.v1"
            or plan["true_equations_for_training"] is not False
            or plan.get("true_graph_for_training",
                        plan.get("truth_graph_for_training")) is not False):
        raise ValueError("Training plan scope changed")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Training source changed: " + name)
    receipt_path = ROOT / plan["preflight_receipt"]
    if digest(receipt_path) != plan["preflight_receipt_sha256"]:
        raise ValueError("Training preflight receipt changed")
    receipt = read(receipt_path)
    candidate = ROOT / plan["candidate_directory"]
    for name, expected in receipt["candidate_files_sha256"].items():
        if digest(candidate / name) != expected:
            raise ValueError("Training candidate data changed: " + name)
    output = ROOT / plan["output"]
    result_path = output / "training_result.json"
    result = read(result_path)
    if (result["schema"] != "ncd.interventional-mechanism-dev-training-result.v1"
            or result["status"] != "trained-development-only"
            or result["plan_sha256"] != digest(plan_path)
            or result["preflight_sha256"] != digest(receipt_path)
            or result["true_graph_used"] is not False
            or result["true_equations_used"] is not False
            or result["independent_confirmation_worlds"] != 0
            or result["original_claim_closed"] is not False
            or len(result["completed"]) != 3):
        raise ValueError("Training result provenance changed")
    graph = np.asarray(result["inferred_graph"], dtype=int)
    candidate_graph = read(candidate / "graph_prediction.json")
    if (graph.shape != (3, 3)
            or result["inferred_graph"] != candidate_graph["inferred_graph"]
            or candidate_graph["oracle_graph_used"] is not False):
        raise ValueError("Inferred training graph changed")
    predictors = []
    for target, item in enumerate(result["completed"]):
        if item["target"] != target:
            raise ValueError("Training target order changed")
        checkpoint = output / ("mechanism_" + str(target) + ".pt")
        record_path = output / ("mechanism_" + str(target) + ".json")
        if (digest(checkpoint) != item["checkpoint_sha256"]
                or digest(record_path) != item["record_sha256"]):
            raise ValueError("Training model or record hash changed")
        record = read(record_path)
        parents = tuple(map(int, np.flatnonzero(graph[:, target])))
        if (record["inferred_parents"] != list(parents)
                or record["checkpoint_sha256"] != item["checkpoint_sha256"]
                or record["true_graph_used"] is not False
                or record["true_equations_used"] is not False):
            raise ValueError("Training parent or truth-source binding changed")
        model = load_mechanism(checkpoint)
        if tuple(model.parents) != parents:
            raise ValueError("Checkpoint parent set differs")
        predictors.append(lambda observations, model=model:
                          neural_values(model, observations))
    return predictors, result, result_path


def verify(protocol_path=None, result_path=None):
    protocol_path = Path(protocol_path or
                         ROOT / "validation/interventional_mechanism_dev_evaluation_protocol_v1.json")
    plan = read(protocol_path)
    if (plan["schema"] != "ncd.interventional-mechanism-dev-evaluation-plan.v1"
            or plan["evaluation_rows"] != 512
            or plan["evaluation_source"] != "heldout_eval_exogenous_v1"
            or plan["same_exogenous_draws_across_interventions"] is not True
            or plan["original_normalized_mse_threshold"] != .01
            or plan["relative_improvement_rule"]
            != "strict_max_executed_normalized_mae_reduction"
            or plan["independent_confirmation_worlds"] != 0
            or plan["original_claim_closed"] is not False):
        raise ValueError("Frozen comparison contract changed")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Evaluation source changed: " + name)
    world_path = ROOT / plan["truth_world"]
    discovery_path = ROOT / plan["discovery"]
    if (digest(world_path) != plan["truth_world_sha256"]
            or digest(discovery_path) != plan["discovery_sha256"]):
        raise ValueError("Truth or common-normalizer source changed")
    world = GraphWorld.from_dict(read(world_path))
    if world.nodes != 3 or world.identity != plan["world_id"]:
        raise ValueError("Wrong development world")
    if verify_mixed_data()["status"] != "verified-development-preflight":
        raise ValueError("Mixed candidate data did not replay")
    if verify_control_data()["status"] != "verified-development-control":
        raise ValueError("Observational candidate data did not replay")
    torch.set_num_threads(1)
    models = {}
    training = {}
    for role in ("mixed", "control"):
        training_path = ROOT / plan[role + "_training_plan"]
        if digest(training_path) != plan[role + "_training_plan_sha256"]:
            raise ValueError(role + " frozen training plan changed")
        models[role], training[role], path = trained_predictors(training_path)
        training[role + "_sha256"] = digest(path)
    if training["mixed"]["inferred_graph"] != training["control"]["inferred_graph"]:
        raise ValueError("Training arms use different inferred graphs")
    with np.load(discovery_path, allow_pickle=False) as archive:
        if set(archive.files) != {"observations"}:
            raise ValueError("Unexpected discovery observations")
        normalizers = np.maximum(
            np.std(archive["observations"], axis=0), .05)
    if normalizers.shape != (3,):
        raise ValueError("Wrong common normalizer dimensions")
    seed = seed_for(world.identity, plan["evaluation_source"])
    _, u = world.sample(seed=seed, samples=512, return_exogenous=True)
    train_rows = set()
    for role in ("mixed", "control"):
        training_path = ROOT / plan[role + "_training_plan"]
        training_plan = read(training_path)
        archive_path = ROOT / training_plan["candidate_directory"] / "mechanism_observations.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            train_rows |= row_hashes(archive[name] for name in archive.files)
    eval_rows = row_hashes(
        world.sample(interventions=intervention, exogenous=u, samples=512)
        for intervention in masks())
    if train_rows & eval_rows:
        raise ValueError("Training and evaluation observations overlap")
    metrics = {
        role: independent_metrics(world, models[role], u, normalizers)
        for role in ("mixed", "control")
    }
    expected = {
        "schema": "ncd.interventional-mechanism-dev-evaluation.v1",
        "status": "development-only",
        "protocol_sha256": digest(protocol_path),
        "world_sha256": digest(world_path),
        "mixed_training_result_sha256": training["mixed_sha256"],
        "control_training_result_sha256": training["control_sha256"],
        "inferred_graph": training["mixed"]["inferred_graph"],
        "true_graph": np.asarray(world.graph, dtype=int).tolist(),
        "source_graph_exact": bool(np.array_equal(
            training["mixed"]["inferred_graph"], np.asarray(world.graph, dtype=int))),
        "common_normalizers": normalizers.tolist(),
        "evaluation_source": plan["evaluation_source"],
        "evaluation_seed": seed,
        "training_evaluation_exact_row_overlap": 0,
        "mixed": metrics["mixed"],
        "control": metrics["control"],
        "mixed_meets_0_01_max_normalized_mse": (
            metrics["mixed"]["max_executed_normalized_mse"] <= .01),
        "mixed_reduces_max_normalized_mae": (
            metrics["mixed"]["max_executed_normalized_mae"]
            < metrics["control"]["max_executed_normalized_mae"]),
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    result_path = Path(result_path or ROOT / plan["output"])
    observed = read(result_path)
    same(expected, observed)
    return {
        "schema": "ncd.interventional-mechanism-dev-evaluation-verification.v1",
        "status": "verified-development-only",
        "protocol_sha256": digest(protocol_path),
        "result_sha256": digest(result_path),
        "verifier_sha256": digest(__file__),
        "source_graph_exact": expected["source_graph_exact"],
        "mixed_meets_0_01_max_normalized_mse": expected[
            "mixed_meets_0_01_max_normalized_mse"],
        "mixed_reduces_max_normalized_mae": expected[
            "mixed_reduces_max_normalized_mae"],
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
