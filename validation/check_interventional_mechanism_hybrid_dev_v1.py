"""Independent model-level replay of the posthoc structural-role hybrid."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
from ncd.multiverse import GraphWorld
from verify_interventional_mechanism_dev_evaluation_v1 import (
    independent_metrics, seed_for, trained_predictors, verify as verify_source,
)

CERTIFICATE = ROOT / "validation/interventional_mechanism_hybrid_dev_v1.json"
OUTPUT = ROOT / "validation/interventional_mechanism_hybrid_dev_verification_v1.json"
GENERATOR_SHA256 = "d3e27f79cdb5901159a12b647b27e7e438441c55e8b76cf039fe1edd29df59bf"
RESULT_SHA256 = "a95722c8411296646c8018f0602e06a736943359e9168c41e2012c50c20899e1"
VERIFICATION_SHA256 = "1775295eb04585761a8188004751626e5cc571151a027deeb225d5eefada1b7c"
GRAPH_SHA256 = "d7d541c5fb0b1d8ca7d0f38ba2e79415011eb05bee326a2f8a9de85f0f583966"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equal(expected, observed, label="result"):
    if type(expected) is not type(observed):
        raise ValueError(label + " type mismatch")
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            raise ValueError(label + " keys mismatch")
        for key in expected:
            equal(expected[key], observed[key], label + "." + str(key))
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            raise ValueError(label + " length mismatch")
        for index, (left, right) in enumerate(zip(expected, observed)):
            equal(left, right, label + "[" + str(index) + "]")
    elif isinstance(expected, float):
        if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(label + " numeric mismatch")
    elif expected != observed:
        raise ValueError(label + " mismatch")


def verify(certificate=CERTIFICATE):
    if digest(ROOT / "validation/evaluate_interventional_mechanism_hybrid_dev_v1.py") != GENERATOR_SHA256:
        raise ValueError("Hybrid generator changed")
    result_path = ROOT / "runs/interventional_mechanism_dev_comparison_v1.json"
    source_receipt = ROOT / "validation/interventional_mechanism_dev_result_verification_v1.json"
    graph_path = ROOT / "runs/interventional_mechanism_dev_v2/candidate/graph_prediction.json"
    if (digest(result_path) != RESULT_SHA256
            or digest(source_receipt) != VERIFICATION_SHA256
            or digest(graph_path) != GRAPH_SHA256):
        raise ValueError("Hybrid source inputs changed")
    prior = verify_source()
    if prior["status"] != "verified-development-only" or prior["result_sha256"] != RESULT_SHA256:
        raise ValueError("Original independent model replay failed")
    graph_record = read(graph_path)
    graph = graph_record["inferred_graph"]
    if graph_record["oracle_graph_used"] is not False or len(graph) != 3:
        raise ValueError("Wrong inferred graph source")
    roles = ["control" if sum(graph[parent][target] for parent in range(3)) == 0
             else "mixed" for target in range(3)]
    protocol = read(ROOT / "validation/interventional_mechanism_dev_evaluation_protocol_v1.json")
    mixed, mixed_training, _ = trained_predictors(ROOT / protocol["mixed_training_plan"])
    control, control_training, _ = trained_predictors(ROOT / protocol["control_training_plan"])
    if mixed_training["inferred_graph"] != graph or control_training["inferred_graph"] != graph:
        raise ValueError("Training graph changed")
    selected = [mixed[i] if roles[i] == "mixed" else control[i] for i in range(3)]
    world = GraphWorld.from_dict(read(ROOT / protocol["truth_world"]))
    if world.identity != protocol["world_id"] or world.nodes != 3:
        raise ValueError("Wrong truth-only evaluation world")
    with np.load(ROOT / protocol["discovery"], allow_pickle=False) as archive:
        if set(archive.files) != {"observations"}:
            raise ValueError("Wrong frozen normalizer source")
        normalizers = np.maximum(np.std(archive["observations"], axis=0), .05)
    _, exogenous = world.sample(
        seed=seed_for(world.identity, protocol["evaluation_source"]),
        samples=512, return_exogenous=True,
    )
    torch.set_num_threads(1)
    metrics = independent_metrics(world, selected, exogenous, normalizers)
    for row in metrics["conditions"]:
        for node in row["nodes"]:
            node["source_arm"] = roles[node["node"]]
    cert = read(certificate)
    if set(cert) != {
        "schema", "status", "source_result_sha256", "source_verification_sha256",
        "inferred_graph_sha256", "generator_source_sha256", "inferred_graph",
        "node_role_rule", "selected_model_arm_by_node", "metrics",
        "local_normalized_mse_threshold", "passes_local_max_normalized_mse_on_this_world",
        "rule_selected_after_development_metrics_seen",
        "same_world_posthoc_selection_and_evaluation",
        "independent_confirmation_worlds", "full_scm_rollout_evaluated",
        "original_claim_closed", "original_objective_achieved",
    }:
        raise ValueError("Hybrid certificate fields changed")
    if (cert["schema"] != "ncd.interventional-mechanism-hybrid-dev.v1"
            or cert["status"] != "posthoc-development-only"
            or cert["source_result_sha256"] != RESULT_SHA256
            or cert["source_verification_sha256"] != VERIFICATION_SHA256
            or cert["inferred_graph_sha256"] != GRAPH_SHA256
            or cert["generator_source_sha256"] != GENERATOR_SHA256
            or cert["inferred_graph"] != graph
            or cert["node_role_rule"] != "indegree zero: observational control; positive indegree: mixed do training"
            or cert["selected_model_arm_by_node"] != roles
            or cert["local_normalized_mse_threshold"] != .01
            or cert["passes_local_max_normalized_mse_on_this_world"] is not True
            or cert["rule_selected_after_development_metrics_seen"] is not True
            or cert["same_world_posthoc_selection_and_evaluation"] is not True
            or cert["independent_confirmation_worlds"] != 0
            or cert["full_scm_rollout_evaluated"] is not False
            or cert["original_claim_closed"] is not False
            or cert["original_objective_achieved"] is not False):
        raise ValueError("Hybrid scope changed")
    equal(metrics, cert["metrics"])
    if metrics["max_executed_normalized_mse"] >= .01:
        raise ValueError("Hybrid does not pass local development MSE gate")
    return {
        "schema": "ncd.interventional-mechanism-hybrid-dev-verification.v1",
        "status": "verified-posthoc-development-only",
        "generator_source_sha256": GENERATOR_SHA256,
        "verifier_source_sha256": digest(__file__),
        "certificate_sha256": digest(certificate),
        "max_executed_normalized_mse": metrics["max_executed_normalized_mse"],
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous hybrid verification receipt")
        OUTPUT.write_text(json.dumps(verify(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    result = verify()
    if read(OUTPUT) != result:
        raise ValueError("Hybrid receipt mismatch")
    print(json.dumps(result, sort_keys=True))
