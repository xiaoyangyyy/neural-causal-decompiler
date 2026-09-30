"""Freeze development-world observations and inferred graph for do-data training.

This preflight sees true world metadata only to generate observations. The
training entry point later reads candidate/ and never reads truth_only/.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ncd.graph_model import (
    dag_completion, decode_graph, graph_probabilities, load_graph_model,
    pair_features,
)
from ncd.multiverse import generate_graph_worlds


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")


def sample_seed(world_id, source):
    raw = sha256((world_id + ":" + source).encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def run(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "ncd.interventional-mechanism-dev-plan.v1":
        raise ValueError("Unexpected protocol schema")
    if (protocol["world_seed"], protocol["world_split"],
            protocol["nodes"], protocol["world_index"],
            protocol["discovery_rows"]) != (8100, "dev", 3, 0, 96):
        raise ValueError("Development world binding changed")
    if protocol["mechanism_observation_budget"] != 512:
        raise ValueError("Mechanism observation budget changed")
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source hash mismatch: " + name)
    teacher_path = ROOT / protocol["graph_teacher"]
    if digest(teacher_path) != protocol["graph_teacher_sha256"]:
        raise ValueError("Frozen graph teacher hash mismatch")
    worlds = generate_graph_worlds(
        protocol["world_split"], 10, protocol["nodes"],
        seed=protocol["world_seed"], samples=protocol["discovery_rows"])
    world = worlds[protocol["world_index"]]
    confirmation = {
        item.identity
        for seed in (8101, 8102)
        for split in ("test_id", "test_function", "test_noise",
                      "test_scale", "test_intervention")
        for item in generate_graph_worlds(split, 10, 3, seed=seed, samples=96)
    }
    if world.identity in confirmation:
        raise ValueError("Development world identity conflicts with confirmation")
    output = ROOT / protocol["output"]
    candidate = output / "candidate"
    truth = output / "truth_only"
    candidate.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    discovery_source = "discovery"
    discovery = world.sample(
        seed=sample_seed(world.identity, discovery_source), samples=96)
    np.savez_compressed(candidate / "discovery.npz", observations=discovery)
    teacher = load_graph_model(teacher_path)
    probabilities = graph_probabilities(
        teacher, pair_features(discovery)[None])[0]
    partial, decode = decode_graph(probabilities)
    inferred, choices = dag_completion(partial)
    save_json(candidate / "graph_prediction.json", {
        "schema": "ncd.interventional-dev-graph-prediction.v1",
        "teacher_sha256": digest(teacher_path),
        "discovery_sha256": digest(candidate / "discovery.npz"),
        "discovery_rows": 96,
        "probabilities": probabilities.tolist(),
        "partial_graph": partial.astype(int).tolist(),
        "inferred_graph": inferred.astype(int).tolist(),
        "decode": decode,
        "completion_choices": choices,
        "oracle_graph_used": False,
        "original_claim_closed": False,
    })
    batches = []
    matrices = {}
    layouts = [("observation", {}, 96, 32)]
    layouts.extend(
        (f"do_{target}_{side}", {target: value}, 48, 16)
        for target in range(3) for side, value in (("minus", -1.), ("plus", 1.))
    )
    for kind, interventions, fit_rows, val_rows in layouts:
        for split, rows in (("fit", fit_rows), ("validation", val_rows)):
            source = split + "_" + kind
            data = world.sample(
                seed=sample_seed(world.identity, source),
                interventions=interventions, samples=rows)
            matrices[source] = data
            batches.append({
                "source": source,
                "split": split,
                "matrix_key": source,
                "row_ids": [f"{world.identity}:{source}:{i}" for i in range(rows)],
                "interventions": {str(k): v for k, v in interventions.items()},
                "rows": rows,
            })
    if sum(item["rows"] for item in batches) != 512:
        raise ValueError("Mechanism observation count mismatch")
    np.savez_compressed(candidate / "mechanism_observations.npz", **matrices)
    save_json(candidate / "batches.json", {
        "schema": "ncd.interventional-mechanism-batches.v1",
        "world_id": world.identity,
        "batches": batches,
        "fit_rows": 384,
        "validation_rows": 128,
        "source_seed_rule": "sha256(world_id + ':' + source), first 8 bytes little-endian mod 2^63-1",
        "oracle_graph_used": False,
        "true_equations_used_for_training": False,
        "sampling_independence_proved": False,
        "original_claim_closed": False,
    })
    save_json(truth / "world.json", world.metadata())
    result = {
        "schema": "ncd.interventional-mechanism-dev-preflight.v1",
        "status": "candidate-ready",
        "world_id": world.identity,
        "protocol_sha256": digest(protocol_path),
        "runner_sha256": digest(__file__),
        "graph_teacher_sha256": digest(teacher_path),
        "seed_conflict": False,
        "candidate_files_sha256": {
            name: digest(candidate / name)
            for name in ("discovery.npz", "graph_prediction.json",
                         "mechanism_observations.npz", "batches.json")
        },
        "truth_world_sha256": digest(truth / "world.json"),
        "mechanism_observation_rows": 512,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    save_json(ROOT / protocol["receipt"], result)
    return result


if __name__ == "__main__":
    path = (sys.argv[1] if len(sys.argv) == 2 else
            "validation/interventional_mechanism_dev_protocol_v1.json")
    print(json.dumps(run(ROOT / path), sort_keys=True))