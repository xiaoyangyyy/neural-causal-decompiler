"""Read-only replay of the frozen development preflight.

The checker regenerates samples and graph inference. It does not train models
or certify the original causal-decompilation objective.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ncd.active_intervention_graph import (
    load_active_factorized_graph, padded_observational_features,
)
from ncd.graph_model import dag_completion, decode_graph, graph_probabilities
from ncd.multiverse import generate_graph_worlds


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def seed_for(world_id, source):
    raw = sha256((world_id + ":" + source).encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def verify(protocol_path=None, receipt_path=None, output=None):
    protocol_path = Path(protocol_path or
                         ROOT / "validation/interventional_mechanism_dev_protocol_v2.json")
    receipt_path = Path(receipt_path or
                        ROOT / "validation/interventional_mechanism_dev_preflight_v2.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (protocol["schema"] != "ncd.interventional-mechanism-dev-plan.v2"
            or receipt["schema"] != "ncd.interventional-mechanism-dev-preflight.v2"
            or receipt["status"] != "candidate-ready"
            or receipt["protocol_sha256"] != digest(protocol_path)):
        raise ValueError("Protocol or receipt mismatch")
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source hash mismatch: " + name)
    if (digest(ROOT / "validation/run_interventional_mechanism_dev_preflight_v2.py")
            != receipt["runner_sha256"]):
        raise ValueError("Preflight runner hash mismatch")
    teacher_path = ROOT / protocol["graph_teacher"]
    if (digest(teacher_path) != protocol["graph_teacher_sha256"]
            or digest(teacher_path) != receipt["graph_teacher_sha256"]):
        raise ValueError("Graph teacher hash mismatch")
    base = Path(output or ROOT / protocol["output"])
    candidate, truth = base / "candidate", base / "truth_only"
    expected_candidate = {
        "discovery.npz", "graph_prediction.json", "mechanism_observations.npz",
        "batches.json",
    }
    if {p.name for p in candidate.iterdir() if p.is_file()} != expected_candidate:
        raise ValueError("Candidate data files differ from protocol")
    if set(receipt["candidate_files_sha256"]) != expected_candidate:
        raise ValueError("Candidate hash list incomplete")
    for name, expected in receipt["candidate_files_sha256"].items():
        if digest(candidate / name) != expected:
            raise ValueError("Candidate file hash mismatch: " + name)
    if digest(truth / "world.json") != receipt["truth_world_sha256"]:
        raise ValueError("Truth-only world hash mismatch")
    worlds = generate_graph_worlds("dev", 10, 3, seed=8100, samples=96)
    world = worlds[0]
    if (world.identity != receipt["world_id"]
            or json.loads((truth / "world.json").read_text(encoding="utf-8"))
            != json.loads(json.dumps(world.metadata()))):
        raise ValueError("True development world mismatch")
    confirmation_ids = {
        item.identity
        for seed in (8101, 8102)
        for split in ("test_id", "test_function", "test_noise",
                      "test_scale", "test_intervention")
        for item in generate_graph_worlds(split, 10, 3, seed=seed, samples=96)
    }
    if world.identity in confirmation_ids or receipt["seed_conflict"] is not False:
        raise ValueError("Development/confirmation world collision")
    with np.load(candidate / "discovery.npz", allow_pickle=False) as data:
        if set(data.files) != {"observations"}:
            raise ValueError("Discovery archive schema mismatch")
        discovery = data["observations"]
    expected_discovery = world.sample(
        seed=seed_for(world.identity, "discovery"), samples=96)
    if not np.array_equal(discovery, expected_discovery):
        raise ValueError("Discovery observations mismatch")
    graph = json.loads((candidate / "graph_prediction.json").read_text(encoding="utf-8"))
    if graph["oracle_graph_used"] is not False:
        raise ValueError("Oracle graph leakage")
    torch.set_num_threads(1)
    teacher = load_active_factorized_graph(teacher_path)
    probabilities = graph_probabilities(
        teacher, padded_observational_features(discovery)[None])[0]
    partial, decode = decode_graph(probabilities)
    inferred, choices = dag_completion(partial)
    if (not np.array_equal(probabilities, np.asarray(graph["probabilities"]))
            or graph["partial_graph"] != partial.astype(int).tolist()
            or graph["inferred_graph"] != inferred.astype(int).tolist()
            or graph["decode"] != decode or graph["completion_choices"] != choices):
        raise ValueError("Frozen graph prediction replay mismatch")
    metadata = json.loads((candidate / "batches.json").read_text(encoding="utf-8"))
    if (metadata["oracle_graph_used"] is not False
            or metadata["true_equations_used_for_training"] is not False
            or metadata["world_id"] != world.identity
            or metadata["fit_rows"] != 384 or metadata["validation_rows"] != 128):
        raise ValueError("Candidate batch metadata mismatch")
    expected_layout = [("observation", {}, 96, 32)]
    expected_layout.extend(
        (f"do_{target}_{side}", {target: value}, 48, 16)
        for target in range(3) for side, value in (("minus", -1.), ("plus", 1.))
    )
    expected_metadata = []
    with np.load(candidate / "mechanism_observations.npz",
                 allow_pickle=False) as archive:
        for kind, interventions, fit_rows, val_rows in expected_layout:
            for split, rows in (("fit", fit_rows), ("validation", val_rows)):
                source = split + "_" + kind
                if source not in archive.files:
                    raise ValueError("Missing mechanism observation batch")
                expected = world.sample(
                    seed=seed_for(world.identity, source),
                    interventions=interventions, samples=rows)
                if not np.array_equal(archive[source], expected):
                    raise ValueError("Mechanism observations mismatch: " + source)
                expected_metadata.append({
                    "source": source, "split": split, "matrix_key": source,
                    "row_ids": [f"{world.identity}:{source}:{i}" for i in range(rows)],
                    "interventions": {str(k): v for k, v in interventions.items()},
                    "rows": rows,
                })
        if set(archive.files) != {item["matrix_key"] for item in expected_metadata}:
            raise ValueError("Extra mechanism observation batch")
    if metadata["batches"] != expected_metadata or receipt["mechanism_observation_rows"] != 512:
        raise ValueError("Frozen batch schedule mismatch")
    return {
        "schema": "ncd.interventional-mechanism-dev-preflight-verification.v2",
        "status": "verified-development-preflight",
        "protocol_sha256": digest(protocol_path),
        "receipt_sha256": digest(receipt_path),
        "candidate_files_sha256": receipt["candidate_files_sha256"],
        "world_id": world.identity,
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))