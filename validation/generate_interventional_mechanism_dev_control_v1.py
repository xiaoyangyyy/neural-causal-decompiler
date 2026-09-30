"""Build an observational-only control for the frozen do-data development world.

The generator reads truth metadata only to sample observed data. The trainer
receives only candidate/ and the same inferred graph as the do-data branch.
"""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ncd.multiverse import GraphWorld


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")


def seed_for(world_id, source):
    raw = sha256((world_id + ":" + source).encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def run(protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "ncd.interventional-dev-observational-control-plan.v1":
        raise ValueError("Wrong control plan")
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Control source hash mismatch: " + name)
    world_path = ROOT / protocol["truth_world"]
    if digest(world_path) != protocol["truth_world_sha256"]:
        raise ValueError("Frozen development world changed")
    world = GraphWorld.from_dict(
        json.loads(world_path.read_text(encoding="utf-8")))
    if world.identity != protocol["world_id"]:
        raise ValueError("Wrong development world")
    graph_path = ROOT / protocol["graph_prediction"]
    discovery_path = ROOT / protocol["discovery"]
    if (digest(graph_path) != protocol["graph_prediction_sha256"]
            or digest(discovery_path) != protocol["discovery_sha256"]):
        raise ValueError("Frozen graph/discovery input changed")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph["oracle_graph_used"] is not False:
        raise ValueError("Oracle graph not allowed")
    output = ROOT / protocol["output"]
    candidate = output / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(graph_path, candidate / "graph_prediction.json")
    shutil.copyfile(discovery_path, candidate / "discovery.npz")
    sources = [("fit_control_observation", 384),
               ("validation_control_observation", 128)]
    matrices = {}
    batches = []
    for source, rows in sources:
        matrix = world.sample(
            seed=seed_for(world.identity, source), samples=rows)
        matrices[source] = matrix
        batches.append({
            "source": source,
            "split": "fit" if source.startswith("fit_") else "validation",
            "matrix_key": source,
            "row_ids": [f"{world.identity}:{source}:{i}" for i in range(rows)],
            "interventions": {},
            "rows": rows,
        })
    mixed = ROOT / protocol["mixed_candidate"]
    if digest(mixed / "mechanism_observations.npz") != protocol["mixed_mechanism_observations_sha256"]:
        raise ValueError("Frozen mixed intervention observations changed")
    with np.load(mixed / "mechanism_observations.npz",
                 allow_pickle=False) as archive:
        mixed_rows = {
            sha256(np.asarray(row, dtype="<f8").tobytes()).digest()
            for name in archive.files for row in archive[name]
        }
    control_rows = {
        sha256(np.asarray(row, dtype="<f8").tobytes()).digest()
        for matrix in matrices.values() for row in matrix
    }
    if mixed_rows & control_rows:
        raise ValueError("Control and do-data observations overlap")
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
    receipt = {
        "schema": "ncd.interventional-dev-observational-control-preflight.v1",
        "status": "candidate-ready",
        "seed_conflict": False,
        "world_id": world.identity,
        "protocol_sha256": digest(protocol_path),
        "runner_sha256": digest(__file__),
        "candidate_files_sha256": {
            name: digest(candidate / name)
            for name in ("discovery.npz", "graph_prediction.json",
                         "mechanism_observations.npz", "batches.json")
        },
        "mechanism_observation_rows": 512,
        "mixed_vs_control_exact_row_overlap": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    save_json(ROOT / protocol["receipt"], receipt)
    return receipt


if __name__ == "__main__":
    path = (sys.argv[1] if len(sys.argv) == 2 else
            "validation/interventional_mechanism_dev_control_protocol_v1.json")
    print(json.dumps(run(ROOT / path), sort_keys=True))