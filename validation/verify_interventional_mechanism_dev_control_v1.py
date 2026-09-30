"""Read-only replay of the observational control for the do-data dev world."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ncd.multiverse import GraphWorld


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def seed_for(world_id, source):
    raw = sha256((world_id + ":" + source).encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def row_hashes(archive):
    return {
        sha256(np.asarray(row, dtype="<f8").tobytes()).digest()
        for name in archive.files for row in archive[name]
    }


def verify(protocol_path=None, receipt_path=None, output=None):
    protocol_path = Path(protocol_path or
                         ROOT / "validation/interventional_mechanism_dev_control_protocol_v1.json")
    receipt_path = Path(receipt_path or
                        ROOT / "validation/interventional_mechanism_dev_control_preflight_v1.json")
    plan = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (plan["schema"] != "ncd.interventional-dev-observational-control-plan.v1"
            or receipt["schema"] != "ncd.interventional-dev-observational-control-preflight.v1"
            or receipt["status"] != "candidate-ready"
            or receipt["protocol_sha256"] != digest(protocol_path)
            or receipt["runner_sha256"] != digest(
                ROOT / "validation/generate_interventional_mechanism_dev_control_v1.py")):
        raise ValueError("Control plan or receipt mismatch")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Control source hash mismatch: " + name)
    world_path = ROOT / plan["truth_world"]
    graph_path = ROOT / plan["graph_prediction"]
    discovery_path = ROOT / plan["discovery"]
    if (digest(world_path) != plan["truth_world_sha256"]
            or digest(graph_path) != plan["graph_prediction_sha256"]
            or digest(discovery_path) != plan["discovery_sha256"]):
        raise ValueError("Frozen control source changed")
    world = GraphWorld.from_dict(
        json.loads(world_path.read_text(encoding="utf-8")))
    if world.identity != plan["world_id"] or world.identity != receipt["world_id"]:
        raise ValueError("Wrong control world")
    base = Path(output or ROOT / plan["output"])
    candidate = base / "candidate"
    names = {"discovery.npz", "graph_prediction.json",
             "mechanism_observations.npz", "batches.json"}
    if {p.name for p in candidate.iterdir() if p.is_file()} != names:
        raise ValueError("Control candidate file list mismatch")
    if set(receipt["candidate_files_sha256"]) != names:
        raise ValueError("Control receipt file list incomplete")
    for name, expected in receipt["candidate_files_sha256"].items():
        if digest(candidate / name) != expected:
            raise ValueError("Control candidate hash mismatch: " + name)
    if (digest(candidate / "graph_prediction.json") != digest(graph_path)
            or digest(candidate / "discovery.npz") != digest(discovery_path)):
        raise ValueError("Control graph or discovery differs from do-data arm")
    metadata = json.loads((candidate / "batches.json").read_text(encoding="utf-8"))
    if (metadata["world_id"] != world.identity
            or metadata["fit_rows"] != 384
            or metadata["validation_rows"] != 128
            or metadata["oracle_graph_used"] is not False
            or metadata["true_equations_used_for_training"] is not False):
        raise ValueError("Control batch metadata mismatch")
    expected_batches = []
    with np.load(candidate / "mechanism_observations.npz",
                 allow_pickle=False) as archive:
        sources = [("fit_control_observation", 384),
                   ("validation_control_observation", 128)]
        if set(archive.files) != {name for name, _ in sources}:
            raise ValueError("Control matrix keys differ")
        for source, rows in sources:
            expected = world.sample(
                seed=seed_for(world.identity, source), samples=rows)
            if not np.array_equal(archive[source], expected):
                raise ValueError("Control observations fail regeneration")
            expected_batches.append({
                "source": source,
                "split": "fit" if source.startswith("fit_") else "validation",
                "matrix_key": source,
                "row_ids": [f"{world.identity}:{source}:{i}" for i in range(rows)],
                "interventions": {},
                "rows": rows,
            })
        control_rows = row_hashes(archive)
    if metadata["batches"] != expected_batches:
        raise ValueError("Control batch schedule mismatch")
    mixed_path = ROOT / plan["mixed_candidate"] / "mechanism_observations.npz"
    if digest(mixed_path) != plan["mixed_mechanism_observations_sha256"]:
        raise ValueError("Mixed-arm observations changed")
    with np.load(mixed_path, allow_pickle=False) as mixed:
        if control_rows & row_hashes(mixed):
            raise ValueError("Control and mixed rows overlap")
    if (receipt["mixed_vs_control_exact_row_overlap"] != 0
            or receipt["mechanism_observation_rows"] != 512
            or receipt["seed_conflict"] is not False):
        raise ValueError("Control receipt overstates separation")
    return {
        "schema": "ncd.interventional-dev-observational-control-verification.v1",
        "status": "verified-development-control",
        "protocol_sha256": digest(protocol_path),
        "receipt_sha256": digest(receipt_path),
        "world_id": world.identity,
        "control_rows": 512,
        "exact_row_overlap_with_mixed_arm": 0,
        "independent_confirmation_worlds": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))