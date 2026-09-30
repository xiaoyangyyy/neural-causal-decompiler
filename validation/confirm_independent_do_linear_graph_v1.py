"""Fresh-seed confirmation of independent-do mean graph recovery on linear worlds."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ncd.active_intervention_graph import load_active_factorized_graph, padded_observational_features
from ncd.graph_model import dag_completion, decode_graph, graph_probabilities
from ncd.independent_do_graph import recover_independent_do_means
from ncd.multiverse import generate_graph_worlds

PLAN = ROOT / "validation/independent_do_linear_confirmation_protocol_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    os.replace(temp, path)


def sample_seed(world_id, source):
    return int.from_bytes(sha256((world_id+":"+source).encode()).digest()[:8],
                          "little") % (2**63-1)


def units(plan):
    return [(seed, nodes, split, index) for seed in plan["seeds"]
            for nodes in plan["nodes"] for split in plan["environments"]
            for index in plan["linear_indices"]]


def world_for(unit):
    seed, nodes, split, index = unit
    world = generate_graph_worlds(split, 10, nodes, seed=seed, samples=96)[index]
    if world.family != "linear_gaussian":
        raise ValueError("Declared world is not linear Gaussian")
    return world


def preflight(plan):
    expected = {
        "schema": "ncd.independent-do-linear-confirmation-plan.v1",
        "status": "frozen-before-confirmation",
        "seeds": [8401, 8402], "nodes": [3, 5, 8],
        "environments": ["test_id", "test_function", "test_noise", "test_scale", "test_intervention"],
        "linear_indices": [0, 5], "discovery_rows": 96,
        "mechanism_observation_rows": 512,
        "do_group_total_rows": 384,
        "threshold": 0.2,
        "relative_gate": {"candidate_min_exact": 30, "candidate_minus_teacher_min": 20,
                          "each_node_size_min_exact": 10, "paired_sign_p_max": "1/100"},
        "max_stage_seconds": 43200,
        "artifact_budget_bytes": 8*1024**3,
        "memory_bytes": 8*1024**3,
        "training_threads": 1,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError("Frozen protocol changed: "+key)
    if plan["original_claim_closed"] is not False:
        raise ValueError("Original claim scope inflated")
    if digest(ROOT / plan["graph_teacher"]) != plan["graph_teacher_sha256"]:
        raise ValueError("Frozen graph teacher changed")
    for relative, expected_hash in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError("Source hash changed: "+relative)
    fresh = {world_for(unit).identity for unit in units(plan)}
    if len(fresh) != 60:
        raise ValueError("Repeated fresh world identity")
    historical = {generate_graph_worlds(split, 10, nodes, seed=seed, samples=96)[index].identity
                  for seed in (8100, 8101, 8102, 8301, 8302)
                  for nodes in (3, 5, 8) for split in plan["environments"]
                  for index in (0, 5)}
    if fresh & historical:
        raise ValueError("Fresh confirmation overlaps prior worlds")
    return plan


def do_groups(world):
    n = world.nodes
    base, extra = divmod(384, 2*n)
    groups, coverage = {}, []
    for source in range(n):
        for level in (-1, 1):
            k = 2*source + int(level == 1)
            count = base + int(k < extra)
            validation = max(6, count//4)
            fit = count-validation
            name = f"do_{source}_{'plus' if level == 1 else 'minus'}"
            parts = []
            for split, rows in (("fit", fit), ("validation", validation)):
                batch_source = split+"_"+name
                parts.append(world.sample(seed=sample_seed(world.identity, batch_source),
                                          interventions={source: float(level)}, samples=rows))
            groups[(source, level)] = np.concatenate(parts)
            coverage.append({"source": source, "level": level,
                             "fit_rows": fit, "validation_rows": validation,
                             "total_rows": count})
    if sum(row["total_rows"] for row in coverage) != 384:
        raise ValueError("Do budget changed")
    return groups, coverage


def candidate(folder, plan, teacher):
    discovery_path = folder / "candidate/discovery.npz"
    groups_path = folder / "candidate/do_groups.npz"
    with np.load(discovery_path, allow_pickle=False) as archive:
        if set(archive.files) != {"observations"}:
            raise ValueError("Discovery archive changed")
        discovery = archive["observations"]
    n = discovery.shape[1]
    if discovery.shape != (96, n):
        raise ValueError("Discovery shape changed")
    with np.load(groups_path, allow_pickle=False) as archive:
        expected = {f"do_{i}_{'plus' if level == 1 else 'minus'}"
                    for i in range(n) for level in (-1, 1)}
        if set(archive.files) != expected:
            raise ValueError("Independent do archive changed")
        groups = {(i, level): archive[f"do_{i}_{'plus' if level == 1 else 'minus'}"]
                  for i in range(n) for level in (-1, 1)}
    recovered = recover_independent_do_means(groups, n, threshold=plan["threshold"])
    with torch.no_grad():
        probabilities = graph_probabilities(
            teacher, padded_observational_features(discovery)[None])[0]
    partial, _ = decode_graph(probabilities)
    neural, _ = dag_completion(partial)
    return recovered, neural.astype(int).tolist()


def compute_unit(unit, plan, teacher):
    name = f"seed_{unit[0]}_n{unit[1]}_{unit[2]}_{unit[3]}"
    folder = ROOT / plan["output"] / "units" / name
    if folder.exists():
        raise FileExistsError("Existing unit retained: "+name)
    (folder / "candidate").mkdir(parents=True)
    (folder / "truth_only").mkdir()
    world = world_for(unit)
    discovery = world.sample(seed=sample_seed(world.identity, "discovery"), samples=96)
    np.savez_compressed(folder / "candidate/discovery.npz", observations=discovery)
    groups, coverage = do_groups(world)
    np.savez_compressed(folder / "candidate/do_groups.npz", **{
        f"do_{i}_{'plus' if level == 1 else 'minus'}": values
        for (i, level), values in groups.items()})
    write(folder / "candidate/coverage.json", coverage)
    write(folder / "truth_only/world.json", world.metadata())
    recovered, neural = candidate(folder, plan, teacher)
    truth = np.asarray(world.graph, dtype=int)
    graph = np.asarray(recovered["graph_source_target"], dtype=int)
    result = {
        "schema": "ncd.independent-do-linear-confirmation-unit.v1",
        "status": "computed-pending-independent-replay",
        "unit": list(unit), "world_id": world.identity,
        "protocol_sha256": digest(PLAN),
        "candidate_inputs_sha256": {p.name: digest(p) for p in (folder / "candidate").iterdir()},
        "teacher_sha256": plan["graph_teacher_sha256"],
        "candidate": recovered,
        "teacher_graph_source_target": neural,
        "truth_graph_source_target": truth.tolist(),
        "candidate_exact": bool(np.array_equal(graph, truth)),
        "teacher_exact": bool(np.array_equal(neural, truth)),
        "candidate_shd": int(np.count_nonzero(graph != truth)),
        "teacher_shd": int(np.count_nonzero(np.asarray(neural) != truth)),
        "original_claim_closed": False,
    }
    write(folder / "result.json", result)
    return {"unit": list(unit), "candidate_exact": result["candidate_exact"],
            "teacher_exact": result["teacher_exact"], "acyclic": recovered["acyclic"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        raise SystemExit("Choose exactly --preflight or --run")
    plan = preflight(read(PLAN))
    if args.preflight:
        print(json.dumps({"status": "ready", "declared_worlds": len(units(plan))}, sort_keys=True))
        return
    torch.set_num_threads(1)
    started = monotonic()
    output = ROOT / plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    write(output / "protocol.json", plan)
    teacher = load_active_factorized_graph(ROOT / plan["graph_teacher"])
    completed = []
    for unit in units(plan):
        if monotonic()-started > plan["max_stage_seconds"]:
            raise TimeoutError("Stage exceeds frozen time budget")
        completed.append(compute_unit(unit, plan, teacher))
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > plan["artifact_budget_bytes"]:
            raise RuntimeError("Artifact budget exceeded")
    print(json.dumps({"status": "computed-pending-independent-replay",
                      "declared_worlds": 60, "completed_worlds": len(completed),
                      "candidate_exact": sum(row["candidate_exact"] for row in completed),
                      "teacher_exact": sum(row["teacher_exact"] for row in completed),
                      "original_claim_closed": False}, sort_keys=True))


if __name__ == "__main__":
    main()