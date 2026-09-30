"""Frozen independent-world confirmation of the inferred-role mechanism rule.

The candidate trainer reads only sampled observations and an inferred graph.
The evaluator opens truth metadata and reports oracle-noise rollout separately.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import numpy as np
import torch

from ncd.active_intervention_graph import load_active_factorized_graph, padded_observational_features
from ncd.graph_model import dag_completion, decode_graph, graph_probabilities
from ncd.graphs import topological_order
from ncd.mechanisms import load_mechanism, neural_values
from ncd.multiverse import generate_graph_worlds
from ncd.proof_process import run_isolated
from interventional_mechanism_fit_v1 import ObservationBatch, fit, save_checkpoint

PLAN = ROOT / "validation/interventional_role_confirmation_protocol_v1.json"
SPLITS = ("test_id", "test_function", "test_noise", "test_scale", "test_intervention")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sample_seed(world_id, source):
    return int.from_bytes(sha256((world_id + ":" + source).encode()).digest()[:8], "little") % (2**63 - 1)


def declared_units(plan):
    return [(seed, nodes, split, i) for seed in plan["seeds"]
            for nodes in plan["nodes"] for split in plan["environments"]
            for i in range(plan["worlds_per_cell"])]


def unit_name(unit):
    seed, nodes, split, index = unit
    return f"seed_{seed}_n{nodes}_{split}_{index}"


def world_for(unit, plan):
    seed, nodes, split, index = unit
    return generate_graph_worlds(split, plan["worlds_per_cell"], nodes,
                                 seed=seed, samples=plan["discovery_rows"])[index]


def preflight(plan_path=PLAN):
    plan = read(plan_path)
    expected = {
        "schema": "ncd.interventional-role-confirmation-plan.v1",
        "status": "frozen-before-confirmation",
        "seeds": [8301, 8302], "nodes": [3, 5, 8],
        "environments": list(SPLITS), "worlds_per_cell": 10,
        "discovery_rows": 96, "mechanism_observation_rows": 512,
        "evaluation_rows": 512, "epochs": 120, "width": 48,
        "normalized_mse_threshold": .01,
        "role_rule": "inferred indegree 0: control; positive indegree: mixed",
        "max_stage_seconds": 43200, "max_unit_seconds": 600,
        "memory_bytes": 8 * 1024**3, "artifact_budget_bytes": 8 * 1024**3,
        "training_threads": 1,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError("Frozen protocol changed: " + key)
    if plan["original_claim_closed"] is not False:
        raise ValueError("Confirmation scope inflated")
    for relative, expected_hash in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError("Source hash mismatch: " + relative)
    if digest(ROOT / plan["graph_teacher"]) != plan["graph_teacher_sha256"]:
        raise ValueError("Graph teacher hash mismatch")
    fresh = {world_for(unit, plan).identity for unit in declared_units(plan)}
    if len(fresh) != 300:
        raise ValueError("Repeated confirmation world identity")
    used = {world.identity for seed in (8100, 8101, 8102)
            for nodes in (3, 5, 8) for split in ("dev", *SPLITS)
            for world in generate_graph_worlds(split, 10, nodes, seed=seed, samples=96)}
    if fresh & used:
        raise ValueError("New confirmation world overlaps historical world")
    return plan


def batch_layout(nodes):
    layout = [("observation", {}, 96, 32)]
    base, extra = divmod(384, 2 * nodes)
    for k, (target, value) in enumerate((t, v) for t in range(nodes) for v in (-1., 1.)):
        rows = base + int(k < extra)
        validation = max(6, rows // 4)
        layout.append((f"do_{target}_{'minus' if value < 0 else 'plus'}",
                       {target: value}, rows - validation, validation))
    assert sum(f + v for _, _, f, v in layout) == 512
    return layout


def masks(nodes):
    return ([{}] + [{j: v} for j in range(nodes) for v in (-1., 1.)]
            + [{0: 1., 1: -1.}, {0: -1., 2: 1.}])


def make_candidate(world, plan, folder):
    candidate = folder / "candidate"
    truth = folder / "truth_only"
    candidate.mkdir(parents=True)
    truth.mkdir(parents=True)
    discovery = world.sample(seed=sample_seed(world.identity, "discovery"), samples=96)
    np.savez_compressed(candidate / "discovery.npz", observations=discovery)
    teacher = load_active_factorized_graph(ROOT / plan["graph_teacher"])
    p = graph_probabilities(teacher, padded_observational_features(discovery)[None])[0]
    partial, decoding = decode_graph(p)
    graph, choices = dag_completion(partial)
    write(candidate / "graph_prediction.json", {
        "inferred_graph": graph.astype(int).tolist(), "oracle_graph_used": False,
        "partial_graph": partial.astype(int).tolist(), "decoding": decoding,
        "completion_choices": choices, "teacher_sha256": plan["graph_teacher_sha256"],
    })
    batches, matrices = [], {}
    for kind, intervention, fit_rows, val_rows in batch_layout(world.nodes):
        for split, count in (("fit", fit_rows), ("validation", val_rows)):
            source = split + "_" + kind
            matrices[source] = world.sample(seed=sample_seed(world.identity, source),
                                            interventions=intervention, samples=count)
            batches.append({
                "source": source, "split": split, "rows": count,
                "interventions": {str(k): v for k, v in intervention.items()},
                "row_ids": [f"{world.identity}:{source}:{i}" for i in range(count)],
            })
    np.savez_compressed(candidate / "mechanism_observations.npz", **matrices)
    write(candidate / "batches.json", {
        "world_id": world.identity, "batches": batches,
        "oracle_graph_used": False, "true_equations_used_for_training": False,
    })
    for split, count in (("fit", 384), ("validation", 128)):
        source = "control_" + split
        observations = world.sample(seed=sample_seed(world.identity, source), samples=count)
        np.savez_compressed(candidate / (source + ".npz"), observations=observations)
    write(truth / "world.json", world.metadata())
    return {p.name: digest(p) for p in candidate.iterdir() if p.is_file()}


def train_candidate(candidate, folder, world_id, plan):
    """No true graph, true equation, or GraphWorld enters this function."""
    graph_record = read(candidate / "graph_prediction.json")
    batch_record = read(candidate / "batches.json")
    if (graph_record["oracle_graph_used"] is not False
            or batch_record["oracle_graph_used"] is not False
            or batch_record["true_equations_used_for_training"] is not False
            or batch_record["world_id"] != world_id):
        raise ValueError("Candidate data provenance failed")
    graph = np.asarray(graph_record["inferred_graph"], dtype=bool)
    nodes = len(graph)
    if nodes not in (3, 5, 8) or graph.shape != (nodes, nodes):
        raise ValueError("Wrong inferred graph dimensions")
    topological_order(graph)
    fitting, validation = [], []
    with np.load(candidate / "mechanism_observations.npz", allow_pickle=False) as archive:
        if set(archive.files) != {b["source"] for b in batch_record["batches"]}:
            raise ValueError("Missing mechanism batch")
        for batch in batch_record["batches"]:
            data = archive[batch["source"]]
            if data.shape != (batch["rows"], nodes):
                raise ValueError("Invalid mechanism batch shape")
            observation = ObservationBatch(
                batch["source"], data, tuple(batch["row_ids"]),
                {int(k): float(v) for k, v in batch["interventions"].items()})
            (fitting if batch["split"] == "fit" else validation).append(observation)
    arms = {"mixed": (fitting, validation)}
    control = []
    for split, count in (("fit", 384), ("validation", 128)):
        source = "control_" + split
        with np.load(candidate / (source + ".npz"), allow_pickle=False) as archive:
            data = archive["observations"]
        if data.shape != (count, nodes):
            raise ValueError("Invalid control batch shape")
        control.append(ObservationBatch(
            source, data, tuple(f"{world_id}:{source}:{i}" for i in range(count)), {}))
    arms["control"] = ([control[0]], [control[1]])
    completed = {}
    for arm, (fit_batches, val_batches) in arms.items():
        records = []
        for target in range(nodes):
            parents = tuple(map(int, np.flatnonzero(graph[:, target])))
            seed = sample_seed(world_id, f"train_{arm}_{target}") % (2**31 - 1)
            model, record = fit(fit_batches, val_batches, target, parents, nodes,
                                seed=seed, epochs=plan["epochs"], width=plan["width"])
            checkpoint = folder / "trained" / arm / f"mechanism_{target}.pt"
            saved = save_checkpoint(model, record, checkpoint)
            write(checkpoint.with_suffix(".json"), saved)
            records.append({
                "node": target, "parents": list(parents),
                "checkpoint_sha256": digest(checkpoint),
                "record_sha256": digest(checkpoint.with_suffix(".json")),
            })
        completed[arm] = records
    return graph, completed


def load_predictors(folder, graph, arm):
    predictors = []
    for target in range(len(graph)):
        model = load_mechanism(folder / "trained" / arm / f"mechanism_{target}.pt")
        parents = tuple(map(int, np.flatnonzero(graph[:, target])))
        if tuple(model.parents) != parents:
            raise ValueError("Checkpoint parent mismatch")
        predictors.append(lambda values, model=model: neural_values(model, values))
    return predictors


def evaluate(world, graph, predictors, exogenous, normalizers):
    """Local teacher-forced error and full rollout with true exogenous noise."""
    scales = np.asarray(world.scales or (1.,) * world.nodes, dtype=float)
    local, rollout, means = [], [], {}
    for intervention in masks(world.nodes):
        actual = world.sample(interventions=intervention, exogenous=exogenous,
                              samples=len(exogenous))
        predicted = np.zeros_like(actual)
        for target in topological_order(graph):
            predicted[:, target] = (
                intervention[target] if target in intervention else
                predictors[target](predicted) + scales[target] * exogenous[:, target])
        means[tuple(sorted(intervention.items()))] = (predicted.mean(0), actual.mean(0))
        for target in range(world.nodes):
            if target in intervention:
                continue
            true_mechanism = actual[:, target] - scales[target] * exogenous[:, target]
            local_error = (predictors[target](actual) - true_mechanism) / normalizers[target]
            rollout_error = (predicted[:, target] - actual[:, target]) / normalizers[target]
            key = {"mask": {str(k): v for k, v in intervention.items()}, "node": target}
            local.append({**key, "normalized_mse": float(np.mean(local_error**2)),
                          "normalized_mae": float(np.mean(abs(local_error)))})
            rollout.append({**key, "normalized_mse": float(np.mean(rollout_error**2)),
                            "normalized_mae": float(np.mean(abs(rollout_error)))})
    contrasts = []
    for source in range(world.nodes):
        plus_p, plus_t = means[((source, 1.),)]
        minus_p, minus_t = means[((source, -1.),)]
        for target in range(world.nodes):
            if target != source:
                contrasts.append({
                    "source": source, "target": target,
                    "normalized_abs_error": float(abs(
                        (plus_p[target] - minus_p[target]) -
                        (plus_t[target] - minus_t[target])) / normalizers[target]),
                })
    return {
        "local": local, "rollout_with_oracle_noise": rollout,
        "max_local_normalized_mse": max(x["normalized_mse"] for x in local),
        "max_rollout_normalized_mse": max(x["normalized_mse"] for x in rollout),
        "max_rollout_paired_contrast_error": max(x["normalized_abs_error"] for x in contrasts),
        "contrasts": contrasts, "independent_exogenous_draws": len(exogenous),
        "noise_source": "true_exogenous_oracle_diagnostic_only",
    }


def run_unit(unit, plan_path=PLAN):
    plan = preflight(plan_path)
    world = world_for(unit, plan)
    folder = ROOT / plan["output"] / "units" / unit_name(unit)
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "unit_manifest.json", {"status": "started", "unit": list(unit),
                                          "world_id": world.identity, "plan_sha256": digest(plan_path)})
    candidate_hashes = make_candidate(world, plan, folder)
    graph, completed = train_candidate(folder / "candidate", folder, world.identity, plan)
    with np.load(folder / "candidate/discovery.npz", allow_pickle=False) as archive:
        normalizers = np.maximum(np.std(archive["observations"], axis=0), .05)
    _, exogenous = world.sample(
        seed=sample_seed(world.identity, "heldout_eval_exogenous_v1"),
        samples=plan["evaluation_rows"], return_exogenous=True)
    roles = ["mixed" if graph[:, j].any() else "control" for j in range(world.nodes)]
    arms = {arm: load_predictors(folder, graph, arm) for arm in ("mixed", "control")}
    hybrid = [arms[roles[j]][j] for j in range(world.nodes)]
    metrics = {arm: evaluate(world, graph, arms[arm], exogenous, normalizers)
               for arm in ("mixed", "control")}
    metrics["hybrid"] = evaluate(world, graph, hybrid, exogenous, normalizers)
    result = {
        "schema": "ncd.interventional-role-confirmation-unit.v1",
        "status": "computed-pending-independent-replay",
        "unit": list(unit), "world_id": world.identity,
        "plan_sha256": digest(plan_path), "candidate_sha256": candidate_hashes,
        "truth_world_sha256": digest(folder / "truth_only/world.json"),
        "training": completed, "inferred_graph": graph.astype(int).tolist(),
        "true_graph": np.asarray(world.graph, dtype=int).tolist(),
        "graph_exact": bool(np.array_equal(graph, world.graph)),
        "selected_arms": roles, "normalizers": normalizers.tolist(),
        "metrics": metrics, "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    write(folder / "result.json", result)
    write(folder / "unit_manifest.json", {
        "status": "computed-pending-independent-replay", "unit": list(unit),
        "world_id": world.identity, "plan_sha256": digest(plan_path),
        "result_sha256": digest(folder / "result.json"),
    })
    return {"unit": unit_name(unit), "status": result["status"],
            "graph_exact": result["graph_exact"],
            "hybrid_max_local_mse": metrics["hybrid"]["max_local_normalized_mse"],
            "hybrid_max_rollout_mse": metrics["hybrid"]["max_rollout_normalized_mse"]}


def run_stage(plan_path=PLAN, max_units=None):
    plan = preflight(plan_path)
    output = ROOT / plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "protocol.json"
    if frozen.exists() and digest(frozen) != digest(plan_path):
        raise ValueError("Existing run has a different protocol")
    if not frozen.exists():
        frozen.write_bytes(Path(plan_path).read_bytes())
    started, new = monotonic(), []
    for unit in declared_units(plan):
        folder = output / "units" / unit_name(unit)
        if folder.exists():
            manifest = read(folder / "unit_manifest.json")
            if (manifest.get("status") == "computed-pending-independent-replay"
                    and digest(folder / "result.json") == manifest["result_sha256"]):
                continue
            raise ValueError("Incomplete/changed unit requires audit: " + unit_name(unit))
        if max_units is not None and len(new) >= max_units:
            break
        remaining = plan["max_stage_seconds"] - (monotonic() - started)
        if remaining <= 0:
            break
        process = run_isolated(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
             "--unit", *map(str, unit), "--plan", str(plan_path)],
            ROOT, min(remaining, plan["max_unit_seconds"]), plan["memory_bytes"])
        if process["exit_code"] != 0 or process["resources"]["timeout"]:
            raise RuntimeError("Unit failed: " + unit_name(unit) + "\n" +
                               process["stderr"].decode("utf-8", errors="replace")[-3000:])
        new.append(json.loads(process["stdout"].decode("utf-8").splitlines()[-1]))
        used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if used > plan["artifact_budget_bytes"]:
            raise RuntimeError("Artifact budget exceeded")
    completed = sum((output / "units" / unit_name(u) / "result.json").exists()
                    for u in declared_units(plan))
    return {"status": "computed-pending-independent-replay", "new_units": new,
            "completed_units": completed, "declared_units": len(declared_units(plan)),
            "original_objective_achieved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--unit", nargs=4)
    args = parser.parse_args()
    if args.preflight:
        plan = preflight(args.plan)
        result = {"status": "frozen-ready", "units": len(declared_units(plan)),
                  "plan_sha256": digest(args.plan)}
    elif args.unit:
        seed, nodes, split, index = args.unit
        unit = (int(seed), int(nodes), split, int(index))
        if unit not in declared_units(preflight(args.plan)):
            raise ValueError("Unit not declared by protocol")
        result = run_unit(unit, args.plan)
    elif args.run:
        if args.max_units is not None and args.max_units <= 0:
            raise ValueError("max-units must be positive")
        result = run_stage(args.plan, args.max_units)
    else:
        parser.error("Specify --preflight, --run or --unit")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
