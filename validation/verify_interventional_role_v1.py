"""Independent read-only replay of one frozen role-confirmation world."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from ncd.active_intervention_graph import load_active_factorized_graph, padded_observational_features
from ncd.graph_model import dag_completion, decode_graph, graph_probabilities
from ncd.graphs import topological_order
from ncd.mechanisms import load_mechanism, neural_values
from ncd.multiverse import GraphWorld, generate_graph_worlds


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def seed(world_id, label):
    data = sha256((world_id + ":" + label).encode()).digest()
    return int.from_bytes(data[:8], "little") % (2**63 - 1)


def assert_same(expected, observed, path="root"):
    if type(expected) is not type(observed):
        raise ValueError(path + ": type changed")
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            raise ValueError(path + ": keys changed")
        for key in expected:
            assert_same(expected[key], observed[key], path + "." + str(key))
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            raise ValueError(path + ": length changed")
        for i, (left, right) in enumerate(zip(expected, observed)):
            assert_same(left, right, path + f"[{i}]")
    elif isinstance(expected, float):
        # Allow observed CPU reduction jitter; raw data and checkpoints remain hash-bound.
        if not math.isclose(expected, observed, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(path + f": float changed {expected!r} versus {observed!r}")
    elif expected != observed:
        raise ValueError(path + ": value changed")


def independent_metrics(world, graph, models, noise, scales):
    world_scales = np.asarray(world.scales or (1.,) * world.nodes)
    rows, rollout, average = [], [], {}
    interventions = [{}]
    for j in range(world.nodes):
        interventions += [{j: -1.}, {j: 1.}]
    interventions += [{0: 1., 1: -1.}, {0: -1., 2: 1.}]
    for mask in interventions:
        observed = world.sample(interventions=mask, exogenous=noise, samples=len(noise))
        simulated = np.zeros_like(observed)
        for j in topological_order(graph):
            if j in mask:
                simulated[:, j] = mask[j]
            else:
                simulated[:, j] = models[j](simulated) + world_scales[j] * noise[:, j]
        average[tuple(sorted(mask.items()))] = (
            simulated.mean(axis=0), observed.mean(axis=0))
        for j in range(world.nodes):
            if j in mask:
                continue
            key = {"mask": {str(k): v for k, v in mask.items()}, "node": j}
            true_mechanism = observed[:, j] - world_scales[j] * noise[:, j]
            delta = (models[j](observed) - true_mechanism) / scales[j]
            rows.append({**key, "normalized_mse": float(np.dot(delta, delta) / len(delta)),
                         "normalized_mae": float(np.abs(delta).mean())})
            delta = (simulated[:, j] - observed[:, j]) / scales[j]
            rollout.append({**key, "normalized_mse": float(np.dot(delta, delta) / len(delta)),
                            "normalized_mae": float(np.abs(delta).mean())})
    effects = []
    for source in range(world.nodes):
        plus_model, plus_true = average[((source, 1.),)]
        minus_model, minus_true = average[((source, -1.),)]
        for target in range(world.nodes):
            if target == source:
                continue
            effect = (plus_model[target] - minus_model[target]
                      - plus_true[target] + minus_true[target])
            effects.append({"source": source, "target": target,
                            "normalized_abs_error": float(abs(effect) / scales[target])})
    return {
        "local": rows, "rollout_with_oracle_noise": rollout,
        "max_local_normalized_mse": max(row["normalized_mse"] for row in rows),
        "max_rollout_normalized_mse": max(row["normalized_mse"] for row in rollout),
        "max_rollout_paired_contrast_error": max(row["normalized_abs_error"] for row in effects),
        "contrasts": effects, "independent_exogenous_draws": len(noise),
        "noise_source": "true_exogenous_oracle_diagnostic_only",
    }


def verify(unit_folder, protocol=ROOT / "validation/interventional_role_confirmation_protocol_v1.json"):
    torch.set_num_threads(1)
    plan = read(protocol)
    folder = Path(unit_folder).resolve()
    base = (ROOT / plan["output"] / "units").resolve()
    if folder.parent != base:
        raise ValueError("Unit outside frozen output")
    if plan["status"] != "frozen-before-confirmation":
        raise ValueError("Protocol not frozen")
    if digest(ROOT / plan["graph_teacher"]) != plan["graph_teacher_sha256"]:
        raise ValueError("Graph teacher changed")
    for relative, expected in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Source changed: " + relative)
    record = read(folder / "result.json")
    manifest = read(folder / "unit_manifest.json")
    if (record["schema"] != "ncd.interventional-role-confirmation-unit.v1"
            or record["status"] != "computed-pending-independent-replay"
            or record["plan_sha256"] != digest(protocol)
            or manifest["result_sha256"] != digest(folder / "result.json")
            or record["original_claim_closed"] is not False
            or record["original_objective_achieved"] is not False):
        raise ValueError("Untrusted unit result")
    seed_value, nodes, split, index = record["unit"]
    if (seed_value not in plan["seeds"] or nodes not in plan["nodes"]
            or split not in plan["environments"] or index not in range(plan["worlds_per_cell"])
            or folder.name != f"seed_{seed_value}_n{nodes}_{split}_{index}"):
        raise ValueError("Unit not declared")
    world = generate_graph_worlds(split, plan["worlds_per_cell"], nodes,
                                  seed=seed_value, samples=plan["discovery_rows"])[index]
    if world.identity != record["world_id"]:
        raise ValueError("Generated world identity mismatch")
    truth_path = folder / "truth_only/world.json"
    if digest(truth_path) != record["truth_world_sha256"]:
        raise ValueError("Truth metadata hash mismatch")
    assert_same(json.loads(json.dumps(world.metadata())), read(truth_path), "truth")
    candidate = folder / "candidate"
    files = {path.name for path in candidate.iterdir() if path.is_file()}
    if files != set(record["candidate_sha256"]):
        raise ValueError("Candidate file set changed")
    for filename, expected in record["candidate_sha256"].items():
        if digest(candidate / filename) != expected:
            raise ValueError("Candidate file changed: " + filename)
    discovery = world.sample(seed=seed(world.identity, "discovery"), samples=96)
    with np.load(candidate / "discovery.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"observations"} or not np.array_equal(
                archive["observations"], discovery):
            raise ValueError("Discovery observations changed")
    teacher = load_active_factorized_graph(ROOT / plan["graph_teacher"])
    probabilities = graph_probabilities(
        teacher, padded_observational_features(discovery)[None])[0]
    partial, _ = decode_graph(probabilities)
    inferred, _ = dag_completion(partial)
    graph_record = read(candidate / "graph_prediction.json")
    if (graph_record["oracle_graph_used"] is not False
            or not np.array_equal(graph_record["inferred_graph"], inferred)
            or not np.array_equal(record["inferred_graph"], inferred)):
        raise ValueError("Inferred graph changed")
    batch_doc = read(candidate / "batches.json")
    if (batch_doc["world_id"] != world.identity
            or batch_doc["oracle_graph_used"] is not False
            or batch_doc["true_equations_used_for_training"] is not False):
        raise ValueError("Training batch provenance changed")
    expected_batches = [("observation", {}, 96, 32)]
    base_rows, extra_rows = divmod(384, 2 * nodes)
    for k in range(2 * nodes):
        target, side = divmod(k, 2)
        value = -1. if side == 0 else 1.
        count = base_rows + int(k < extra_rows)
        validation_count = max(6, count // 4)
        expected_batches.append((
            f"do_{target}_{'minus' if side == 0 else 'plus'}",
            {str(target): value}, count - validation_count, validation_count))
    expected_metadata = []
    for kind, mask, fit_count, validation_count in expected_batches:
        expected_metadata.extend([
            ("fit_" + kind, "fit", mask, fit_count),
            ("validation_" + kind, "validation", mask, validation_count),
        ])
    actual_metadata = [(b["source"], b["split"], b["interventions"], b["rows"])
                       for b in batch_doc["batches"]]
    if actual_metadata != expected_metadata:
        raise ValueError("Frozen mechanism batch layout changed")
    with np.load(candidate / "mechanism_observations.npz", allow_pickle=False) as archive:
        if set(archive.files) != {b["source"] for b in batch_doc["batches"]}:
            raise ValueError("Mechanism batch keys changed")
        if sum(b["rows"] for b in batch_doc["batches"]) != 512:
            raise ValueError("Mechanism observation budget changed")
        for batch in batch_doc["batches"]:
            mask = {int(k): float(v) for k, v in batch["interventions"].items()}
            regenerated = world.sample(
                seed=seed(world.identity, batch["source"]),
                interventions=mask, samples=batch["rows"])
            if not np.array_equal(archive[batch["source"]], regenerated):
                raise ValueError("Mechanism observations changed")
            if batch["row_ids"] != [
                    f"{world.identity}:{batch['source']}:{i}" for i in range(batch["rows"])]:
                raise ValueError("Mechanism row IDs changed")
    for split_name, count in (("fit", 384), ("validation", 128)):
        source = "control_" + split_name
        with np.load(candidate / (source + ".npz"), allow_pickle=False) as archive:
            regenerated = world.sample(seed=seed(world.identity, source), samples=count)
            if set(archive.files) != {"observations"} or not np.array_equal(
                    archive["observations"], regenerated):
                raise ValueError("Control observations changed")
    role = ["mixed" if inferred[:, j].any() else "control" for j in range(nodes)]
    if (record["selected_arms"] != role
            or record["true_graph"] != np.asarray(world.graph, dtype=int).tolist()
            or record["graph_exact"] is not bool(np.array_equal(inferred, world.graph))):
        raise ValueError("Graph or role result changed")
    models = {}
    for arm in ("mixed", "control"):
        models[arm] = []
        if len(record["training"][arm]) != nodes:
            raise ValueError("Missing model checkpoint")
        for j, bound in enumerate(record["training"][arm]):
            checkpoint = folder / "trained" / arm / f"mechanism_{j}.pt"
            saved_record = checkpoint.with_suffix(".json")
            parents = tuple(map(int, np.flatnonzero(inferred[:, j])))
            training_record = read(saved_record)
            expected_seed = seed(world.identity, f"train_{arm}_{j}") % (2**31 - 1)
            if (bound["node"] != j or bound["parents"] != list(parents)
                    or bound["checkpoint_sha256"] != digest(checkpoint)
                    or bound["record_sha256"] != digest(saved_record)
                    or training_record["checkpoint_sha256"] != digest(checkpoint)
                    or training_record["target"] != j
                    or training_record["inferred_parents"] != list(parents)
                    or training_record["seed"] != expected_seed
                    or training_record["epochs"] != plan["epochs"]
                    or training_record["true_graph_used"] is not False
                    or training_record["true_equations_used"] is not False):
                raise ValueError("Checkpoint provenance changed")
            if arm == "control":
                expected_fit, expected_val = 384, 128
            else:
                excluded_fit = sum(fit_count for _, mask, fit_count, _ in expected_batches
                                   if str(j) in mask)
                excluded_val = sum(val_count for _, mask, _, val_count in expected_batches
                                   if str(j) in mask)
                expected_fit = sum(fit_count for _, _, fit_count, _ in expected_batches) - excluded_fit
                expected_val = sum(val_count for _, _, _, val_count in expected_batches) - excluded_val
            if (training_record["fit_rows"] != expected_fit
                    or training_record["validation_rows"] != expected_val):
                raise ValueError("Executed-node training coverage changed")
            model = load_mechanism(checkpoint)
            if tuple(model.parents) != parents:
                raise ValueError("Checkpoint parent set changed")
            models[arm].append(lambda values, m=model: neural_values(m, values))
    with np.load(candidate / "discovery.npz", allow_pickle=False) as archive:
        normalizers = np.maximum(archive["observations"].std(axis=0), .05)
    assert_same(normalizers.tolist(), record["normalizers"], "normalizers")
    _, noise = world.sample(seed=seed(world.identity, "heldout_eval_exogenous_v1"),
                            samples=plan["evaluation_rows"], return_exogenous=True)
    metrics = {arm: independent_metrics(world, inferred, models[arm], noise, normalizers)
               for arm in ("mixed", "control")}
    hybrid = [models[role[j]][j] for j in range(nodes)]
    metrics["hybrid"] = independent_metrics(world, inferred, hybrid, noise, normalizers)
    assert_same(metrics, record["metrics"], "metrics")
    for arm in ("mixed", "control", "hybrid"):
        if ((metrics[arm]["max_local_normalized_mse"] <= .01) !=
                (record["metrics"][arm]["max_local_normalized_mse"] <= .01)):
            raise ValueError("metrics." + arm + ": local gate classification changed")
    return {
        "schema": "ncd.interventional-role-confirmation-verification.v1",
        "status": "verified-one-independent-world",
        "unit": record["unit"], "world_id": world.identity,
        "plan_sha256": digest(protocol), "result_sha256": digest(folder / "result.json"),
        "verifier_sha256": digest(__file__),
        "graph_exact": record["graph_exact"],
        "hybrid_meets_local_0_01": metrics["hybrid"]["max_local_normalized_mse"] <= .01,
        "hybrid_max_local_normalized_mse": metrics["hybrid"]["max_local_normalized_mse"],
        "hybrid_max_rollout_normalized_mse": metrics["hybrid"]["max_rollout_normalized_mse"],
        "oracle_noise_used_for_rollout": True,
        "original_claim_closed": False, "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verifier UNIT_DIRECTORY")
    print(json.dumps(verify(ROOT / sys.argv[1]), sort_keys=True))
