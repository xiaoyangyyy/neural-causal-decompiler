"""Independent full replay of fresh-seed, independently sampled do graph units."""
from fractions import Fraction as Q
from hashlib import sha256
import json
from math import comb
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ncd.active_intervention_graph import load_active_factorized_graph, padded_observational_features
from ncd.graph_model import dag_completion, decode_graph, graph_probabilities
from ncd.multiverse import generate_graph_worlds

PLAN = ROOT / "validation/independent_do_linear_confirmation_protocol_v1.json"
OUTPUT = ROOT / "validation/independent_do_linear_confirmation_verified_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    os.replace(temp, path)


def sample_seed(identity, name):
    return int.from_bytes(sha256((identity+":"+name).encode()).digest()[:8],
                          "little") % (2**63-1)


def expected_units(plan):
    return [(s,n,e,i) for s in plan["seeds"] for n in plan["nodes"]
            for e in plan["environments"] for i in plan["linear_indices"]]


def verify_unit(unit, plan, teacher):
    seed, n, env, index = unit
    name = f"seed_{seed}_n{n}_{env}_{index}"
    folder = ROOT / plan["output"] / "units" / name
    world = generate_graph_worlds(env, 10, n, seed=seed, samples=96)[index]
    if world.family != "linear_gaussian":
        raise ValueError("Unit is not linear Gaussian")
    truth = read(folder / "truth_only/world.json")
    if truth != world.metadata():
        raise ValueError("Truth evaluator world changed")
    result = read(folder / "result.json")
    if (result["unit"] != list(unit) or result["world_id"] != world.identity
            or result["protocol_sha256"] != digest(PLAN)
            or result["teacher_sha256"] != plan["graph_teacher_sha256"]
            or result["status"] != "computed-pending-independent-replay"
            or result["original_claim_closed"] is not False):
        raise ValueError("Unit identity or scope changed")
    candidate = folder / "candidate"
    files = {p.name: digest(p) for p in candidate.iterdir() if p.is_file()}
    if result["candidate_inputs_sha256"] != files or set(files) != {
            "discovery.npz", "do_groups.npz", "coverage.json"}:
        raise ValueError("Candidate input hashes changed")
    with np.load(candidate / "discovery.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"observations"}:
            raise ValueError("Discovery fields changed")
        discovery = archive["observations"]
    expected_discovery = world.sample(seed=sample_seed(world.identity, "discovery"), samples=96)
    if not np.array_equal(discovery, expected_discovery):
        raise ValueError("Discovery data changed")
    groups = {}
    coverage = []
    with np.load(candidate / "do_groups.npz", allow_pickle=False) as archive:
        expected_keys = {f"do_{i}_{'plus' if level == 1 else 'minus'}"
                         for i in range(n) for level in (-1, 1)}
        if set(archive.files) != expected_keys:
            raise ValueError("Independent do groups changed")
        base, extra = divmod(384, 2*n)
        for source in range(n):
            for level in (-1, 1):
                k = 2*source + int(level == 1)
                count = base + int(k < extra)
                val = max(6, count//4)
                fit = count-val
                name = f"do_{source}_{'plus' if level == 1 else 'minus'}"
                expected_parts = [world.sample(
                    seed=sample_seed(world.identity, split+"_"+name),
                    interventions={source: float(level)}, samples=rows)
                    for split, rows in (("fit", fit), ("validation", val))]
                expected = np.concatenate(expected_parts)
                actual = archive[name]
                if not np.array_equal(actual, expected):
                    raise ValueError("Independent do samples changed")
                groups[(source, level)] = actual
                coverage.append({"source": source, "level": level,
                                 "fit_rows": fit, "validation_rows": val,
                                 "total_rows": count})
    if read(candidate / "coverage.json") != coverage:
        raise ValueError("Do coverage changed")
    # Recompute the estimator without importing the candidate implementation.
    total = np.column_stack([(groups[(i,1)].mean(axis=0)-groups[(i,-1)].mean(axis=0))/2.
                             for i in range(n)])
    direct = np.eye(n)-np.linalg.inv(total)
    graph = (np.abs(direct.T) > plan["threshold"]).astype(int)
    np.fill_diagonal(graph, 0)
    remaining = set(range(n))
    while remaining:
        roots = {j for j in remaining if all(graph[i,j] == 0 for i in remaining)}
        if not roots:
            break
        remaining -= roots
    recovered = result["candidate"]
    if (recovered["schema"] != "ncd.independent-do-mean-graph-candidate.v1"
            or recovered["truth_graph_read"] is not False
            or recovered["exogenous_pairing_used"] is not False
            or recovered["finite_sample_graph_guarantee"] is not False
            or recovered["threshold"] != plan["threshold"]
            or recovered["coverage"] != [
                {"source": i, "plus_rows": len(groups[(i,1)]),
                 "minus_rows": len(groups[(i,-1)])} for i in range(n)]
            or recovered["graph_source_target"] != graph.tolist()
            or recovered["acyclic"] is not (not remaining)):
        raise ValueError("Candidate graph or provenance changed")
    np.testing.assert_allclose(recovered["total_effect"], total, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(recovered["direct_effect"], direct, rtol=1e-10, atol=1e-10)
    with torch.no_grad():
        probabilities = graph_probabilities(
            teacher, padded_observational_features(discovery)[None])[0]
    partial, _ = decode_graph(probabilities)
    teacher_graph, _ = dag_completion(partial)
    teacher_graph = teacher_graph.astype(int)
    truth_graph = np.asarray(world.graph, dtype=int)
    if (result["teacher_graph_source_target"] != teacher_graph.tolist()
            or result["truth_graph_source_target"] != truth_graph.tolist()
            or result["candidate_exact"] is not bool(np.array_equal(graph, truth_graph))
            or result["teacher_exact"] is not bool(np.array_equal(teacher_graph, truth_graph))
            or result["candidate_shd"] != int(np.count_nonzero(graph != truth_graph))
            or result["teacher_shd"] != int(np.count_nonzero(teacher_graph != truth_graph))):
        raise ValueError("Independent graph evaluation changed")
    return {"unit": list(unit), "world_id": world.identity,
            "result_sha256": digest(folder / "result.json"),
            "candidate_exact": result["candidate_exact"],
            "teacher_exact": result["teacher_exact"],
            "candidate_acyclic": recovered["acyclic"],
            "candidate_shd": result["candidate_shd"],
            "teacher_shd": result["teacher_shd"]}


def compute():
    plan = read(PLAN)
    if (plan["status"] != "frozen-before-confirmation"
            or plan["seeds"] != [8401,8402]
            or plan["threshold"] != 0.2
            or plan["relative_gate"] != {"candidate_min_exact": 30,
                                          "candidate_minus_teacher_min": 20,
                                          "each_node_size_min_exact": 10,
                                          "paired_sign_p_max": "1/100"}
            or digest(ROOT / plan["graph_teacher"]) != plan["graph_teacher_sha256"]):
        raise ValueError("Frozen plan or teacher changed")
    for relative, expected in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Source hash changed: "+relative)
    run_plan = ROOT / plan["output"] / "protocol.json"
    if digest(run_plan) != digest(PLAN):
        raise ValueError("Run protocol not frozen plan")
    torch.set_num_threads(1)
    teacher = load_active_factorized_graph(ROOT / plan["graph_teacher"])
    rows = [verify_unit(unit, plan, teacher) for unit in expected_units(plan)]
    if len(rows) != 60 or len({r["world_id"] for r in rows}) != 60:
        raise ValueError("Not all independent worlds replayed")
    candidate_only = sum(r["candidate_exact"] and not r["teacher_exact"] for r in rows)
    teacher_only = sum(r["teacher_exact"] and not r["candidate_exact"] for r in rows)
    discordant = candidate_only+teacher_only
    p = Q(sum(comb(discordant, k) for k in range(candidate_only, discordant+1)),
          2**discordant) if discordant else Q(1)
    by_nodes = [{"nodes": n, "worlds": 20,
                 "candidate_exact": sum(r["candidate_exact"] for r in rows if r["unit"][1] == n),
                 "teacher_exact": sum(r["teacher_exact"] for r in rows if r["unit"][1] == n),
                 "candidate_acyclic": sum(r["candidate_acyclic"] for r in rows if r["unit"][1] == n)}
                for n in (3,5,8)]
    candidate_count = sum(r["candidate_exact"] for r in rows)
    teacher_count = sum(r["teacher_exact"] for r in rows)
    gate = (candidate_count >= 30 and candidate_count-teacher_count >= 20
            and all(r["candidate_exact"] >= 10 for r in by_nodes)
            and p <= Q(1,100))
    return {
        "schema": "ncd.independent-do-linear-confirmation-verification.v1",
        "status": "verified-all-60-candidate-only",
        "protocol_sha256": digest(PLAN),
        "verifier_source_sha256": digest(__file__),
        "worlds": rows, "declared_worlds": 60, "verified_worlds": 60,
        "candidate_exact": candidate_count, "teacher_exact": teacher_count,
        "candidate_only_exact": candidate_only, "teacher_only_exact": teacher_only,
        "candidate_acyclic": sum(r["candidate_acyclic"] for r in rows),
        "by_nodes": by_nodes,
        "one_sided_paired_sign_p_exact": str(p),
        "predeclared_relative_gate_passed": gate,
        "statistical_test_null": "independent world-level discordance signs are conditionally fair; extra intervention access is allowed for candidate",
        "sign_test_alpha": "1/100",
        "candidate_has_more_intervention_data_than_teacher": True,
        "confidence_in_all_world_graph_recovery": False,
        "neural_to_program_graph_fidelity_proved": False,
        "original_claim_closed": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: verifier --write|--verify")
    result = compute()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Existing independent verification retained")
        write(OUTPUT, result)
    elif read(OUTPUT) != result:
        raise ValueError("Confirmation verification changed")
    print(json.dumps({"status": result["status"],
                      "candidate_exact": result["candidate_exact"],
                      "teacher_exact": result["teacher_exact"],
                      "relative_gate": result["predeclared_relative_gate_passed"],
                      "paired_sign_p": result["one_sided_paired_sign_p_exact"]}, sort_keys=True))