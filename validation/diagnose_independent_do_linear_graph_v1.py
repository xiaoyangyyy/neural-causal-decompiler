"""Posthoc audit of independent two-level do means on archived linear worlds.

The estimator sees only candidate intervention batches. Truth metadata enters
only after its total-effect and graph candidates have been fixed.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ncd.multiverse import GraphWorld

BASE = ROOT / "runs/interventional_role_confirmation_v1/units"
SUMMARY = ROOT / "validation/interventional_role_confirmation_verified_v1.json"
OUTPUT = ROOT / "validation/independent_do_linear_graph_diagnostic_v1.json"
THRESHOLDS = (0.05, 0.1, 0.2)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidate_total(folder, nodes):
    batches = read(folder / "candidate/batches.json")["batches"]
    if not all(b["split"] in ("fit", "validation") for b in batches):
        raise ValueError("Unknown split")
    total = np.zeros((nodes, nodes))
    coverage = []
    with np.load(folder / "candidate/mechanism_observations.npz", allow_pickle=False) as archive:
        for source in range(nodes):
            means = {}
            counts = {}
            for level in (-1., 1.):
                selected = [b for b in batches if b["interventions"] == {str(source): level}]
                if len(selected) != 2 or {b["split"] for b in selected} != {"fit", "validation"}:
                    raise ValueError("Missing independent do group")
                rows = np.concatenate([archive[b["source"]] for b in selected])
                if not np.allclose(rows[:, source], level, rtol=0, atol=1e-12):
                    raise ValueError("Do coordinate changed")
                means[level] = rows.mean(axis=0)
                counts[level] = len(rows)
            total[:, source] = (means[1.] - means[-1.]) / 2.
            coverage.append({"source": source, "plus_rows": counts[1.],
                             "minus_rows": counts[-1.]})
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite total-effect estimate")
    direct = np.eye(nodes) - np.linalg.inv(total)
    return total, direct, coverage


def compute():
    summary = read(SUMMARY)
    if summary["status"] != "verified-all-300" or summary["verified_worlds"] != 300:
        raise ValueError("Full verified source cohort required")
    rows = []
    for source in summary["worlds"]:
        unit = source["unit"]
        folder = BASE / f"seed_{unit[0]}_n{unit[1]}_{unit[2]}_{unit[3]}"
        if digest(folder / "result.json") != source["result_sha256"]:
            raise ValueError("Source world result changed")
        world = GraphWorld.from_dict(read(folder / "truth_only/world.json"))
        if world.family != "linear_gaussian":
            continue
        n = world.nodes
        total, direct, coverage = candidate_total(folder, n)
        scales = np.asarray(world.scales or (1.,)*n)
        truth_direct = np.zeros((n,n))
        for target, terms in enumerate(world.equations):
            for term in terms:
                if term.operator != "linear":
                    raise ValueError("Linear family contains nonlinear term")
                parent = term.parents[0]
                truth_direct[target,parent] += term.coefficient*scales[target]/scales[parent]
        truth_total = np.linalg.inv(np.eye(n)-truth_direct)
        if not np.array_equal((truth_direct.T != 0).astype(int), np.asarray(world.graph)):
            raise ValueError("Truth metadata graph changed")
        candidates = {}
        for threshold in THRESHOLDS:
            graph = (np.abs(direct.T) > threshold).astype(int)
            np.fill_diagonal(graph, 0)
            candidates[str(threshold)] = {
                "exact": bool(np.array_equal(graph, world.graph)),
                "shd": int(np.count_nonzero(graph != world.graph)),
            }
        rows.append({
            "unit": unit, "world_id": world.identity,
            "result_sha256": source["result_sha256"],
            "graph_teacher_exact": source["graph_exact"],
            "do_group_rows": coverage,
            "max_total_effect_abs_error": float(np.max(np.abs(total-truth_total))),
            "max_direct_effect_abs_error": float(np.max(np.abs(direct-truth_direct))),
            "estimated_total_condition_inf": float(np.linalg.cond(total, np.inf)),
            "threshold_candidates": candidates,
        })
    if len(rows) != 60:
        raise ValueError("Expected exactly 60 linear-Gaussian worlds")
    by_nodes = []
    for n in (3,5,8):
        selected = [r for r in rows if r["unit"][1] == n]
        by_nodes.append({"nodes": n, "worlds": len(selected),
                         "teacher_exact": sum(r["graph_teacher_exact"] for r in selected),
                         "candidate_exact": {str(t): sum(r["threshold_candidates"][str(t)]["exact"]
                                                       for r in selected) for t in THRESHOLDS}})
    return {
        "schema": "ncd.independent-do-linear-graph-diagnostic.v1",
        "status": "posthoc-archived-independent-do-diagnostic",
        "source_summary_sha256": digest(SUMMARY),
        "source_program_sha256": digest(__file__),
        "linear_gaussian_worlds": 60,
        "thresholds_exploratory_after_source_results_seen": list(THRESHOLDS),
        "by_nodes": by_nodes, "worlds": rows,
        "mean_total_effect_max_abs_error": float(np.mean([r["max_total_effect_abs_error"] for r in rows])),
        "max_total_effect_abs_error": max(r["max_total_effect_abs_error"] for r in rows),
        "oracle_truth_used_by_candidate": False,
        "independent_intervention_samples": True,
        "paired_exogenous_oracle_used": False,
        "confirmatory_accuracy_claim_established": False,
        "original_claim_closed": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: diagnostic --write|--verify")
    result = compute()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Existing posthoc diagnostic retained")
        OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    elif read(OUTPUT) != result:
        raise ValueError("Posthoc diagnostic changed")
    print(json.dumps({"status": result["status"], "by_nodes": result["by_nodes"],
                      "mean_total_effect_max_abs_error": result["mean_total_effect_max_abs_error"]}, sort_keys=True))