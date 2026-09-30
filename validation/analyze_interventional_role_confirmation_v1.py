"""World-level analysis of the independently verified 300-world role experiment."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "validation/interventional_role_confirmation_verified_v1.json"
OUTPUT = ROOT / "validation/interventional_role_confirmation_analysis_v1.json"
THRESHOLD = .01
ALPHA = .01
MULTIPLICITY = 4


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def simultaneous_hoeffding(successes, worlds, alpha=ALPHA, comparisons=MULTIPLICITY):
    """Two-sided Bonferroni/Hoeffding interval for independent world indicators."""
    if (type(successes) is not int or type(worlds) is not int
            or not 0 <= successes <= worlds or worlds < 1
            or not 0 < alpha < 1 or comparisons < 1):
        raise ValueError("Invalid world-level uncertainty inputs")
    proportion = successes / worlds
    radius = math.sqrt(math.log(2 * comparisons / alpha) / (2 * worlds))
    return {
        "successes": successes, "worlds": worlds,
        "observed_rate": proportion,
        "lower": max(0., proportion - radius),
        "upper": min(1., proportion + radius),
        "method": "Hoeffding two-sided with Bonferroni correction",
        "confidence_joint": 1 - alpha,
        "comparisons": comparisons,
        "unit_of_independence": "world",
        "assumption": "independent world draws in the frozen stratified generator; no within-world intervention pair is counted separately",
    }


def compute():
    summary = read(SUMMARY)
    if (summary["status"] != "verified-all-300"
            or summary["verified_worlds"] != 300
            or summary["declared_worlds"] != 300
            or len(summary["worlds"]) != 300
            or summary["original_claim_closed"] is not False):
        raise ValueError("The entire independently verified cohort is required")
    worlds = summary["worlds"]
    if len({tuple(row["unit"]) for row in worlds}) != 300:
        raise ValueError("Repeated unit in verified summary")
    counts = {
        "hybrid_local_0_01": sum(row["hybrid_max_local_mse"] <= THRESHOLD
                                 for row in worlds),
        "mixed_local_0_01": sum(row["mixed_max_local_mse"] <= THRESHOLD
                                for row in worlds),
        "control_local_0_01": sum(row["control_max_local_mse"] <= THRESHOLD
                                  for row in worlds),
        "graph_exact": sum(row["graph_exact"] for row in worlds),
    }
    if (counts["hybrid_local_0_01"] != summary["hybrid_local_gate_passes"]
            or counts["graph_exact"] != summary["graph_exact_worlds"]):
        raise ValueError("Verified summary counts changed")
    exact = [row for row in worlds if row["graph_exact"]]
    failures_with_exact_graph = sum(
        row["hybrid_max_local_mse"] > THRESHOLD for row in exact)
    by_nodes = []
    for nodes in (3, 5, 8):
        selected = [row for row in worlds if row["unit"][1] == nodes]
        by_nodes.append({
            "nodes": nodes, "worlds": len(selected),
            "graph_exact": sum(row["graph_exact"] for row in selected),
            "hybrid_local_passes": sum(
                row["hybrid_max_local_mse"] <= THRESHOLD for row in selected),
            "hybrid_max_local_mse": max(
                row["hybrid_max_local_mse"] for row in selected),
            "hybrid_max_rollout_mse_with_oracle_noise": max(
                row["hybrid_max_rollout_mse_with_oracle_noise"] for row in selected),
        })
    return {
        "schema": "ncd.interventional-role-confirmation-analysis.v1",
        "status": "verified-world-level-analysis",
        "summary_sha256": digest(SUMMARY),
        "analysis_source_sha256": digest(__file__),
        "worlds": 300,
        "threshold": THRESHOLD,
        "observed_counts": counts,
        "simultaneous_99_percent_intervals": {
            key: simultaneous_hoeffding(value, 300)
            for key, value in counts.items()},
        "graph_exact_worlds": len(exact),
        "hybrid_local_failures_with_exact_graph": failures_with_exact_graph,
        "by_nodes": by_nodes,
        "oracle_noise_used_for_rollout": True,
        "recovered_noise_law_validated": False,
        "analysis_fixed_after_initial_confirmation_worlds_seen": True,
        "confirmatory_99_percent_claim_established": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: analysis --write|--verify")
    result = compute()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Existing analysis retained")
        OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    elif read(OUTPUT) != result:
        raise ValueError("World-level analysis changed")
    print(json.dumps({
        "status": result["status"], "worlds": result["worlds"],
        "hybrid_local_passes": result["observed_counts"]["hybrid_local_0_01"],
        "graph_exact_worlds": result["graph_exact_worlds"],
        "original_claim_closed": False,
    }, sort_keys=True))
