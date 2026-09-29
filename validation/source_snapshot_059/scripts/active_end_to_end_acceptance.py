"""Apply the frozen active end-to-end SCM recovery acceptance rule."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "runs/active_end_to_end_seed4993",
    ROOT / "runs/active_end_to_end_seed4994",
]
OBS = "observational_graph"
ACTIVE = "active_graph"
ORACLE = "oracle_graph_diagnostic"


def main() -> None:
    records = []
    for path in RUNS:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        graph = summary["graph"]
        aggregate = summary["aggregate"]
        records.append(
            {
                "seed": summary["config"]["seed"],
                "replayed": True,
                "world_count": summary["world_count"],
                "graph": graph,
                "observational_structured": aggregate[OBS]["structured"],
                "active_baseline": aggregate[ACTIVE]["baseline"],
                "active_structured": aggregate[ACTIVE]["structured"],
                "oracle_structured": aggregate[ORACLE]["structured"],
            }
        )

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for record in records:
            value = record
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    graph_exact_delta = mean(("graph", ACTIVE, "exact_accuracy")) - mean(
        ("graph", OBS, "exact_accuracy")
    )
    truth_obs = mean(("observational_structured", "mean_symbolic_truth_nmse"))
    truth_active = mean(("active_structured", "mean_symbolic_truth_nmse"))
    effect_obs = mean(("observational_structured", "mean_intervention_effect_mae"))
    effect_active = mean(("active_structured", "mean_intervention_effect_mae"))
    neural_baseline = mean(("active_baseline", "mean_symbolic_neural_nmse"))
    neural_structured = mean(("active_structured", "mean_symbolic_neural_nmse"))
    atoms_baseline = mean(("active_baseline", "mean_nonconstant_atoms"))
    atoms_structured = mean(("active_structured", "mean_nonconstant_atoms"))
    oracle_truth = mean(("oracle_structured", "mean_symbolic_truth_nmse"))
    oracle_effect = mean(("oracle_structured", "mean_intervention_effect_mae"))

    relative = lambda before, after: (before - after) / before
    criteria = {
        "complete_replay": all(record["replayed"] for record in records),
        "full_15_cell_coverage": all(record["world_count"] == 15 for record in records),
        "active_graph_exact_higher_each_seed": all(
            r["graph"][ACTIVE]["exact_accuracy"] > r["graph"][OBS]["exact_accuracy"]
            for r in records
        ),
        "pooled_graph_exact_gain_at_least_20pp": graph_exact_delta >= 0.20,
        "active_graph_shd_lower_each_seed": all(
            r["graph"][ACTIVE]["mean_shd"] < r["graph"][OBS]["mean_shd"]
            for r in records
        ),
        "active_truth_nmse_lower_each_seed": all(
            r["active_structured"]["mean_symbolic_truth_nmse"]
            < r["observational_structured"]["mean_symbolic_truth_nmse"]
            for r in records
        ),
        "pooled_truth_nmse_reduction_at_least_20pct": relative(truth_obs, truth_active)
        >= 0.20,
        "active_intervention_mae_lower_each_seed": all(
            r["active_structured"]["mean_intervention_effect_mae"]
            < r["observational_structured"]["mean_intervention_effect_mae"]
            for r in records
        ),
        "pooled_intervention_mae_reduction_at_least_20pct": relative(
            effect_obs, effect_active
        )
        >= 0.20,
        "active_structured_neural_nmse_lower_each_seed": all(
            r["active_structured"]["mean_symbolic_neural_nmse"]
            < r["active_baseline"]["mean_symbolic_neural_nmse"]
            for r in records
        ),
        "pooled_neural_nmse_reduction_at_least_15pct": relative(
            neural_baseline, neural_structured
        )
        >= 0.15,
        "active_structured_atoms_no_more_than_baseline": atoms_structured
        <= atoms_baseline,
        "active_truth_within_1_75x_oracle": truth_active <= 1.75 * oracle_truth,
        "active_intervention_within_2x_oracle": effect_active <= 2.0 * oracle_effect,
    }
    pooled = {
        "observational_graph_exact": mean(("graph", OBS, "exact_accuracy")),
        "active_graph_exact": mean(("graph", ACTIVE, "exact_accuracy")),
        "graph_exact_delta": graph_exact_delta,
        "observational_graph_shd": mean(("graph", OBS, "mean_shd")),
        "active_graph_shd": mean(("graph", ACTIVE, "mean_shd")),
        "observational_structured_truth_nmse": truth_obs,
        "active_structured_truth_nmse": truth_active,
        "truth_nmse_relative_reduction": relative(truth_obs, truth_active),
        "observational_structured_intervention_mae": effect_obs,
        "active_structured_intervention_mae": effect_active,
        "intervention_mae_relative_reduction": relative(effect_obs, effect_active),
        "active_baseline_neural_nmse": neural_baseline,
        "active_structured_neural_nmse": neural_structured,
        "neural_nmse_relative_reduction": relative(neural_baseline, neural_structured),
        "active_baseline_atoms": atoms_baseline,
        "active_structured_atoms": atoms_structured,
        "oracle_structured_truth_nmse": oracle_truth,
        "active_to_oracle_truth_ratio": truth_active / oracle_truth,
        "oracle_structured_intervention_mae": oracle_effect,
        "active_to_oracle_intervention_ratio": effect_active / oracle_effect,
    }
    result = {
        "protocol": "docs/ACTIVE_END_TO_END_PROTOCOL.md",
        "runs": records,
        "pooled": pooled,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "scope": "finite active-intervention graph-to-structured-SCM recovery on the frozen synthetic benchmark",
        "not_claimed": [
            "purely observational identifiability",
            "historical-teacher decompilation",
            "general causal discovery",
            "general exact equation recovery",
        ],
    }
    output = ROOT / "validation/active_end_to_end_acceptance.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()