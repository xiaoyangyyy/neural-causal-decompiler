"""Evaluate the frozen 0.55 active-intervention graph decoder."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ncd.graph_model import decode_graph, decode_graph_threshold, graph_metrics

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "runs/active_intervention_seed4793",
    ROOT / "runs/active_intervention_seed4794",
]
ENVS = ("test_id", "test_function", "test_noise", "test_scale", "test_intervention")
NODES = (3, 5, 8)
THRESHOLD = 0.55


def targets(labels: np.ndarray) -> np.ndarray:
    count, n, _ = labels.shape
    result = np.zeros((count, n, n), dtype=bool)
    for k in range(count):
        for i in range(n):
            for j in range(i + 1, n):
                cls = int(labels[k, i, j])
                if cls == 1:
                    result[k, i, j] = True
                elif cls == 2:
                    result[k, j, i] = True
                elif cls == 3:
                    result[k, i, j] = result[k, j, i] = True
    return result


def main() -> None:
    runs = []
    for root in RUNS:
        cells = {}
        for env in ENVS:
            for n in NODES:
                key = f"n{n}_{env}"
                with np.load(root / "datasets" / key / "features.npz") as archive:
                    truth = targets(archive["labels"])
                with np.load(
                    root / "evaluations" / "active_intervention" / key / "predictions.npz"
                ) as archive:
                    probabilities = archive["probabilities"]
                    saved = archive["graphs"]
                control = np.stack([decode_graph(p)[0] for p in probabilities])
                np.testing.assert_array_equal(control, saved)
                candidate = np.stack(
                    [decode_graph_threshold(p, THRESHOLD)[0] for p in probabilities]
                )
                cells[key] = {
                    "control": graph_metrics(truth, control, control),
                    "candidate": graph_metrics(truth, candidate, candidate),
                }

        aggregate = {}
        for mode in ("control", "candidate"):
            aggregate[mode] = {
                metric: float(np.mean([cell[mode][metric] for cell in cells.values()]))
                for metric in (
                    "exact_graph_accuracy",
                    "mean_pair_shd",
                    "skeleton_f1",
                    "directed_target_accuracy",
                )
            }
        runs.append(
            {
                "seed": json.loads((root / "config.json").read_text())["seed"],
                "source": root.relative_to(ROOT).as_posix(),
                "threshold": THRESHOLD,
                "cells": cells,
                "aggregate": aggregate,
                "replayed": True,
            }
        )

    def pooled(mode: str, metric: str) -> float:
        return float(np.mean([run["aggregate"][mode][metric] for run in runs]))

    exact_delta = pooled("candidate", "exact_graph_accuracy") - pooled(
        "control", "exact_graph_accuracy"
    )
    direction_delta = pooled("candidate", "directed_target_accuracy") - pooled(
        "control", "directed_target_accuracy"
    )
    criteria = {
        "complete_probability_and_decode_replay": all(r["replayed"] for r in runs),
        "exact_higher_each_seed": all(
            r["aggregate"]["candidate"]["exact_graph_accuracy"]
            > r["aggregate"]["control"]["exact_graph_accuracy"]
            for r in runs
        ),
        "pooled_exact_gain_at_least_0_5pp": exact_delta >= 0.005,
        "shd_lower_each_seed": all(
            r["aggregate"]["candidate"]["mean_pair_shd"]
            < r["aggregate"]["control"]["mean_pair_shd"]
            for r in runs
        ),
        "skeleton_no_lower_each_seed": all(
            r["aggregate"]["candidate"]["skeleton_f1"]
            >= r["aggregate"]["control"]["skeleton_f1"]
            for r in runs
        ),
        "pooled_direction_decline_no_more_than_2pp": direction_delta >= -0.02,
        "full_15_cell_coverage": all(len(r["cells"]) == 15 for r in runs),
    }
    result = {
        "protocol": "docs/CALIBRATED_ACTIVE_DECODER_PROTOCOL.md",
        "runs": runs,
        "pooled": {
            "control_exact": pooled("control", "exact_graph_accuracy"),
            "candidate_exact": pooled("candidate", "exact_graph_accuracy"),
            "exact_delta": exact_delta,
            "control_shd": pooled("control", "mean_pair_shd"),
            "candidate_shd": pooled("candidate", "mean_pair_shd"),
            "control_skeleton_f1": pooled("control", "skeleton_f1"),
            "candidate_skeleton_f1": pooled("candidate", "skeleton_f1"),
            "control_directed_accuracy": pooled("control", "directed_target_accuracy"),
            "candidate_directed_accuracy": pooled(
                "candidate", "directed_target_accuracy"
            ),
            "direction_delta": direction_delta,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "scope": "fixed-threshold decoding of frozen active-intervention graph probabilities",
        "not_claimed": [
            "purely observational identifiability",
            "historical-teacher decompilation",
            "general causal discovery",
        ],
    }
    output = ROOT / "validation/calibrated_active_decoder_acceptance.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pooled": result["pooled"], "criteria": criteria, "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
