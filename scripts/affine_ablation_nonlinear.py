"""Matched-data affine diagnostic for the frozen nonlinear study."""
from pathlib import Path
import json

import numpy as np

from ncd.trained_nonlinear_realization import NonlinearTrainingConfig, _samples, ALPHAS
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs" / "trained_nonlinear_global_v1"
OUTPUT = ROOT / "validation" / "trained_nonlinear_affine_ablation.json"


def case(seed: int, d: int) -> dict:
    config = NonlinearTrainingConfig(seed=seed, state_dim=d)
    train_x, train_a, train_y, _ = _samples(config, 1, config.train_samples)
    val_x, val_a, val_y, _ = _samples(config, 2, config.selection_samples)
    test_x, test_a, test_y, _ = _samples(config, 3, config.test_samples)

    def design(states, actions, i):
        return np.column_stack((
            states[:, i], states[:, (i - 1) % d], actions, np.ones(len(states))))

    prediction = np.empty_like(test_y)
    chosen = []
    for i in range(d):
        train, val, test = (design(x, a, i) for x, a in
                            ((train_x, train_a), (val_x, val_a), (test_x, test_a)))
        alternatives = []
        for alpha in ALPHAS:
            penalty = alpha * np.diag((1, 1, 1, 1, 0))
            fitted = np.linalg.solve(train.T @ train + penalty, train.T @ train_y[:, i])
            val_rmse = float(np.sqrt(np.mean((val @ fitted - val_y[:, i]) ** 2)))
            alternatives.append((val_rmse, alpha, fitted))
        _, alpha, fitted = min(alternatives, key=lambda item: item[0])
        chosen.append(alpha)
        prediction[:, i] = test @ fitted
    error = prediction - test_y
    neural = read_json(STUDY / f"seed_{seed}" / f"d_{d}" / "training.json")["test"]
    affine_rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "seed": seed, "state_dim": d,
        "affine_test_rmse": affine_rmse,
        "affine_test_max_error": float(np.max(np.abs(error))),
        "nonlinear_test_rmse": neural["rmse"],
        "nonlinear_test_max_error": neural["max_error"],
        "rmse_ratio_affine_to_nonlinear": affine_rmse / neural["rmse"],
        "chosen_affine_alphas": chosen,
        "scope": "post hoc descriptive ablation, same train/selection/test data",
    }


def calculate() -> dict:
    return {"schema": "ncd.trained-nonlinear-affine-ablation.v1",
            "cases": [case(seed, d) for seed in (6101, 6102)
                      for d in (8, 32, 64, 128)]}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    expected = calculate()
    if args.verify:
        if read_json(OUTPUT) != expected:
            raise ValueError("Affine ablation replay mismatch")
    else:
        save_json(OUTPUT, expected)
    print(json.dumps({"cases": len(expected["cases"]),
                      "min_rmse_ratio": min(x["rmse_ratio_affine_to_nonlinear"]
                                            for x in expected["cases"])}))


if __name__ == "__main__":
    main()

