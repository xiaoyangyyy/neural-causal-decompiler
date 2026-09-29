"""Matched-data fixed-dictionary comparator for trainable local ReLU study."""
from pathlib import Path
import json

from ncd.trained_nonlinear_realization import NonlinearTrainingConfig, train_nonlinear
from ncd.io import read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs" / "learned_local_global_v1"
OUTPUT = ROOT / "validation" / "learned_local_dictionary_comparator.json"


def compute() -> dict:
    cases = []
    for seed in (7201, 7202):
        for d in (8, 32, 64, 128):
            _, fixed = train_nonlinear(NonlinearTrainingConfig(seed=seed, state_dim=d))
            learned = read_json(STUDY / f"seed_{seed}" / f"d_{d}" / "training.json")
            cases.append({
                "seed": seed, "state_dim": d,
                "fixed_dictionary_test_rmse": fixed["test"]["rmse"],
                "learned_hidden_test_rmse": learned["test"]["rmse"],
                "ratio_fixed_to_learned": (
                    fixed["test"]["rmse"] / learned["test"]["rmse"]),
                "scope": "post hoc matched-data comparison, not a frozen success gate",
            })
    return {"schema": "ncd.learned-local-dictionary-comparator.v1", "cases": cases}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.verify:
        if read_json(OUTPUT) != result:
            raise ValueError("Matched-data dictionary comparator replay mismatch")
    else:
        save_json(OUTPUT, result)
    print(json.dumps({
        "cases": len(result["cases"]),
        "min_ratio": min(x["ratio_fixed_to_learned"] for x in result["cases"]),
        "max_ratio": max(x["ratio_fixed_to_learned"] for x in result["cases"]),
    }))


if __name__ == "__main__":
    main()

