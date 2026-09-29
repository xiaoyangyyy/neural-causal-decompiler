"""Frozen protocol for learned local ReLU directions and global realization."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import argparse
import json

from ncd.end_to_end_local_realization import LocalLearningConfig, run_case, verify_case
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "learned_local_global_v1"
ACCEPTANCE = ROOT / "validation" / "learned_local_global_acceptance.json"
SEEDS = (7201, 7202)
DIMENSIONS = (8, 32, 64, 128)
PROTOCOL = {
    "train_samples": 2048,
    "selection_samples": 512,
    "test_samples": 1024,
    "epochs": 600,
    "rollout_cases": 64,
    "rollout_horizon": 10,
    "epsilon": "0.17",
    "candidate_state_bins": "(6,5,5,5,1...1,3)",
    "candidate_action_bins": 128,
    "test_rmse_limit": 0.0001,
    "test_max_error_limit": 0.001,
    "rollout_max_error_limit": 0.001,
    "phase_min_abs_second_difference": "1/10000",
    "mean_hidden_direction_movement_min": 0.01,
    "min_hidden_direction_movement_min": 0.001,
    "mean_hidden_bias_movement_min": 0.01,
    "required_state_interval": [81, 2250],
}


def config(seed: int, dimension: int) -> LocalLearningConfig:
    return LocalLearningConfig(
        seed=seed, state_dim=dimension,
        train_samples=PROTOCOL["train_samples"],
        selection_samples=PROTOCOL["selection_samples"],
        test_samples=PROTOCOL["test_samples"],
        epochs=PROTOCOL["epochs"],
        rollout_cases=PROTOCOL["rollout_cases"],
        rollout_horizon=PROTOCOL["rollout_horizon"])


def path(seed: int, dimension: int) -> Path:
    return OUTPUT / f"seed_{seed}" / f"d_{dimension}"


def assess(seed: int, dimension: int, record: dict) -> dict:
    training = record["training"]
    certificate_interval = [record["lower_bound"], int(record["upper_bound"] or 0)]
    gates = {
        "certified": record["certificate_status"] == "certified",
        "state_interval": certificate_interval == [81, 2250],
        "state_and_control_phase": all(
            abs(Fraction(x)) > Fraction(1, 10000)
            for x in record["phase_witnesses"].values()),
        "disjoint_splits": training["splits_disjoint"],
        "test_rmse": training["test"]["rmse"] < PROTOCOL["test_rmse_limit"],
        "test_max_error": training["test"]["max_error"]
                          < PROTOCOL["test_max_error_limit"],
        "rollout": training["rollout_test"]["max_state_error"]
                   < PROTOCOL["rollout_max_error_limit"],
        "direction_learned": (
            training["mean_hidden_direction_movement"]
            > PROTOCOL["mean_hidden_direction_movement_min"]
            and training["min_hidden_direction_movement"]
            > PROTOCOL["min_hidden_direction_movement_min"]),
        "bias_learned": training["mean_hidden_bias_movement"]
                        > PROTOCOL["mean_hidden_bias_movement_min"],
    }
    return {
        "seed": seed, "state_dim": dimension,
        "system_sha256": training["system_sha256"],
        "certificate_sha256": digest(path(seed, dimension) / "certificate.json"),
        "selected_epoch": training["selected_epoch"],
        "test": training["test"],
        "rollout_test": training["rollout_test"],
        "phase_witnesses": record["phase_witnesses"],
        "mean_hidden_direction_movement": training["mean_hidden_direction_movement"],
        "min_hidden_direction_movement": training["min_hidden_direction_movement"],
        "mean_hidden_bias_movement": training["mean_hidden_bias_movement"],
        "lower_bound": record["lower_bound"],
        "upper_bound": record["upper_bound"],
        "gates": gates,
        "accepted": all(gates.values()),
    }


def generate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cases = []
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            record = run_case(config(seed, dimension), path(seed, dimension))
            cases.append(assess(seed, dimension, record))
            print(f"seed={seed} d={dimension} accepted={cases[-1]['accepted']}", flush=True)
    result = {"schema": "ncd.learned-local-global-study.v1",
              "protocol": PROTOCOL, "cases": cases,
              "accepted": all(case["accepted"] for case in cases)}
    save_json(OUTPUT / "summary.json", result)
    return result


def verify() -> dict:
    cases = []
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            record = verify_case(path(seed, dimension))
            cases.append(assess(seed, dimension, record))
            print(f"replayed seed={seed} d={dimension}", flush=True)
    expected = {"schema": "ncd.learned-local-global-study.v1",
                "protocol": PROTOCOL, "cases": cases,
                "accepted": all(case["accepted"] for case in cases)}
    if read_json(OUTPUT / "summary.json") != expected:
        raise ValueError("Learned-local study summary replay mismatch")
    acceptance = {
        "schema": "ncd.learned-local-global-acceptance.v1",
        "state": "verified" if expected["accepted"] else "failed",
        "cases": len(cases),
        "all_training_and_certificates_replayed": True,
        "all_frozen_gates_passed": expected["accepted"],
        "summary_sha256": digest(OUTPUT / "summary.json"),
        "seeds": list(SEEDS), "dimensions": list(DIMENSIONS),
        "minimum_state_intervals": {
            str(d): [81, 2250] for d in DIMENSIONS
        } if expected["accepted"] else None,
    }
    save_json(ACCEPTANCE, acceptance)
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = generate() if args.generate else verify()
    print(json.dumps({"accepted": result["accepted"]} if args.generate
                     else {"state": result["state"]}))


if __name__ == "__main__":
    main()

