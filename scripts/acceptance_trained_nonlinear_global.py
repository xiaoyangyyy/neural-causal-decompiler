"""Preregistered trained nonlinear global-realization study and replay."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import argparse

from ncd.trained_nonlinear_realization import (
    NonlinearTrainingConfig, run_case, verify_case,
)
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "trained_nonlinear_global_v1"
ACCEPTANCE = ROOT / "validation" / "trained_nonlinear_global_acceptance.json"
SEEDS = (6101, 6102)
DIMENSIONS = (8, 32, 64, 128)
PROTOCOL = {
    "train_samples": 2048,
    "selection_samples": 512,
    "test_samples": 1024,
    "rollout_cases": 64,
    "rollout_horizon": 10,
    "epsilon": "0.17",
    "abstraction": "state bins (6,5,5,5,1...1,3), state radii (0.17 x4,1.01...,0.4), 128 action bins",
    "test_rmse_gate": 0.0002,
    "test_max_error_gate": 0.001,
    "rollout_max_state_error_gate": 0.001,
    "minimum_phase_second_difference": "1/10000",
    "required_interval": [81, 2250],
}


def _case_path(seed: int, dimension: int) -> Path:
    return OUTPUT / f"seed_{seed}" / f"d_{dimension}"


def _record(seed: int, dimension: int, record: dict) -> dict:
    training = record["training"]
    witnesses = record["phase_witnesses"]
    phase = all(abs(Fraction(value)) > Fraction(1, 10000)
                for value in witnesses.values())
    gates = {
        "certificate": record["certificate_status"] == "certified",
        "state_interval": [record["lower_bound"], int(record["upper_bound"] or 0)]
                          == [81, 2250],
        "phase_crossings": phase,
        "disjoint_splits": training["splits_disjoint"],
        "test_rmse": training["test"]["rmse"] < PROTOCOL["test_rmse_gate"],
        "test_max_error": training["test"]["max_error"] < PROTOCOL["test_max_error_gate"],
        "rollout": training["rollout_test"]["max_state_error"]
                   < PROTOCOL["rollout_max_state_error_gate"],
    }
    case = _case_path(seed, dimension)
    return {
        "seed": seed, "state_dim": dimension,
        "system_sha256": training["system_sha256"],
        "certificate_sha256": digest(case / "certificate.json"),
        "chosen_ridge_alpha": training["chosen_ridge_alpha"],
        "test": training["test"],
        "rollout_test": training["rollout_test"],
        "phase_witnesses": witnesses,
        "lower_bound": record["lower_bound"],
        "upper_bound": record["upper_bound"],
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _config(seed: int, dimension: int) -> NonlinearTrainingConfig:
    return NonlinearTrainingConfig(
        seed=seed, state_dim=dimension,
        train_samples=PROTOCOL["train_samples"],
        selection_samples=PROTOCOL["selection_samples"],
        test_samples=PROTOCOL["test_samples"],
        rollout_cases=PROTOCOL["rollout_cases"],
        rollout_horizon=PROTOCOL["rollout_horizon"])


def generate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cases = []
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            record = run_case(_config(seed, dimension), _case_path(seed, dimension))
            cases.append(_record(seed, dimension, record))
            print(f"seed={seed} d={dimension} accepted={cases[-1]['accepted']}", flush=True)
    summary = {
        "schema": "ncd.trained-nonlinear-global-study.v1",
        "protocol": PROTOCOL, "cases": cases,
        "accepted": all(case["accepted"] for case in cases),
    }
    save_json(OUTPUT / "summary.json", summary)
    return summary


def verify() -> dict:
    stored = read_json(OUTPUT / "summary.json")
    cases = []
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            record = verify_case(_case_path(seed, dimension))
            cases.append(_record(seed, dimension, record))
            print(f"replayed seed={seed} d={dimension}", flush=True)
    expected = {
        "schema": "ncd.trained-nonlinear-global-study.v1",
        "protocol": PROTOCOL, "cases": cases,
        "accepted": all(case["accepted"] for case in cases),
    }
    if stored != expected:
        raise ValueError("Nonlinear study summary replay mismatch")
    status = {
        "schema": "ncd.trained-nonlinear-global-acceptance.v1",
        "state": "verified" if expected["accepted"] else "failed",
        "cases": len(cases),
        "all_certificates_replayed": True,
        "all_frozen_gates_passed": expected["accepted"],
        "summary_sha256": digest(OUTPUT / "summary.json"),
        "dimensions": list(DIMENSIONS),
        "seeds": list(SEEDS),
        "minimum_state_intervals": {
            str(d): [81, 2250] for d in DIMENSIONS
        } if expected["accepted"] else None,
    }
    save_json(ACCEPTANCE, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.generate == args.verify:
        parser.error("Choose exactly one of --generate and --verify")
    result = generate() if args.generate else verify()
    print(result["accepted"] if args.generate else result["state"])


if __name__ == "__main__":
    main()

