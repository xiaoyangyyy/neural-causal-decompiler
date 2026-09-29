"""Replay exact trained-network realization certificates and record acceptance."""
from pathlib import Path

from ncd.continuous_compositional_realization import (
    verify_frozen_scale_models, verify_weighted_scale_models,
)
from ncd.io import digest, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "certified_continuous_scale_seed4701"
UNIFORM = ROOT / "runs" / "compositional_scale_seed4701"
WEIGHTED = ROOT / "runs" / "weighted_compositional_scale_seed4701"
OUTPUT = ROOT / "validation" / "compositional_trained_acceptance.json"


def main() -> None:
    uniform = verify_frozen_scale_models(SOURCE, UNIFORM)
    weighted = verify_weighted_scale_models(SOURCE, WEIGHTED)
    dimensions = [8, 32, 64, 128]
    if ([r["state_dim"] for r in weighted["records"]] != dimensions
            or any(r["lower_bound"] != 81 or r["upper_bound"] != "31104"
                   for r in weighted["records"])
            or [r["system_sha256"] for r in weighted["records"]]
            != [r["system_sha256"] for r in uniform["records"]]):
        raise ValueError("Frozen-model acceptance bounds or hashes changed")
    status = {
        "schema": "ncd.compositional-trained-acceptance.v1",
        "state": "verified",
        "epsilon": "0.17",
        "uniform_upper_by_dimension": {
            str(r["state_dim"]): r["upper_bound"] for r in uniform["records"]},
        "weighted_interval_by_dimension": {
            str(r["state_dim"]): [r["lower_bound"], int(r["upper_bound"])]
            for r in weighted["records"]},
        "source_system_sha256": {
            str(r["state_dim"]): r["system_sha256"] for r in weighted["records"]},
        "uniform_summary_sha256": digest(UNIFORM / "summary.json"),
        "weighted_summary_sha256": digest(WEIGHTED / "summary.json"),
        "scope": "exact rational full-unit-domain infinite-horizon simulation of four frozen trained affine ReLU systems",
    }
    save_json(OUTPUT, status)
    print(OUTPUT)


if __name__ == "__main__":
    main()


