"""Replay every completed frozen role unit and summarize only verified worlds."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from verify_interventional_role_v1 import verify

PLAN = ROOT / "validation/interventional_role_confirmation_protocol_v1.json"
OUTPUT = ROOT / "validation/interventional_role_confirmation_verified_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def summarize(write=False, require_complete=False):
    plan = read(PLAN)
    expected = [(seed, n, split, i) for seed in plan["seeds"]
                for n in plan["nodes"] for split in plan["environments"]
                for i in range(plan["worlds_per_cell"])]
    base = ROOT / plan["output"] / "units"
    frozen = ROOT / plan["output"] / "protocol.json"
    if digest(frozen) != digest(PLAN):
        raise ValueError("Run protocol differs from frozen plan")
    rows, missing = [], []
    for unit in expected:
        seed, n, split, i = unit
        folder = base / f"seed_{seed}_n{n}_{split}_{i}"
        if not (folder / "result.json").exists():
            missing.append(list(unit))
            continue
        receipt = verify(folder)
        if receipt["unit"] != list(unit):
            raise ValueError("Verifier returned wrong unit")
        result = read(folder / "result.json")
        if result["metrics"]["hybrid"]["noise_source"] != "true_exogenous_oracle_diagnostic_only":
            raise ValueError("Noise provenance changed")
        if write:
            atomic_write(folder / "verification.json", receipt)
        rows.append({
            "unit": list(unit), "world_id": result["world_id"],
            "result_sha256": receipt["result_sha256"],
            "graph_exact": receipt["graph_exact"],
            "selected_arms": result["selected_arms"],
            "mixed_max_local_mse": result["metrics"]["mixed"]["max_local_normalized_mse"],
            "control_max_local_mse": result["metrics"]["control"]["max_local_normalized_mse"],
            "hybrid_max_local_mse": result["metrics"]["hybrid"]["max_local_normalized_mse"],
            "hybrid_max_rollout_mse_with_oracle_noise": result["metrics"]["hybrid"]["max_rollout_normalized_mse"],
            "hybrid_local_pass": receipt["hybrid_meets_local_0_01"],
            "verification_sha256": digest(folder / "verification.json") if write else None,
        })
    if require_complete and missing:
        raise ValueError(f"{len(missing)} declared units remain unverified")
    cells = []
    for seed in plan["seeds"]:
        for n in plan["nodes"]:
            for split in plan["environments"]:
                subset = [r for r in rows if r["unit"][:3] == [seed, n, split]]
                cells.append({
                    "seed": seed, "nodes": n, "environment": split,
                    "verified_worlds": len(subset),
                    "local_gate_passes": sum(r["hybrid_local_pass"] for r in subset),
                    "graph_exact_worlds": sum(r["graph_exact"] for r in subset),
                    "max_hybrid_local_mse": (max(r["hybrid_max_local_mse"] for r in subset)
                                             if subset else None),
                })
    output = {
        "schema": "ncd.interventional-role-confirmation-summary.v1",
        "status": "verified-all-300" if not missing else "verified-partial",
        "protocol_sha256": digest(PLAN),
        "verifier_sha256": digest(ROOT / "validation/verify_interventional_role_v1.py"),
        "summary_source_sha256": digest(__file__),
        "declared_worlds": len(expected), "verified_worlds": len(rows),
        "missing_units": missing, "cells": cells, "worlds": rows,
        "hybrid_local_gate_passes": sum(r["hybrid_local_pass"] for r in rows),
        "graph_exact_worlds": sum(r["graph_exact"] for r in rows),
        "frozen_hybrid_all_worlds_local_gate": (
            False if any(not r["hybrid_local_pass"] for r in rows)
            else True if not missing else None),
        "oracle_noise_used_for_rollout": True,
        "recovered_noise_model_verified": False,
        "statistical_99_percent_claim_established": False,
        "original_claim_closed": False, "original_objective_achieved": False,
    }
    if write:
        atomic_write(OUTPUT, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = summarize(write=args.write, require_complete=args.require_complete)
    print(json.dumps({
        "status": result["status"], "verified_worlds": result["verified_worlds"],
        "declared_worlds": result["declared_worlds"],
        "hybrid_local_gate_passes": result["hybrid_local_gate_passes"],
        "graph_exact_worlds": result["graph_exact_worlds"],
        "frozen_hybrid_all_worlds_local_gate": result["frozen_hybrid_all_worlds_local_gate"],
        "original_objective_achieved": False,
    }, sort_keys=True))
