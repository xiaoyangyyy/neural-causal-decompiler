"""Posthoc candidate-data-only OLS diagnostic on the first verified cell.

This is analysis after seeing confirmation failures. It cannot be reused as
confirmation evidence for a new selection rule.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))
import numpy as np

from ncd.multiverse import GraphWorld
from confirm_interventional_role_v1 import evaluate, sample_seed
from verify_interventional_role_v1 import verify

BASE = ROOT / "runs/interventional_role_confirmation_v1/units"
OUTPUT = ROOT / "validation/interventional_role_ols_diagnostic_v1.json"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ols_models(folder, inferred, selection):
    batches = read(folder / "candidate/batches.json")["batches"]
    models = []
    with np.load(folder / "candidate/mechanism_observations.npz", allow_pickle=False) as archive:
        for target in range(len(inferred)):
            parents = tuple(map(int, np.flatnonzero(inferred[:, target])))
            allowed = [archive[batch["source"]] for batch in batches
                       if (selection == "all" or batch["split"] == "fit")
                       and str(target) not in batch["interventions"]]
            data = np.concatenate(allowed)
            design = np.column_stack((np.ones(len(data)), data[:, parents]))
            coefficients = np.linalg.lstsq(design, data[:, target], rcond=None)[0]
            models.append(
                lambda observed, parents=parents, coefficients=coefficients:
                np.column_stack((np.ones(len(observed)), observed[:, parents])) @ coefficients)
    return models


def compute():
    rows = []
    for index in range(10):
        folder = BASE / f"seed_8301_n3_test_id_{index}"
        receipt = verify(folder)
        if receipt["status"] != "verified-one-independent-world":
            raise ValueError("Source confirmation world not independently replayed")
        result = read(folder / "result.json")
        world = GraphWorld.from_dict(read(folder / "truth_only/world.json"))
        inferred = np.asarray(result["inferred_graph"], dtype=bool)
        _, exogenous = world.sample(
            seed=sample_seed(world.identity, "heldout_eval_exogenous_v1"),
            samples=512, return_exogenous=True)
        scores = {}
        for selection in ("fit", "all"):
            models = ols_models(folder, inferred, selection)
            evaluated = evaluate(
                world, inferred, models, exogenous,
                np.asarray(result["normalizers"], dtype=float))
            scores[selection] = {
                "max_local_normalized_mse": evaluated["max_local_normalized_mse"],
                "max_rollout_normalized_mse_with_oracle_noise":
                    evaluated["max_rollout_normalized_mse"],
            }
        rows.append({
            "unit": result["unit"], "world_id": world.identity,
            "world_family_for_diagnostic_stratification_only": world.family,
            "graph_exact": result["graph_exact"],
            "source_result_sha256": digest(folder / "result.json"),
            "frozen_hybrid_max_local_mse": result["metrics"]["hybrid"]["max_local_normalized_mse"],
            "ols_fit_only": scores["fit"],
            "ols_all_usable_training_rows": scores["all"],
        })
    return {
        "schema": "ncd.interventional-role-ols-diagnostic.v1",
        "status": "posthoc-first-cell-diagnostic-only",
        "source_protocol_sha256": digest(
            ROOT / "validation/interventional_role_confirmation_protocol_v1.json"),
        "diagnostic_source_sha256": digest(__file__),
        "worlds": rows,
        "independent_new_confirmation_worlds_for_ols": 0,
        "rule_chosen_after_source_confirmation_seen": True,
        "candidate_training_uses_truth_graph_or_equations": False,
        "rollout_uses_true_noise_oracle": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: diagnostic --write|--verify")
    expected = compute()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain existing diagnostic")
        OUTPUT.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    elif read(OUTPUT) != expected:
        raise ValueError("Posthoc diagnostic changed")
    print(json.dumps({
        "status": expected["status"],
        "worlds": len(expected["worlds"]),
        "ols_fit_local_passes": sum(
            row["ols_fit_only"]["max_local_normalized_mse"] <= .01
            for row in expected["worlds"]),
        "ols_all_local_passes": sum(
            row["ols_all_usable_training_rows"]["max_local_normalized_mse"] <= .01
            for row in expected["worlds"]),
        "original_claim_closed": False,
    }, sort_keys=True))
