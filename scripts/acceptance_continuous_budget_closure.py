"""Acceptance gate for the 64D/H3 nonlinear budget-closure supplement."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "certified_continuous_nonlinear_budget28_seed7719"
OUTPUT = ROOT / "validation" / "continuous_nonlinear_budget28_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.continuous_nonlinear import verify_continuous_nonlinear
from ncd.io import digest, read_json, save_json


METHODS = (
    "hybrid_best",
    "hybrid_widest",
    "independent_best",
    "independent_widest",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    verification = verify_continuous_nonlinear(RUN)
    summary = read_json(RUN / "summary.json")
    profile = read_json(RUN / "profiles" / "profile_000" / "profile.json")
    near = profile["results"]["near"]
    far = profile["results"]["far"]

    require(verification == {
        "status": "verified",
        "profiles_replayed": 1,
        "certificates_verified": 8,
        "largest_state_dim": 64,
        "largest_horizon": 3,
    }, "Unexpected budget-closure replay result")
    require(summary["all_models_fit_max_error_below_1e-9"], "Model fit tolerance failed")
    require(summary["all_training_domains_certify_phase_crossings"], "Phase crossing missing")
    require(all(far[method]["status"] == "separated" for method in METHODS),
            "Every far-pair ablation must be separated")
    require(all(near[method]["status"] == "certified-within-epsilon"
                for method in ("hybrid_best", "independent_best")),
            "Best-bound methods must close the near pair")
    require(all(near[method]["status"] == "unresolved"
                for method in ("hybrid_widest", "independent_widest")),
            "Widest baselines must remain unresolved at the same budget")
    require(near["hybrid_best"]["leaves"] == 28
            and near["independent_best"]["leaves"] == 28,
            "Closure must occur at the frozen 28-leaf budget")
    require(near["hybrid_best"]["upper_bound"] <= 2 * near["epsilon"],
            "Hybrid best-bound upper proof did not close")
    require(near["independent_best"]["upper_bound"] <= 2 * near["epsilon"],
            "Independent best-bound upper proof did not close")
    require(near["leaf_methods"]["hybrid_best"]["relational-stable"] == 0,
            "The three-step closure must not be attributed to relational leaves")

    result = {
        "accepted": True,
        "run": RUN.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "verification": verification,
        "formal_runtime_seconds": summary["runtime_seconds"],
        "assertions": {
            "best_bound_closes_at_28_leaves": True,
            "widest_unresolved_at_28_leaves": True,
            "closure_uses_independent_ibp": True,
            "far_pair_separated": True,
        },
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()