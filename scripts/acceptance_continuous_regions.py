"""Independent acceptance gate for continuous initial-state region evidence."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "certified_continuous_regions_seed8701"
OUTPUT = ROOT / "validation" / "certified_continuous_regions_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.continuous_regions import verify_continuous_regions
from ncd.io import digest, read_json, save_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    verification = verify_continuous_regions(RUN)
    summary = read_json(RUN / "summary.json")
    profiles = [
        read_json(RUN / "profiles" / f"profile_{index:03d}" / "profile.json")
        for index in range(summary["profiles"])
    ]
    require(verification == {
        "status": "verified",
        "profiles_replayed": 3,
        "certificates_verified": 12,
        "largest_state_dim": 64,
        "largest_horizon": 10,
    }, "Unexpected continuous-region replay result")
    require(summary["all_models_fit_max_error_below_1e-9"], "Model fit tolerance failed")
    require(summary["statuses"]["relational"] == {
        "robustly-separated": 3,
        "certified-within-epsilon": 3,
        "unresolved": 0,
    }, "Relational region statuses did not reproduce")
    require(summary["statuses"]["independent"] == {
        "robustly-separated": 3,
        "certified-within-epsilon": 0,
        "unresolved": 3,
    }, "Independent region baseline did not reproduce")

    for profile in profiles:
        require(profile["region_radius"] == 0.001, "Unexpected region radius")
        near = profile["results"]["near"]
        far = profile["results"]["far"]
        require(near["relational"]["upper_bound"] <= 2 * near["epsilon"]
                and near["relational"]["leaves"] == 1,
                "Relational near-region upper proof failed")
        require(near["independent"]["upper_bound"] > 2 * near["epsilon"]
                and near["independent"]["leaves"] == profile["leaf_budget"],
                "Independent near-region budget frontier failed")
        require(far["relational"]["robust_lower_bound"] > 2 * far["epsilon"]
                and far["relational"]["witness_bound_method"] == "relational-stable",
                "Relational robust-separation witness failed")
        require(far["independent"]["robust_lower_bound"] > 2 * far["epsilon"],
                "Independent robust-separation witness failed")

    result = {
        "accepted": True,
        "run": RUN.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "verification": verification,
        "formal_runtime_seconds": summary["runtime_seconds"],
        "assertions": {
            "continuous_initial_boxes_covered": True,
            "relational_near_regions_closed": 3,
            "robust_far_regions_separated": 3,
            "independent_near_regions_unresolved": 3,
        },
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()