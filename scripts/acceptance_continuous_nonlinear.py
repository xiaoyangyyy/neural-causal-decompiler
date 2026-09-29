"""Independent acceptance gate for formal phase-crossing continuous evidence."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "certified_continuous_nonlinear_seed5701"
OUTPUT = ROOT / "validation" / "certified_continuous_nonlinear_acceptance.json"
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
    profiles = [
        read_json(RUN / "profiles" / f"profile_{index:03d}" / "profile.json")
        for index in range(summary["profiles"])
    ]
    require(verification == {
        "status": "verified",
        "profiles_replayed": 3,
        "certificates_verified": 24,
        "largest_state_dim": 64,
        "largest_horizon": 3,
    }, "Unexpected formal replay result")
    require(summary["all_models_fit_max_error_below_1e-9"], "A learned model missed fit tolerance")
    require(summary["all_training_domains_certify_phase_crossings"], "A gate domain does not cross zero")
    require(all(profile["results"]["far"][method]["status"] == "separated"
                for profile in profiles for method in METHODS),
            "Every far pair must be formally separated")

    first = profiles[0]["results"]["near"]
    require(all(first[method]["status"] == "certified-within-epsilon" for method in METHODS),
            "The one-step near pair must close under every ablation")
    require(first["hybrid_best"]["leaves"] < first["hybrid_widest"]["leaves"]
            < first["independent_widest"]["leaves"],
            "Expected one-step leaf-efficiency ordering was not reproduced")
    require(first["leaf_methods"]["hybrid_widest"]["relational-stable"] > 0
            and first["leaf_methods"]["hybrid_widest"]["independent-ibp"] > 0,
            "The hybrid certificate must contain stable and fallback leaves")

    second = profiles[1]["results"]["near"]
    require(second["hybrid_widest"]["status"] == "certified-within-epsilon"
            and second["hybrid_best"]["status"] == "unresolved",
            "The two-step greedy-strategy limitation was not reproduced")
    require(all(profiles[2]["results"]["near"][method]["status"] == "unresolved"
                for method in METHODS),
            "The three-step budget frontier must remain explicitly unresolved")

    result = {
        "accepted": True,
        "run": RUN.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "verification": verification,
        "formal_runtime_seconds": summary["runtime_seconds"],
        "assertions": {
            "phase_crossings_certified": True,
            "far_pairs_separated": 3,
            "one_step_ablation_closed": True,
            "two_step_greedy_limitation_reproduced": True,
            "three_step_budget_frontier_unresolved": True,
        },
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()