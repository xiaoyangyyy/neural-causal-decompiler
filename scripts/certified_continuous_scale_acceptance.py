"""Replay the formal continuous scaling study and emit acceptance JSON."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.continuous_scale import verify_continuous_scale
from ncd.io import read_json, save_json


def main() -> None:
    run = ROOT / "runs" / "certified_continuous_scale_seed4701"
    replay = verify_continuous_scale(run)
    summary = read_json(run / "summary.json")
    result = {
        "formal_seed": 4701,
        "scope": summary["scope"],
        "checks": {
            "complete_retraining_replay": replay["status"] == "verified",
            "four_profiles": summary["profiles"] == 4,
            "reaches_128_dimensions": summary["largest_state_dim"] == 128,
            "reaches_horizon_20": summary["largest_horizon"] == 20,
            "all_models_fit": summary["all_training_fit_max_error_below_1e-9"],
            "relational_resolves_all_pairs": summary["relational_statuses"]["unresolved"] == 0,
            "relational_certifies_near_pairs": summary["relational_statuses"]["certified-within-epsilon"] == 4,
            "independent_ibp_preserves_unresolved": summary["independent_statuses"]["unresolved"] == 4,
            "all_certificates_replayed": replay["certificates_verified"] == 16,
        },
        "replay": replay,
        "summary": summary,
    }
    result["accepted"] = all(result["checks"].values())
    if not result["accepted"]:
        raise SystemExit("Continuous scale acceptance failed")
    save_json(ROOT / "validation" / "certified_continuous_scale_acceptance.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
