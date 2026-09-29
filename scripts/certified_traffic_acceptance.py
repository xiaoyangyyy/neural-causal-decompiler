"""Replay formal traffic realization evidence and emit acceptance JSON."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.io import read_json, save_json
from ncd.traffic_realization import verify_certified_traffic


def main() -> None:
    run = ROOT / "runs" / "certified_traffic_seed3701"
    replay = verify_certified_traffic(run)
    summary = read_json(run / "summary.json")
    result = {
        "scope": summary["scope"],
        "checks": {
            "complete_retraining_replay": replay["status"] == "verified",
            "all_training_pairs_fit": summary["all_training_pairs_fit"],
            "all_certificates_verified": summary["all_certificates_verified"],
            "all_active_exact": summary["active_exact_cases"] == summary["cases"],
            "budget_preserves_unresolved": summary["limited_unresolved_pairs"] > 0,
        },
        "replay": replay,
        "summary": summary,
    }
    result["accepted"] = all(result["checks"].values())
    if not result["accepted"]:
        raise SystemExit("Certified traffic acceptance failed")
    save_json(ROOT / "validation" / "certified_traffic_acceptance.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
