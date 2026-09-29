"""Replay formal continuous-separation evidence and emit acceptance JSON."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.continuous_separation import verify_continuous_separation
from ncd.io import read_json, save_json


def main() -> None:
    run = ROOT / "runs" / "certified_continuous"
    replay = verify_continuous_separation(run)
    summary = read_json(run / "summary.json")
    result = {
        "scope": summary["scope"],
        "checks": {
            "complete_replay": replay["status"] == "verified",
            "all_certificates_verified": summary["all_certificates_verified"],
            "separation_witness_present": summary["statuses"]["separated"] > 0,
            "upper_certificate_present": summary["statuses"]["certified-within-epsilon"] > 0,
            "budget_preserves_unresolved": summary["statuses"]["unresolved"] > 0,
            "chromatic_lower_bound_verified": replay["incompatibility_lower_bound"]["lower_bound"] == 3,
            "continuous_realization_interval_closed": (
                replay["continuous_realization"]["minimal"]
                and replay["continuous_realization"]["lower_bound"]
                == replay["continuous_realization"]["upper_bound"] == 2
            ),
        },
        "replay": replay,
        "summary": summary,
    }
    result["accepted"] = all(result["checks"].values())
    if not result["accepted"]:
        raise SystemExit("Certified continuous acceptance failed")
    save_json(ROOT / "validation" / "certified_continuous_acceptance.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
