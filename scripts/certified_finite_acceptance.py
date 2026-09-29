"""Replay the formal finite-realization run and emit machine-readable acceptance."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ncd.certified_experiment import verify_certified_finite
from ncd.io import read_json, save_json


def main() -> None:
    run = ROOT / "runs" / "certified_finite_seed2701"
    replay = verify_certified_finite(run)
    summary = read_json(run / "summary.json")
    checks = {
        "formal_seed": 2701,
        "scope": summary["claim_scope"],
        "checks": {
            "complete_replay": replay["status"] == "verified",
            "all_active_exact": summary["active_exact_cases"] == summary["cases"],
            "all_oracle_certificates": summary["all_oracle_certificates_verified"],
            "all_active_certificates": summary["all_active_certificates_verified"],
            "all_approximate_intervals_closed": summary["all_approximate_intervals_closed"],
            "budget_preserves_unresolved": replay["limited_unresolved_pairs"] > 0,
        },
        "replay": replay,
        "summary": summary,
    }
    checks["accepted"] = all(checks["checks"].values())
    if not checks["accepted"]:
        raise SystemExit("Certified finite acceptance failed")
    save_json(ROOT / "validation" / "certified_finite_acceptance.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
