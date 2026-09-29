"""Independent acceptance gate for global continuous behavioral covers."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "certified_continuous_cover_seed9701"
OUTPUT = ROOT / "validation" / "certified_continuous_cover_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.continuous_cover import verify_continuous_covers
from ncd.io import digest, read_json, save_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    verification = verify_continuous_covers(RUN)
    summary = read_json(RUN / "summary.json")
    profiles = [
        read_json(RUN / "profiles" / f"profile_{index:03d}" / "profile.json")
        for index in range(summary["profiles"])
    ]
    require(verification == {
        "status": "verified",
        "profiles_replayed": 2,
        "certificates_verified": 340,
        "largest_state_dim": 2,
        "largest_cover": 25,
    }, "Unexpected global-cover replay result")
    require(summary["all_minimal"], "A behavioral cover interval did not close")
    require(summary["total_cells_verified"] == 30, "Unexpected cover-cell count")
    require(summary["total_packing_pairs_verified"] == 310,
            "Unexpected packing-pair count")
    require(
        [(profile["state_dim"],
          profile["certificate"]["lower_bound"],
          profile["certificate"]["upper_bound"])
         for profile in profiles] == [(1, 5, 5), (2, 25, 25)],
        "Expected 1D and 2D minimal cover numbers did not reproduce",
    )
    require(all(
        cell["certificate"]["status"] == "certified-within-epsilon"
        for profile in profiles
        for cell in profile["certificate"]["cells"]
    ), "A global-cover cell upper proof is not closed")
    require(all(
        pair["certificate"]["status"] == "separated"
        for profile in profiles
        for pair in profile["certificate"]["packing_pair_certificates"]
    ), "A packing lower proof is not separated")

    result = {
        "accepted": True,
        "run": RUN.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "verification": verification,
        "formal_runtime_seconds": summary["runtime_seconds"],
        "assertions": {
            "one_dimensional_cover_number": 5,
            "two_dimensional_cover_number": 25,
            "all_cover_intervals_closed": True,
            "complete_unit_domains_covered": True,
        },
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()