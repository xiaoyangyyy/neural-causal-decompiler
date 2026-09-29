"""Joint acceptance of improved nine-center upper and seven-state lower."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
UPPER = ROOT / "runs" / "certified_shifted_realization_seed14701"
LOWER = ROOT / "runs" / "certified_multiswitch_lower_seed13701"
OUTPUT = ROOT / "validation" / "certified_shifted_realization_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.shifted_realization import verify_shifted_realizations
from ncd.multiswitch_lower import verify_multiswitch_run
from ncd.io import digest, read_json, save_json


def main() -> None:
    upper = verify_shifted_realizations(UPPER)
    lower = verify_multiswitch_run(LOWER)
    profiles = [read_json(UPPER / "profiles" / f"profile_{i:03d}" / "profile.json")
                for i in range(2)]
    if upper != {
        "status": "verified", "profiles_replayed": 2,
        "largest_state_dim": 2, "largest_realization": 81,
        "initial_cells_verified": 18, "observation_relations_verified": 18,
        "action_segments_verified": 84, "packing_pairs_verified": 310,
    }:
        raise ValueError("Unexpected shifted-realization replay")
    if lower["lower_bound"] != 7 or lower["status"] != "verified":
        raise ValueError("Multi-switch lower proof did not replay")
    if [(p["state_dim"], p["certificate"]["upper_bound"])
        for p in profiles] != [(1, 9), (2, 81)]:
        raise ValueError("Unexpected nine-center realization counts")
    result = {
        "accepted": True,
        "minimum_state_interval_1d": [7, 9],
        "minimum_state_interval_2d": [25, 81],
        "upper_run": UPPER.relative_to(ROOT).as_posix(),
        "upper_manifest_sha256": digest(UPPER / "manifest.json"),
        "lower_run": LOWER.relative_to(ROOT).as_posix(),
        "lower_manifest_sha256": digest(LOWER / "manifest.json"),
        "upper_verification": upper, "lower_verification": lower,
        "claim": "nine- and eighty-one-state executable infinite-horizon realizations for the frozen separable benchmark",
        "limitations": "1D minimum still unresolved among seven, eight and nine; 2D lower bound remains packing only",
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

