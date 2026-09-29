"""Joint lower/upper acceptance for 1D infinite-horizon realization complexity."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
LOWER = ROOT / "runs" / "certified_transition_lower_seed12701"
UPPER = ROOT / "runs" / "certified_continuous_realization_seed10701"
OUTPUT = ROOT / "validation" / "certified_transition_lower_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.transition_overlap_lower import verify_transition_lower_run
from ncd.continuous_closed_realization import verify_closed_realizations
from ncd.io import digest, read_json, save_json


def main() -> None:
    lower = verify_transition_lower_run(LOWER)
    upper_replay = verify_closed_realizations(UPPER)
    upper_profile = read_json(
        UPPER / "profiles" / "profile_000" / "profile.json")
    if lower != {
        "status": "verified", "lower_bound": 6, "excluded_sizes": 5,
        "transition_aware_exclusions": 1, "arithmetic": "exact-rational",
    }:
        raise ValueError("Transition-aware lower result did not reproduce")
    if upper_replay["status"] != "verified" or upper_profile["state_dim"] != 1:
        raise ValueError("Finite upper realization did not replay")
    if upper_profile["certificate"]["upper_bound"] != 10:
        raise ValueError("Unexpected one-dimensional upper realization")
    result = {
        "accepted": True,
        "lower_run": LOWER.relative_to(ROOT).as_posix(),
        "lower_manifest_sha256": digest(LOWER / "manifest.json"),
        "upper_run": UPPER.relative_to(ROOT).as_posix(),
        "upper_manifest_sha256": digest(UPPER / "manifest.json"),
        "minimum_state_interval": [6, 10],
        "lower_verification": lower,
        "upper_verification": upper_replay,
        "claim": "five states impossible under deterministic transition closure for the frozen scalar ReLU system",
        "limitations": "the minimum is not identified; affine identity-output 1D theorem only",
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

