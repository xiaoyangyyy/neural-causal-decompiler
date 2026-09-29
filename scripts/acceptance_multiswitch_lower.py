"""Joint acceptance of seven-state lower and ten-state executable upper."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
LOWER = ROOT / "runs" / "certified_multiswitch_lower_seed13701"
PRIOR = ROOT / "runs" / "certified_transition_lower_seed12701"
UPPER = ROOT / "runs" / "certified_continuous_realization_seed10701"
OUTPUT = ROOT / "validation" / "certified_multiswitch_lower_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.multiswitch_lower import verify_multiswitch_run
from ncd.transition_overlap_lower import verify_transition_lower_run
from ncd.continuous_closed_realization import verify_closed_realizations
from ncd.io import digest, read_json, save_json


def main() -> None:
    lower = verify_multiswitch_run(LOWER)
    prior = verify_transition_lower_run(PRIOR)
    upper = verify_closed_realizations(UPPER)
    profile = read_json(UPPER / "profiles" / "profile_000" / "profile.json")
    if lower != {
        "status": "verified", "lower_bound": 7,
        "excluded_sizes": 6, "multi_switch_exclusions": 2,
        "arithmetic": "exact-rational",
    }:
        raise ValueError("Unexpected multi-switch lower proof")
    if prior["lower_bound"] != 6 or upper["status"] != "verified":
        raise ValueError("Prior lower or finite upper proof failed replay")
    if profile["state_dim"] != 1 or profile["certificate"]["upper_bound"] != 10:
        raise ValueError("Unexpected scalar upper realization")
    result = {
        "accepted": True,
        "minimum_state_interval": [7, 10],
        "lower_run": LOWER.relative_to(ROOT).as_posix(),
        "lower_manifest_sha256": digest(LOWER / "manifest.json"),
        "prior_lower_replayed": prior,
        "upper_run": UPPER.relative_to(ROOT).as_posix(),
        "upper_manifest_sha256": digest(UPPER / "manifest.json"),
        "lower_verification": lower, "upper_verification": upper,
        "claim": "all deterministic realizations with six or fewer states fail on the complete continuous intervention domain",
        "limitations": "seven through nine states unresolved; scalar affine identity-output theorem",
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

