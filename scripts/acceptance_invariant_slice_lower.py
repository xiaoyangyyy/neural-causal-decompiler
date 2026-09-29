"""Acceptance for 2D invariant-slice lift of the scalar neural lower bound."""
from pathlib import Path

from ncd.continuous_separation import ContinuousReLUSystem
from ncd.invariant_slice_lower import run_case, verify_case
from ncd.multiswitch_lower import verify_multiswitch_run
from ncd.shifted_realization import verify_shifted_realizations, verify_shifted_realization
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "certified_shifted_realization_seed14701"
SCALAR = SOURCE / "profiles" / "profile_000" / "system.json"
PRODUCT = SOURCE / "profiles" / "profile_001" / "system.json"
OLD_LOWER = ROOT / "runs" / "certified_multiswitch_lower_seed13701"
OUTPUT = ROOT / "runs" / "invariant_slice_lower_v1"
STATUS = ROOT / "validation" / "invariant_slice_lower_acceptance.json"


def main():
    scalar = ContinuousReLUSystem.from_dict(read_json(SCALAR))
    product = ContinuousReLUSystem.from_dict(read_json(PRODUCT))
    old_lower = verify_multiswitch_run(OLD_LOWER)
    upper_run = verify_shifted_realizations(SOURCE)
    profile = read_json(SOURCE / "profiles" / "profile_001" / "profile.json")
    upper = verify_shifted_realization(product, profile["certificate"])
    if (old_lower["lower_bound"] != 7 or upper["lower_bound"] != 25
            or upper["upper_bound"] != 81):
        raise ValueError("Historical scalar/2D certificates did not replay")
    result = run_case(scalar, product, OUTPUT)
    if verify_case(scalar, product, OUTPUT) != result:
        raise ValueError("Fresh invariant-slice certificate failed replay")
    if result["status"] != "certified-lower-bound" or result["lower_bound"] != 28:
        raise ValueError("Invariant-slice lift did not improve the lower bound")
    certificate = read_json(OUTPUT / "certificate.json")
    summary = {
        "schema": "ncd.invariant-slice-acceptance.v1",
        "status": "accepted",
        "scalar_model": str(SCALAR.relative_to(ROOT)).replace("\\", "/"),
        "scalar_sha256": digest(SCALAR),
        "product_model": str(PRODUCT.relative_to(ROOT)).replace("\\", "/"),
        "product_sha256": digest(PRODUCT),
        "certificate_sha256": digest(OUTPUT / "certificate.json"),
        "scalar_lower": old_lower["lower_bound"],
        "slice_count": result["slice_count"],
        "previous_2d_lower": upper["lower_bound"],
        "new_2d_lower": result["lower_bound"],
        "verified_2d_upper": upper["upper_bound"],
        "minimum_state_interval_2d": [result["lower_bound"], upper["upper_bound"]],
        "epsilon": certificate["epsilon"],
        "upper_manifest_sha256": digest(SOURCE / "manifest.json"),
        "scalar_lower_manifest_sha256": digest(OLD_LOWER / "manifest.json"),
    }
    save_json(OUTPUT / "summary.json", summary)
    save_json(STATUS, summary)
    print(summary["minimum_state_interval_2d"], flush=True)


if __name__ == "__main__":
    main()
