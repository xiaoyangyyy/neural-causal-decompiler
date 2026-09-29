"""Independent acceptance gate for unbounded-horizon finite realization."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "certified_continuous_realization_seed10701"
OUTPUT = ROOT / "validation" / "certified_continuous_realization_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.continuous_closed_realization import verify_closed_realizations
from ncd.io import digest, read_json, save_json


def main() -> None:
    verification = verify_closed_realizations(RUN)
    summary = read_json(RUN / "summary.json")
    profiles = [read_json(RUN / "profiles" / f"profile_{i:03d}" / "profile.json")
                for i in range(summary["profiles"])]
    expected = {
        "status": "verified", "profiles_replayed": 2,
        "largest_state_dim": 2, "largest_realization": 100,
        "initial_cells_verified": 20,
        "transition_segments_verified": 100,
        "packing_pairs_verified": 310,
    }
    if verification != expected:
        raise ValueError("Unexpected infinite-horizon replay result")
    bounds = [(p["state_dim"], p["certificate"]["lower_bound"],
               p["certificate"]["upper_bound"]) for p in profiles]
    if bounds != [(1, 5, 10), (2, 25, 100)]:
        raise ValueError("Unexpected realization-size intervals")
    if any(p["verification"]["horizon"] != "unbounded"
           or p["certificate"]["minimal"] for p in profiles):
        raise ValueError("Invalid horizon or minimality claim")
    result = {
        "accepted": True, "run": RUN.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "verification": verification, "size_bounds": bounds,
        "formal_runtime_seconds": summary["runtime_seconds"],
        "claim": "uniform epsilon simulation for arbitrary-length continuous action words on the frozen separable ReLU benchmark",
        "limitations": "1D/2D only; lower and upper size bounds differ; no arbitrary-network or minimal-realization claim",
    }
    save_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

