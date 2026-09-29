"""Independent acceptance gate for generic coupled ReLU realization."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "certified_generic_grid_seed11701"
OUTPUT = ROOT / "validation" / "certified_generic_grid_acceptance.json"
sys.path.insert(0, str(ROOT))

from ncd.continuous_generic_realization import verify_generic_grid
from ncd.continuous_separation import ContinuousReLUSystem
from ncd.io import digest, read_json, save_json


def main() -> None:
    result = verify_generic_grid(RUN)
    expected = {
        "status": "certified", "state_dim": 2, "states": 100,
        "initial_cells_verified": 100, "relation_observations_verified": 100,
        "transition_boxes_verified": 10000, "packing_pairs_verified": 300,
        "lower_bound": 25, "upper_bound": 100,
        "worst_transition_bound": 0.1092500000000016,
        "horizon": "unbounded",
    }
    if result != expected:
        raise ValueError("Unexpected coupled-system certificate replay")
    system = ContinuousReLUSystem.from_dict(read_json(RUN / "system.json"))
    x = system.step([0.6, 0.4], [0.3, 0.4])
    y = system.step([0.4, 0.6], [0.3, 0.4])
    if not np.isclose(x[0], 0.1 + 0.25*0.6 + 0.1*0.4 + 0.2*0.3 + 0.05*0.2):
        raise ValueError("Coupled ReLU positive phase mismatch")
    if not np.isclose(y[0], 0.1 + 0.25*0.4 + 0.1*0.6 + 0.2*0.3):
        raise ValueError("Coupled ReLU negative phase mismatch")
    summary = read_json(RUN / "summary.json")
    acceptance = {
        "accepted": True, "run": RUN.relative_to(ROOT).as_posix(),
        "manifest_sha256": digest(RUN / "manifest.json"),
        "verification": result, "formal_runtime_seconds": summary["runtime_seconds"],
        "phase_crossing_and_coupling_checked": True,
        "claim": "infinite-horizon 0.12-output-error upper certificate on a coupled nonlinear ReLU network",
        "limitations": "minimum state count remains between 25 and 100; the benchmark is constructed rather than trained",
    }
    save_json(OUTPUT, acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

