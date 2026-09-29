"""Frozen automatic-grid synthesis study on trained and coupled ReLU systems."""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path
import argparse
import json

from ncd.automatic_grid_realization import run_case, verify_case
from ncd.continuous_compositional_realization import value
from ncd.continuous_generic_realization import benchmark_coupled_system
from ncd.continuous_separation import ContinuousReLUSystem, _digest
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "automatic_grid_global_v1"
ACCEPTANCE = ROOT / "validation" / "automatic_grid_global_acceptance.json"
PROTOCOL = {
    "epsilon_trained": "0.17",
    "epsilon_coupled": "0.12",
    "action_bins": 128,
    "max_state_bins_per_coordinate": 32,
    "search_output_reserve_fraction": 0.98,
    "candidate_radius_scale": 1.005,
    "learned_upper_gate": 2250,
    "fixed_dictionary_upper_gate": 2250,
    "affine_upper_gate": 100000,
    "coupled_upper_gate": 64,
    "required_learned_strict_improvements": 4,
}


def specifications() -> list[dict]:
    cases = []
    for seed in (7201, 7202):
        for d in (8, 32, 64, 128):
            cases.append({
                "name": f"learned_{seed}_d{d}", "family": "learned_local",
                "model": str(Path("runs") / "learned_local_global_v1"
                             / f"seed_{seed}" / f"d_{d}" / "system.json"),
                "dimension": d, "epsilon": "0.17", "packing_axes": 4,
            })
    for seed in (6101, 6102):
        for d in (8, 32, 64, 128):
            cases.append({
                "name": f"fixed_{seed}_d{d}", "family": "fixed_dictionary",
                "model": str(Path("runs") / "trained_nonlinear_global_v1"
                             / f"seed_{seed}" / f"d_{d}" / "system.json"),
                "dimension": d, "epsilon": "0.17", "packing_axes": 4,
            })
    for index, d in enumerate((8, 32, 64, 128)):
        cases.append({
            "name": f"affine_profile_{index:03d}", "family": "trained_affine",
            "model": str(Path("runs") / "certified_continuous_scale_seed4701"
                         / "profiles" / f"profile_{index:03d}" / "system.json"),
            "dimension": d, "epsilon": "0.17", "packing_axes": 4,
        })
    cases.append({
        "name": "coupled_phase_crossing_2d", "family": "coupled_benchmark",
        "model": "constructed:benchmark_coupled_system",
        "dimension": 2, "epsilon": "0.12", "packing_axes": 2,
    })
    return cases


def model_for(spec: dict) -> ContinuousReLUSystem:
    if spec["family"] == "coupled_benchmark":
        return benchmark_coupled_system()
    return ContinuousReLUSystem.from_dict(read_json(ROOT / spec["model"]))


def coupled_packing(system: ContinuousReLUSystem) -> tuple[int, str]:
    levels = (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1))
    points = [list(x) for x in product(levels, repeat=2)]
    outputs = [value(system.observation, point) for point in points]
    distance = min(max(abs(a - b) for a, b in zip(outputs[i], outputs[j]))
                   for i, j in combinations(range(len(points)), 2))
    if distance <= Q(24, 100):
        raise ValueError("Coupled benchmark packing does not separate 25 points")
    return 25, str(distance)


def assess(spec: dict, system: ContinuousReLUSystem, record: dict) -> dict:
    proposal = read_json(OUTPUT / spec["name"] / "proposal.json")
    coupled = spec["family"] == "coupled_benchmark"
    extra_lower, separation = coupled_packing(system) if coupled else (None, None)
    upper = int(record["upper_bound"] or 0)
    upper_limit = {
        "learned_local": PROTOCOL["learned_upper_gate"],
        "fixed_dictionary": PROTOCOL["fixed_dictionary_upper_gate"],
        "trained_affine": PROTOCOL["affine_upper_gate"],
        "coupled_benchmark": PROTOCOL["coupled_upper_gate"],
    }[spec["family"]]
    lower = extra_lower if coupled else record["lower_bound"]
    gates = {
        "exact_certificate": record["certificate_status"] == "certified",
        "upper": 0 < upper <= upper_limit,
        "lower": lower == (25 if coupled else 81),
        "automatic_candidate": proposal["status"] == "candidate",
    }
    return {
        **spec,
        "system_sha256": _digest(system.to_dict()),
        "proposal_sha256": digest(OUTPUT / spec["name"] / "proposal.json"),
        "certificate_sha256": digest(OUTPUT / spec["name"] / "certificate.json"),
        "coordinate_bins": proposal["coordinate_bins"],
        "certificate_lower_bound": record["lower_bound"],
        "reported_lower_bound": lower,
        "coupled_packing_min_separation": separation,
        "upper_bound": record["upper_bound"],
        "gates": gates,
        "accepted": all(gates.values()),
    }


def _summary(cases: list[dict]) -> dict:
    improvements = sum(c["family"] == "learned_local" and
                       int(c["upper_bound"]) < 2250 for c in cases)
    return {
        "schema": "ncd.automatic-grid-global-study.v1",
        "protocol": PROTOCOL,
        "cases": cases,
        "learned_strict_improvements": improvements,
        "accepted": (all(c["accepted"] for c in cases) and
                     improvements >= PROTOCOL["required_learned_strict_improvements"]),
    }


def generate() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cases = []
    for spec in specifications():
        system = model_for(spec)
        case_dir = OUTPUT / spec["name"]
        if spec["family"] == "coupled_benchmark":
            case_dir.mkdir(parents=True, exist_ok=True)
            save_json(case_dir / "system.json", system.to_dict())
        record = run_case(system, case_dir,
                          epsilon=spec["epsilon"],
                          action_bins=PROTOCOL["action_bins"],
                          max_bin=PROTOCOL["max_state_bins_per_coordinate"],
                          packing_axes=spec["packing_axes"])
        cases.append(assess(spec, system, record))
        print(f"{spec['name']} upper={record['upper_bound']} accepted={cases[-1]['accepted']}",
              flush=True)
    summary = _summary(cases)
    save_json(OUTPUT / "summary.json", summary)
    return summary


def verify() -> dict:
    cases = []
    for spec in specifications():
        system = model_for(spec)
        case_dir = OUTPUT / spec["name"]
        if (spec["family"] == "coupled_benchmark" and
                read_json(case_dir / "system.json") != system.to_dict()):
            raise ValueError("Coupled benchmark model replay mismatch")
        record = verify_case(system, case_dir)
        cases.append(assess(spec, system, record))
        print(f"replayed {spec['name']}", flush=True)
    expected = _summary(cases)
    if read_json(OUTPUT / "summary.json") != expected:
        raise ValueError("Automatic-grid study summary replay mismatch")
    status = {
        "schema": "ncd.automatic-grid-global-acceptance.v1",
        "state": "verified" if expected["accepted"] else "failed",
        "cases": len(cases),
        "all_proposals_and_exact_certificates_replayed": True,
        "all_frozen_gates_passed": expected["accepted"],
        "learned_strict_improvements": expected["learned_strict_improvements"],
        "summary_sha256": digest(OUTPUT / "summary.json"),
    }
    save_json(ACCEPTANCE, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = generate() if args.generate else verify()
    print(json.dumps({"accepted": result["accepted"]} if args.generate
                     else {"state": result["state"]}))


if __name__ == "__main__":
    main()

