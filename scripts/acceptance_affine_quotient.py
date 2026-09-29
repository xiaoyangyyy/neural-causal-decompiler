"""Formal exact affine behavioral quotient acceptance and controlled ablation."""
from copy import deepcopy
from pathlib import Path

from ncd.affine_observability import _affine_network, run_case, verify_case
from ncd.automatic_grid_realization import verify_case as verify_grid_case
from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "affine_quotient_global_v1"
STATUS = ROOT / "validation" / "affine_quotient_acceptance.json"
PROFILES = ROOT / "runs" / "certified_continuous_scale_seed4701" / "profiles"


def ablated_profile(original):
    source = original.to_dict()
    modified = deepcopy(source)
    d, u = original.state_dim, original.action_dim
    first = modified["transition"]["weights"][0]
    second = modified["transition"]["weights"][1]
    if (len(first) <= d - 1 or
            first[d - 1] != [float(i == d - 1) for i in range(d + u)] or
            modified["transition"]["biases"][0][d - 1] != 1.0):
        raise ValueError("Frozen model lacks the declared wrap-edge hidden unit")
    weight = second[0][d - 1]
    if weight == 0:
        raise ValueError("Frozen wrap edge already absent")
    first.append([0.0] * (d + u))
    modified["transition"]["biases"][0].append(1.0)
    for i, row in enumerate(second):
        row.append(weight if i == 0 else 0.0)
    second[0][d - 1] = 0.0
    ablated = ContinuousReLUSystem.from_dict(modified)
    before, before_bias, _ = _affine_network(original.transition)
    after, after_bias, _ = _affine_network(ablated.transition)
    changed = [(i, j) for i in range(d) for j in range(d + u)
               if before[i][j] != after[i][j]]
    if changed != [(0, d - 1)] or before_bias != after_bias or after[0][d - 1] != 0:
        raise ValueError("Ablation changed more than one effective affine edge")
    return ablated


def mixed_control(constant_output=False):
    transition = ReLUMLP((((.2, .2, .1), (.1, .1, .2)),), ((.1, .2),))
    observation = (ReLUMLP((((0., 0.),),), ((.7,),)) if constant_output
                   else ReLUMLP((((.5, .5),),), ((0.,),)))
    return ContinuousReLUSystem(2, 1, transition, observation)


def cases():
    for index, d in enumerate((8, 32, 64, 128)):
        yield f"frozen_affine_d{d}", PROFILES / f"profile_{index:03d}" / "system.json", d, "cyclic-triangular-delayed-observation"
    original = ContinuousReLUSystem.from_dict(read_json(
        PROFILES / "profile_003" / "system.json"))
    controls = {
        "ablated_affine_d128": (ablated_profile(original), 4, "closed-coordinate-projection"),
        "mixed_sum_d2": (mixed_control(), 1, "exact-observability-row-space"),
        "constant_output_d2": (mixed_control(True), 0, "constant-observation"),
    }
    for name, (system, dimension, proof) in controls.items():
        model = OUTPUT / "controls" / name / "system.json"
        save_json(model, system.to_dict())
        yield name, model, dimension, proof
    nonlinear = ROOT / "runs" / "functional_support_global_v1" / "controls" / "narrow_tent" / "system.json"
    yield "nonlinear_phase_boundary", nonlinear, None, None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, model, dimension, proof in cases():
        system = ContinuousReLUSystem.from_dict(read_json(model))
        target = OUTPUT / name
        result = run_case(system, target)
        if verify_case(system, target) != result:
            raise ValueError(f"Certificate replay mismatch: {name}")
        if result["quotient_dim"] != dimension or result["proof"] != proof:
            raise ValueError(f"Unexpected behavioral quotient: {name}")
        if (result["status"] == "certified") != (dimension is not None):
            raise ValueError(f"Unexpected quotient status: {name}")
        records.append({
            "name": name,
            "model": str(model.relative_to(ROOT)).replace("\\", "/"),
            "state_dim": system.state_dim,
            "action_dim": system.action_dim,
            "result": result,
            "model_sha256": digest(model),
            "certificate_sha256": digest(target / "certificate.json"),
        })
        print(name, result, flush=True)
    frozen128 = ContinuousReLUSystem.from_dict(read_json(
        PROFILES / "profile_003" / "system.json"))
    approximate = verify_grid_case(
        frozen128, ROOT / "runs" / "automatic_grid_global_v1" / "affine_profile_003")
    if (approximate["certificate_status"] != "certified"
            or approximate["lower_bound"] != 81
            or approximate["upper_bound"] != "40824"):
        raise ValueError("Prior approximate realization does not replay")
    summary = {
        "schema": "ncd.affine-quotient-acceptance.v1",
        "case_count": len(records),
        "certified_count": sum(r["result"]["status"] == "certified" for r in records),
        "unresolved_count": sum(r["result"]["status"] == "unresolved" for r in records),
        "records": records,
        "exact_vs_approximate": {
            "model": "profile_003",
            "exact_quotient_dimension": 128,
            "epsilon": "0.17",
            "finite_state_lower": approximate["lower_bound"],
            "finite_state_upper": approximate["upper_bound"],
            "delayed_witness_horizon": 124,
        },
    }
    if (summary["case_count"], summary["certified_count"],
            summary["unresolved_count"]) != (8, 7, 1):
        raise ValueError("Unexpected acceptance totals")
    save_json(OUTPUT / "summary.json", summary)
    save_json(STATUS, summary)
    print("acceptance", summary["case_count"], summary["certified_count"], flush=True)


if __name__ == "__main__":
    main()
