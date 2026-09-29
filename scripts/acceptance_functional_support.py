"""Formal exact-region support acceptance on frozen networks and controls."""
from __future__ import annotations
import time
from pathlib import Path

from ncd.continuous_generic_realization import benchmark_coupled_system
from ncd.continuous_separation import ContinuousReLUSystem, ReLUMLP
from ncd.functional_support import run_case, verify_case
from ncd.io import digest, read_json, save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "functional_support_global_v1"
STATUS = ROOT / "validation" / "functional_support_acceptance.json"


def control(weights, biases, head, offset):
    transition = ReLUMLP(
        (tuple(tuple(float(x) for x in row) for row in weights),
         tuple(tuple(float(x) for x in row) for row in head)),
        (tuple(float(x) for x in biases), (float(offset),)))
    observation = ReLUMLP((((1.0,),),), ((0.0,),))
    return ContinuousReLUSystem(1, 1, transition, observation)


def coupled_three_input_control():
    transition = ReLUMLP(
        (((1., 1., 1.), (1., 1., 1.), (1., 1., 1.)),
         ((.1, -.2, .1), (0., 0., 0.))),
        ((-1.33, -1.37, -1.41), (.1, .2)))
    observation = ReLUMLP((((1., 0.), (0., 1.)),), ((0., 0.),))
    return ContinuousReLUSystem(2, 1, transition, observation)


def cases():
    for family, seeds in (("learned_local_global_v1", (7201, 7202)),
                          ("trained_nonlinear_global_v1", (6101, 6102))):
        for seed in seeds:
            for dimension in (8, 32, 64, 128):
                path = ROOT / "runs" / family / f"seed_{seed}" / f"d_{dimension}" / "system.json"
                yield f"{family}_seed_{seed}_d_{dimension}", path
    for profile in range(4):
        path = ROOT / "runs" / "certified_continuous_scale_seed4701" / "profiles" / f"profile_{profile:03d}" / "system.json"
        yield f"affine_profile_{profile:03d}", path
    controls = {
        "coupled_phase_crossing_2d": benchmark_coupled_system(),
        "interior_tent": control(((1, 0), (1, 0), (1, 0)),
                                 (-.25, -.5, -.75), ((.1, -.2, .1),), .1),
        "narrow_tent": control(((1, 0), (1, 0), (1, 0)),
                                (-.55, -.60, -.65), ((.1, -.2, .1),), .1),
        "path_cancellation": control(((1, 0), (1, 0)),
                                     (0, 0), ((1, -1),), .1),
        "oblique_narrow_tent": control(((1, 1), (1, 1), (1, 1)),
                                       (-.83, -.87, -.91), ((.1, -.2, .1),), .1),
        "coupled_oblique_3input": coupled_three_input_control(),
        "hinge_identity_cancellation": control(
            ((1, 0), (-1, 0), (1, 0), (-1, 0)),
            (-.25, .25, -.75, .75), ((1, -1, -1, 1),), .1),
    }
    for name, system in controls.items():
        path = OUTPUT / "controls" / name / "system.json"
        save_json(path, system.to_dict())
        yield name, path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, model in cases():
        start = time.perf_counter()
        system = ContinuousReLUSystem.from_dict(read_json(model))
        output = OUTPUT / name
        if (output / "proposal.json").exists() and (output / "certificate.json").exists():
            result = verify_case(system, output)
        else:
            result = run_case(system, output)
        replay = verify_case(system, output)
        if result != replay or result["status"] != "certified":
            raise ValueError(f"Functional support incomplete: {name}")
        expected_regions = {
            "narrow_tent": (1, 0),
            "path_cancellation": (0, 1),
            "oblique_narrow_tent": (2, 0),
            "coupled_oblique_3input": (3, 0),
            "hinge_identity_cancellation": (0, 1),
        }
        if name in expected_regions and (
                result["region_witness_count"], result["functional_absence_count"]
                ) != expected_regions[name]:
            raise ValueError(f"Unexpected exact-region result: {name}")
        proposal = read_json(output / "proposal.json")
        records.append({"name": name,
                        "model": str(model.relative_to(ROOT)).replace("\\", "/"),
                        "state_dim": system.state_dim,
                        "action_dim": system.action_dim,
                        "query_calls": proposal["query_calls"],
                        "result": result,
                        "model_sha256": digest(model),
                        "proposal_sha256": digest(output / "proposal.json"),
                        "certificate_sha256": digest(output / "certificate.json"),
                        "elapsed_seconds": round(time.perf_counter() - start, 3)})
        print(name, result, flush=True)
    summary = {"schema": "ncd.functional-support-acceptance.v1",
               "case_count": len(records),
               "certified_count": sum(r["result"]["status"] == "certified" for r in records),
               "unresolved_count": sum(r["result"]["status"] == "unresolved" for r in records),
               "records": records}
    if (summary["case_count"], summary["certified_count"],
            summary["unresolved_count"]) != (27, 27, 0):
        raise ValueError("Incomplete formal acceptance")
    save_json(OUTPUT / "summary.json", summary)
    save_json(STATUS, summary)
    print("acceptance", summary["case_count"], summary["certified_count"], flush=True)


if __name__ == "__main__":
    main()
