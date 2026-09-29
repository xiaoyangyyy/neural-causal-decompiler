"""Exact-rational, succinct infinite-horizon grid certificates for ReLU systems."""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product
from pathlib import Path

from .continuous_separation import ContinuousReLUSystem, ReLUMLP, _digest
from .io import read_json, save_json


def interval(net: ReLUMLP, low: list[Q], high: list[Q]):
    """Exact image enclosure for a rational box and serialized float weights."""
    for layer, (weights, biases) in enumerate(zip(net.weights, net.biases)):
        next_low, next_high = [], []
        for row, bias in zip(weights, biases):
            lo = hi = Q(bias)
            for coefficient, left, right in zip(row, low, high):
                w = Q(coefficient)
                lo += w * (left if w >= 0 else right)
                hi += w * (right if w >= 0 else left)
            next_low.append(lo)
            next_high.append(hi)
        low, high = next_low, next_high
        if layer + 1 < len(net.weights):
            low = [max(Q(0), x) for x in low]
            high = [max(Q(0), x) for x in high]
    return low, high


def sensitivity(net: ReLUMLP, radii: list[Q]) -> list[Q]:
    """Coordinatewise global ReLU Lipschitz bound from absolute weight matrices."""
    for weights in net.weights:
        radii = [sum((abs(Q(w)) * r for w, r in zip(row, radii)), Q(0))
                 for row in weights]
    return radii


def value(net: ReLUMLP, point: list[Q]) -> list[Q]:
    low, high = interval(net, point, point)
    assert low == high
    return low


def _quantize(x: Q, bins: int) -> int:
    return min(max(x.numerator * bins // x.denominator, 0), bins - 1)


def machine_initial(point: tuple[float, ...], bins: int) -> tuple[int, ...]:
    if bins < 2 or any(not 0 <= x <= 1 for x in point):
        raise ValueError("Invalid initial state or bins")
    return tuple(_quantize(Q(x), bins) for x in point)


def machine_step(system: ContinuousReLUSystem, state: tuple[int, ...],
                 action: tuple[float, ...], bins: int, action_bins: int) -> tuple[int, ...]:
    if (bins < 2 or action_bins < 1 or len(state) != system.state_dim
            or any(not 0 <= q < bins for q in state)
            or len(action) != system.action_dim
            or any(not 0 <= a <= 1 for a in action)):
        raise ValueError("Invalid abstract state or action")
    center = [Q(2 * q + 1, 2 * bins) for q in state]
    controls = [Q(2 * _quantize(Q(a), action_bins) + 1, 2 * action_bins)
                for a in action]
    return tuple(_quantize(x, bins) for x in value(system.transition, center + controls))


def machine_output(system: ContinuousReLUSystem, state: tuple[int, ...],
                   bins: int) -> tuple[Q, ...]:
    if len(state) != system.state_dim or any(not 0 <= q < bins for q in state):
        raise ValueError("Invalid abstract state")
    return tuple(value(system.observation,
                       [Q(2 * q + 1, 2 * bins) for q in state]))


def _recompute(system: ContinuousReLUSystem, bins: int, action_bins: int,
               radius: Q, epsilon: Q, packing_axes: int) -> dict:
    if (bins < 2 or action_bins < 1 or radius <= 0 or epsilon <= 0
            or not 0 <= packing_axes <= system.state_dim):
        raise ValueError("Invalid certificate settings")
    d, u = system.state_dim, system.action_dim
    image_low, image_high = interval(
        system.transition, [Q(0)] * (d + u), [Q(1)] * (d + u))
    invariant = all(0 <= lo <= hi <= 1 for lo, hi in zip(image_low, image_high))
    step_bound = [x + Q(1, 2 * bins) for x in sensitivity(
        system.transition, [radius] * d + [Q(1, 2 * action_bins)] * u)]
    output_bound = sensitivity(system.observation, [radius] * d)
    certified = (invariant and Q(1, 2 * bins) <= radius
                 and all(x <= radius for x in step_bound)
                 and all(x <= epsilon for x in output_bound))
    points = [list(coords) + [Q(0)] * (d - packing_axes)
              for coords in product((Q(0), Q(1, 2), Q(1)), repeat=packing_axes)] if packing_axes else []
    outputs = [value(system.observation, point) for point in points]
    packing = all(max(abs(a - b) for a, b in zip(left, right)) > 2 * epsilon
                  for i, left in enumerate(outputs) for right in outputs[i + 1:])
    return {
        "schema": "ncd.compositional-realization.v1",
        "system_sha256": _digest(system.to_dict()),
        "state_dim": d, "action_dim": u,
        "state_bins": bins, "action_bins": action_bins,
        "relation_radius": str(radius), "epsilon": str(epsilon),
        "domain": "full unit state and action cubes",
        "transition_image_low": [str(x) for x in image_low],
        "transition_image_high": [str(x) for x in image_high],
        "transition_error_upper": [str(x) for x in step_bound],
        "observation_error_upper": [str(x) for x in output_bound],
        "packing_axes": packing_axes,
        "lower_bound": len(points) if packing and points else None,
        "upper_bound": str(bins ** d) if certified else None,
        "status": "certified" if certified else "unresolved",
    }


def certify_compositional(system: ContinuousReLUSystem, *, bins: int = 14,
                          action_bins: int = 14, radius: str = "0.16",
                          epsilon: str = "0.17", packing_axes: int = 4) -> dict:
    certificate = _recompute(system, bins, action_bins, Q(radius), Q(epsilon), packing_axes)
    verify_compositional(system, certificate)
    return certificate


def verify_compositional(system: ContinuousReLUSystem, certificate: dict) -> dict:
    if certificate.get("schema") != "ncd.compositional-realization.v1":
        raise ValueError("Unsupported compositional certificate")
    expected = _recompute(system, int(certificate["state_bins"]),
                          int(certificate["action_bins"]),
                          Q(certificate["relation_radius"]), Q(certificate["epsilon"]),
                          int(certificate["packing_axes"]))
    if certificate != expected:
        raise ValueError("Certificate does not match frozen network")
    return {"status": expected["status"], "lower_bound": expected["lower_bound"],
            "upper_bound": expected["upper_bound"]}


def certify_frozen_scale_models(source: Path, output: Path) -> dict:
    """Certify previously frozen trained models without any refitting."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for model_path in sorted((source / "profiles").glob("profile_*/system.json")):
        system = ContinuousReLUSystem.from_dict(read_json(model_path))
        certificate = certify_compositional(system)
        if certificate["status"] != "certified":
            raise RuntimeError(f"Global realization unresolved: {model_path}")
        target = output / model_path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        save_json(target / "certificate.json", certificate)
        records.append({"profile": model_path.parent.name,
                        "model": str(model_path.relative_to(source)).replace("\\", "/"),
                        "system_sha256": certificate["system_sha256"],
                        "state_dim": system.state_dim, "action_dim": system.action_dim,
                        "lower_bound": certificate["lower_bound"],
                        "upper_bound": certificate["upper_bound"]})
    if not records:
        raise ValueError("No frozen trained models found")
    summary = {"schema": "ncd.compositional-scale-run.v1",
               "records": records, "all_certified": True}
    save_json(output / "summary.json", summary)
    return summary




def verify_frozen_scale_models(source: Path, output: Path) -> dict:
    """Replay all certificates against the frozen model files and summary."""
    stored = read_json(output / "summary.json")
    records = []
    for model_path in sorted((source / "profiles").glob("profile_*/system.json")):
        system = ContinuousReLUSystem.from_dict(read_json(model_path))
        certificate = read_json(output / model_path.parent.name / "certificate.json")
        check = verify_compositional(system, certificate)
        if check["status"] != "certified":
            raise ValueError("Stored profile is unresolved")
        records.append({"profile": model_path.parent.name,
                        "model": str(model_path.relative_to(source)).replace("\\", "/"),
                        "system_sha256": certificate["system_sha256"],
                        "state_dim": system.state_dim, "action_dim": system.action_dim,
                        "lower_bound": certificate["lower_bound"],
                        "upper_bound": certificate["upper_bound"]})
    expected = {"schema": "ncd.compositional-scale-run.v1",
                "records": records, "all_certified": True}
    if not records or stored != expected:
        raise ValueError("Frozen-model certificate summary mismatch")
    return expected





def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()
    if args.weighted:
        operation = verify_weighted_scale_models if args.verify else certify_weighted_scale_models
    else:
        operation = verify_frozen_scale_models if args.verify else certify_frozen_scale_models
    result = operation(args.source, args.output)
    print(f"{len(result['records'])} frozen models certified and replayable")


def weighted_machine_initial(point: tuple[float, ...],
                             bins: tuple[int, ...]) -> tuple[int, ...]:
    if len(point) != len(bins) or any(n < 1 for n in bins) or any(not 0 <= x <= 1 for x in point):
        raise ValueError("Invalid initial state or coordinate bins")
    return tuple(_quantize(Q(x), n) for x, n in zip(point, bins))


def weighted_machine_step(system: ContinuousReLUSystem, state: tuple[int, ...],
                          action: tuple[float, ...], bins: tuple[int, ...],
                          action_bins: int) -> tuple[int, ...]:
    if (len(bins) != system.state_dim or len(state) != system.state_dim
            or any(n < 1 or not 0 <= q < n for q, n in zip(state, bins))
            or len(action) != system.action_dim
            or any(not 0 <= a <= 1 for a in action) or action_bins < 1):
        raise ValueError("Invalid weighted abstract state or action")
    center = [Q(2 * q + 1, 2 * n) for q, n in zip(state, bins)]
    controls = [Q(2 * _quantize(Q(a), action_bins) + 1, 2 * action_bins)
                for a in action]
    image = value(system.transition, center + controls)
    return tuple(_quantize(x, n) for x, n in zip(image, bins))


def weighted_machine_output(system: ContinuousReLUSystem, state: tuple[int, ...],
                            bins: tuple[int, ...]) -> tuple[Q, ...]:
    if len(bins) != system.state_dim or len(state) != system.state_dim or any(
            n < 1 or not 0 <= q < n for q, n in zip(state, bins)):
        raise ValueError("Invalid weighted abstract state")
    return tuple(value(system.observation,
                       [Q(2 * q + 1, 2 * n) for q, n in zip(state, bins)]))


def _weighted_recompute(system: ContinuousReLUSystem, bins: tuple[int, ...],
                        radii: tuple[Q, ...], action_bins: int,
                        epsilon: Q, packing_axes: int) -> dict:
    d, u = system.state_dim, system.action_dim
    if (len(bins) != d or len(radii) != d or any(n < 1 for n in bins)
            or any(r <= 0 for r in radii) or action_bins < 1 or epsilon <= 0
            or not 0 <= packing_axes <= d):
        raise ValueError("Invalid weighted certificate settings")
    low, high = interval(system.transition, [Q(0)] * (d + u), [Q(1)] * (d + u))
    invariant = all(0 <= a <= b <= 1 for a, b in zip(low, high))
    step_bound = [delta + Q(1, 2 * n) for delta, n in zip(
        sensitivity(system.transition, list(radii) + [Q(1, 2 * action_bins)] * u), bins)]
    output_bound = sensitivity(system.observation, list(radii))
    certified = (invariant
                 and all(Q(1, 2 * n) <= r for n, r in zip(bins, radii))
                 and all(delta <= r for delta, r in zip(step_bound, radii))
                 and all(delta <= epsilon for delta in output_bound))
    points = [list(coords) + [Q(0)] * (d - packing_axes)
              for coords in product((Q(0), Q(1, 2), Q(1)), repeat=packing_axes)] if packing_axes else []
    outputs = [value(system.observation, point) for point in points]
    packing = all(max(abs(a - b) for a, b in zip(left, right)) > 2 * epsilon
                  for i, left in enumerate(outputs) for right in outputs[i + 1:])
    size = 1
    for n in bins:
        size *= n
    return {
        "schema": "ncd.weighted-compositional-realization.v1",
        "system_sha256": _digest(system.to_dict()),
        "state_dim": d, "action_dim": u,
        "coordinate_bins": list(bins),
        "coordinate_radii": [str(x) for x in radii],
        "action_bins": action_bins, "epsilon": str(epsilon),
        "domain": "full unit state and action cubes",
        "transition_image_low": [str(x) for x in low],
        "transition_image_high": [str(x) for x in high],
        "transition_error_upper": [str(x) for x in step_bound],
        "observation_error_upper": [str(x) for x in output_bound],
        "packing_axes": packing_axes,
        "lower_bound": len(points) if packing and points else None,
        "upper_bound": str(size) if certified else None,
        "status": "certified" if certified else "unresolved",
    }


def certify_weighted(system: ContinuousReLUSystem, bins: tuple[int, ...],
                     radii: tuple[str, ...], *, action_bins: int = 64,
                     epsilon: str = "0.17", packing_axes: int = 4) -> dict:
    certificate = _weighted_recompute(
        system, tuple(bins), tuple(Q(x) for x in radii),
        action_bins, Q(epsilon), packing_axes)
    verify_weighted(system, certificate)
    return certificate


def verify_weighted(system: ContinuousReLUSystem, certificate: dict) -> dict:
    if certificate.get("schema") != "ncd.weighted-compositional-realization.v1":
        raise ValueError("Unsupported weighted certificate")
    expected = _weighted_recompute(
        system, tuple(certificate["coordinate_bins"]),
        tuple(Q(x) for x in certificate["coordinate_radii"]),
        int(certificate["action_bins"]), Q(certificate["epsilon"]),
        int(certificate["packing_axes"]))
    if certificate != expected:
        raise ValueError("Weighted certificate does not match frozen network")
    return {"status": expected["status"], "lower_bound": expected["lower_bound"],
            "upper_bound": expected["upper_bound"]}


def certify_weighted_scale_models(source: Path, output: Path) -> dict:
    """Test a sparse, observability-guided candidate on frozen models."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for model_path in sorted((source / "profiles").glob("profile_*/system.json")):
        system = ContinuousReLUSystem.from_dict(read_json(model_path))
        d = system.state_dim
        if d < 5:
            raise ValueError("Weighted scale profile requires five resolved coordinates")
        bins = (12, 6, 6, 6) + (1,) * (d - 5) + (12,)
        radii = ("0.17",) * 4 + ("1.01",) * (d - 5) + ("0.17",)
        certificate = certify_weighted(system, bins, radii, action_bins=128)
        if certificate["status"] != "certified":
            raise RuntimeError(f"Weighted realization unresolved: {model_path}")
        target = output / model_path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        save_json(target / "certificate.json", certificate)
        records.append({"profile": model_path.parent.name,
                        "model": str(model_path.relative_to(source)).replace("\\", "/"),
                        "system_sha256": certificate["system_sha256"],
                        "state_dim": d, "action_dim": system.action_dim,
                        "lower_bound": certificate["lower_bound"],
                        "upper_bound": certificate["upper_bound"]})
    if not records:
        raise ValueError("No frozen trained models found")
    summary = {"schema": "ncd.weighted-compositional-scale-run.v1",
               "records": records, "all_certified": True}
    save_json(output / "summary.json", summary)
    return summary


def verify_weighted_scale_models(source: Path, output: Path) -> dict:
    stored = read_json(output / "summary.json")
    records = []
    for model_path in sorted((source / "profiles").glob("profile_*/system.json")):
        system = ContinuousReLUSystem.from_dict(read_json(model_path))
        certificate = read_json(output / model_path.parent.name / "certificate.json")
        if verify_weighted(system, certificate)["status"] != "certified":
            raise ValueError("Stored weighted certificate is unresolved")
        records.append({"profile": model_path.parent.name,
                        "model": str(model_path.relative_to(source)).replace("\\", "/"),
                        "system_sha256": certificate["system_sha256"],
                        "state_dim": system.state_dim, "action_dim": system.action_dim,
                        "lower_bound": certificate["lower_bound"],
                        "upper_bound": certificate["upper_bound"]})
    expected = {"schema": "ncd.weighted-compositional-scale-run.v1",
                "records": records, "all_certified": True}
    if not records or stored != expected:
        raise ValueError("Weighted certificate summary mismatch")
    return expected








if __name__ == "__main__":
    main()
