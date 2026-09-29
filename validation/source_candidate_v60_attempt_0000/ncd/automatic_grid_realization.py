"""Automatic finite-grid proposals from frozen ReLU influence, exact replay proof.

The float search is intentionally untrusted.  Only the exact-rational
weighted-realization checker can assign a certified upper bound.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np

from .continuous_compositional_realization import certify_weighted, verify_weighted
from .continuous_separation import ContinuousReLUSystem, ReLUMLP, _digest
from .io import read_json, save_json


def absolute_influence(network: ReLUMLP) -> np.ndarray:
    """Global coordinatewise ReLU Lipschitz matrix, used for proposals."""
    result = np.eye(network.input_dim)
    for weight in network.weights:
        result = np.abs(np.asarray(weight, dtype=float)) @ result
    return result


def propose_grid(system: ContinuousReLUSystem, *, epsilon: str = "0.17",
                 action_bins: int = 128, max_bin: int = 32,
                 reserve_fraction: float = 0.98) -> dict:
    if (float(epsilon) <= 0 or action_bins < 1 or max_bin < 1
            or not 0 < reserve_fraction < 1):
        raise ValueError("Invalid automatic-grid settings")
    d = system.state_dim
    transition = absolute_influence(system.transition)
    state_influence = transition[:, :d]
    control_influence = transition[:, d:]
    observation = absolute_influence(system.observation)
    spectral_radius = float(max(abs(np.linalg.eigvals(state_influence))))
    base = {
        "schema": "ncd.automatic-grid-proposal.v1",
        "system_sha256": _digest(system.to_dict()),
        "epsilon": str(epsilon), "action_bins": action_bins,
        "max_bin": max_bin, "reserve_fraction": reserve_fraction,
        "spectral_radius": spectral_radius,
    }
    if not math.isfinite(spectral_radius) or spectral_radius >= 1 - 1e-9:
        return {**base, "status": "unresolved",
                "reason": "absolute-influence dynamics not contractive"}
    transfer = np.linalg.solve(np.eye(d) - state_influence, np.eye(d))
    if not np.isfinite(transfer).all() or np.min(transfer) < -1e-8:
        return {**base, "status": "unresolved",
                "reason": "influence transfer is numerically unresolved"}
    transfer = np.maximum(transfer, 0.0)
    output_impact = observation @ transfer
    control_error = control_influence.sum(axis=1) / (2 * action_bins)
    base_output = output_impact @ control_error
    target = float(epsilon) * reserve_fraction
    if np.max(base_output) >= target:
        return {**base, "status": "unresolved",
                "reason": "continuous-action partition exceeds reserved output budget"}

    bins = np.ones(d, dtype=int)
    output = base_output + output_impact @ (1 / (2 * bins))
    steps: list[int] = []
    while np.max(output) > target:
        deficits = np.maximum(output - target, 0.0)
        best = None
        for j in range(d):
            if bins[j] >= max_bin:
                continue
            reduction = (1 / (2 * bins[j]) - 1 / (2 * (bins[j] + 1))) * output_impact[:, j]
            gain = float(np.minimum(deficits, reduction).sum())
            score = gain / math.log((bins[j] + 1) / bins[j])
            if best is None or score > best[0]:
                best = (score, j, reduction)
        if best is None or best[0] <= 0:
            return {**base, "status": "unresolved",
                    "reason": "maximum state-bin count reached before output closure"}
        _, coordinate, reduction = best
        bins[coordinate] += 1
        output -= reduction
        steps.append(int(coordinate))

    for coordinate in reversed(steps):
        if bins[coordinate] <= 1:
            continue
        added = ((1 / (2 * (bins[coordinate] - 1)) - 1 / (2 * bins[coordinate]))
                 * output_impact[:, coordinate])
        if np.max(output + added) <= target:
            bins[coordinate] -= 1
            output += added

    radius = transfer @ (control_error + 1 / (2 * bins))
    radius = np.ceil(radius * 1.005 * 1e8) / 1e8
    if not np.isfinite(radius).all() or np.min(radius) <= 0:
        return {**base, "status": "unresolved",
                "reason": "nonfinite proposed relation radius"}
    return {
        **base, "status": "candidate",
        "coordinate_bins": [int(x) for x in bins],
        "coordinate_radii": [f"{x:.8f}" for x in radius],
        "greedy_increment_trace": steps,
        "candidate_state_count": str(math.prod(int(x) for x in bins)),
    }


def run_case(system: ContinuousReLUSystem, output: Path, *,
             epsilon: str = "0.17", action_bins: int = 128,
             max_bin: int = 32, packing_axes: int = 4) -> dict:
    proposal = propose_grid(
        system, epsilon=epsilon, action_bins=action_bins, max_bin=max_bin)
    if proposal["status"] != "candidate":
        raise RuntimeError(f"Automatic grid unresolved: {proposal['reason']}")
    certificate = certify_weighted(
        system, tuple(proposal["coordinate_bins"]),
        tuple(proposal["coordinate_radii"]),
        action_bins=action_bins, epsilon=epsilon,
        packing_axes=packing_axes)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "proposal.json", proposal)
    save_json(output / "certificate.json", certificate)
    record = {
        "schema": "ncd.automatic-grid-case.v1",
        "system_sha256": proposal["system_sha256"],
        "proposal_state": proposal["status"],
        "certificate_status": certificate["status"],
        "lower_bound": certificate["lower_bound"],
        "upper_bound": certificate["upper_bound"],
    }
    save_json(output / "record.json", record)
    return record


def verify_case(system: ContinuousReLUSystem, output: Path) -> dict:
    stored = read_json(output / "proposal.json")
    regenerated = propose_grid(
        system, epsilon=stored["epsilon"],
        action_bins=stored["action_bins"], max_bin=stored["max_bin"],
        reserve_fraction=stored["reserve_fraction"])
    if stored != regenerated:
        raise ValueError("Automatic grid search replay mismatch")
    certificate = read_json(output / "certificate.json")
    verify_weighted(system, certificate)
    expected = {
        "schema": "ncd.automatic-grid-case.v1",
        "system_sha256": stored["system_sha256"],
        "proposal_state": stored["status"],
        "certificate_status": certificate["status"],
        "lower_bound": certificate["lower_bound"],
        "upper_bound": certificate["upper_bound"],
    }
    if read_json(output / "record.json") != expected:
        raise ValueError("Automatic grid case replay mismatch")
    return expected


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--epsilon", default="0.17")
    parser.add_argument("--packing-axes", type=int, default=4)
    args = parser.parse_args()
    system = ContinuousReLUSystem.from_dict(read_json(args.system))
    result = (verify_case(system, args.output) if args.verify
              else run_case(system, args.output, epsilon=args.epsilon,
                            packing_axes=args.packing_axes))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

