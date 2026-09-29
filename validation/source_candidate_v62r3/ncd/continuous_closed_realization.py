"""Inductive finite realization of the frozen separable continuous ReLU system."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
import json
import time
import numpy as np

from .continuous_cover import benchmark_cover_system, _parameter_count
from .continuous_separation import ContinuousReLUSystem, _digest, _network_interval, _abs_difference_bounds
from .io import digest, read_json, save_json


@dataclass
class ClosedRealizationConfig:
    seed: int = 10701
    state_dims: tuple[int, ...] = (1, 2)
    bins_per_dim: int = 10
    packing_bins: int = 5
    epsilon: float = 0.101

    @classmethod
    def quick(cls, seed: int = 10701) -> "ClosedRealizationConfig":
        return cls(seed=seed, state_dims=(1,))

    def validate(self) -> None:
        self.state_dims = tuple(int(x) for x in self.state_dims)
        self.bins_per_dim = int(self.bins_per_dim)
        self.packing_bins = int(self.packing_bins)
        self.epsilon = float(self.epsilon)
        if (not self.state_dims or any(d < 1 for d in self.state_dims)
                or self.bins_per_dim < 2 or self.packing_bins < 2
                or not np.isfinite(self.epsilon) or self.epsilon <= 0
                or 1 / (2 * self.bins_per_dim) > self.epsilon
                or 1 / (self.packing_bins - 1) <= 2 * self.epsilon):
            raise ValueError("Invalid realization configuration")


def _axis_proof(bins: int, epsilon: float) -> tuple[list[dict], list[dict]]:
    system = benchmark_cover_system(1)
    centers = [(i + 0.5) / bins for i in range(bins)]
    initial, transitions = [], []
    for i, center in enumerate(centers):
        low, high = i / bins, (i + 1) / bins
        output_low, output_high = _network_interval(
            system.observation, np.array([low]), np.array([high]))
        initial_bound = _abs_difference_bounds(
            output_low, output_high, np.array([center]), np.array([center]))[1]
        initial.append({"index": i, "low": low, "high": high,
                        "representative": center, "output_error_upper": initial_bound})
        cuts = sorted({(j / bins - 0.5 * center) / 0.4 for j in range(1, bins)
                       if 0 < (j / bins - 0.5 * center) / 0.4 < 1})
        boundaries = [0.0, *cuts, 1.0]
        segments = []
        for action_low, action_high in zip(boundaries[:-1], boundaries[1:]):
            nominal = float(system.step([center], [(action_low + action_high) / 2])[0])
            target = min(range(bins), key=lambda j: abs(nominal - centers[j]))
            state_low = max(0.0, center - epsilon)
            state_high = min(1.0, center + epsilon)
            next_low, next_high = _network_interval(
                system.transition, np.array([state_low, action_low]),
                np.array([state_high, action_high]))
            bound = _abs_difference_bounds(
                next_low, next_high, np.array([centers[target]]),
                np.array([centers[target]]))[1]
            segments.append({"action_low": action_low, "action_high": action_high,
                             "target": target, "next_error_upper": bound})
        transitions.append({"source": i, "segments": segments})
    return initial, transitions


def _packing(system: ContinuousReLUSystem, bins: int) -> tuple[list[list[float]], list[dict]]:
    axis = np.linspace(0.0, 1.0, bins).tolist()
    points = [list(p) for p in product(axis, repeat=system.state_dim)]
    outputs = [_network_interval(system.observation, np.array(p), np.array(p))
               for p in points]
    pairs = []
    for left, right in combinations(range(len(points)), 2):
        distance = _abs_difference_bounds(*outputs[left], *outputs[right])[0]
        pairs.append({"left": left, "right": right,
                      "initial_output_distance_lower": distance})
    return points, pairs


def certified_closed_realization(system: ContinuousReLUSystem, bins: int = 10,
                                 packing_bins: int = 5, epsilon: float = 0.101) -> dict:
    if system.to_dict() != benchmark_cover_system(system.state_dim).to_dict():
        raise ValueError("Factorized proof requires the frozen separable benchmark")
    ClosedRealizationConfig(state_dims=(system.state_dim,), bins_per_dim=bins,
                            packing_bins=packing_bins, epsilon=epsilon).validate()
    initial, transitions = _axis_proof(bins, epsilon)
    points, pairs = _packing(system, packing_bins)
    certificate = {
        "schema": "ncd.continuous-closed-realization.v1",
        "system_sha256": _digest(system.to_dict()),
        "domain_low": [0.0] * system.state_dim,
        "domain_high": [1.0] * system.state_dim,
        "action_lower": [0.0] * system.action_dim,
        "action_upper": [1.0] * system.action_dim,
        "epsilon": epsilon, "bins_per_dim": bins, "packing_bins": packing_bins,
        "axis_initial_cells": initial, "axis_transitions": transitions,
        "packing_points": points, "packing_pairs": pairs,
        "lower_bound": len(points), "upper_bound": bins ** system.state_dim,
        "minimal": len(points) == bins ** system.state_dim,
        "claim": "transition-closed epsilon simulation for all continuous action words and all times",
        "boundary": "frozen coordinate-separable ReLU system; size bounds need not match",
    }
    verify_closed_realization(system, certificate)
    return certificate


def verify_closed_realization(system: ContinuousReLUSystem, certificate: dict) -> dict:
    if certificate.get("schema") != "ncd.continuous-closed-realization.v1":
        raise ValueError("Unsupported closed realization")
    if (system.to_dict() != benchmark_cover_system(system.state_dim).to_dict()
            or certificate.get("system_sha256") != _digest(system.to_dict())):
        raise ValueError("Closed realization system mismatch")
    d = system.state_dim
    bins, packing_bins = int(certificate["bins_per_dim"]), int(certificate["packing_bins"])
    epsilon = float(certificate["epsilon"])
    ClosedRealizationConfig(state_dims=(d,), bins_per_dim=bins,
                            packing_bins=packing_bins, epsilon=epsilon).validate()
    if (certificate.get("domain_low") != [0.0] * d
            or certificate.get("domain_high") != [1.0] * d
            or certificate.get("action_lower") != [0.0] * d
            or certificate.get("action_upper") != [1.0] * d):
        raise ValueError("Closed realization domain mismatch")
    initial, transitions = _axis_proof(bins, epsilon)
    if certificate.get("axis_initial_cells") != initial:
        raise ValueError("Initial cells failed interval replay")
    if certificate.get("axis_transitions") != transitions:
        raise ValueError("Transition partition failed interval replay")
    if any(cell["output_error_upper"] > epsilon for cell in initial):
        raise ValueError("Initial relation fails")
    if any(s["next_error_upper"] > epsilon for entry in transitions
           for s in entry["segments"]):
        raise ValueError("Simulation relation is not forward invariant")
    points, pairs = _packing(system, packing_bins)
    if (certificate.get("packing_points") != points
            or certificate.get("packing_pairs") != pairs
            or any(p["initial_output_distance_lower"] <= 2 * epsilon for p in pairs)):
        raise ValueError("Packing lower bound fails")
    lower, upper = len(points), bins ** d
    if (certificate.get("lower_bound") != lower
            or certificate.get("upper_bound") != upper
            or certificate.get("minimal") != (lower == upper)):
        raise ValueError("Realization size bounds mismatch")
    return {"status": "verified", "state_dim": d,
            "initial_cells_verified": len(initial),
            "transition_segments_verified": sum(len(e["segments"]) for e in transitions),
            "packing_pairs_verified": len(pairs),
            "lower_bound": lower, "upper_bound": upper,
            "minimal": lower == upper, "horizon": "unbounded"}


def realize_initial(certificate: dict, state: tuple[float, ...]) -> tuple[int, ...]:
    d, bins = len(certificate["domain_low"]), certificate["bins_per_dim"]
    if len(state) != d or any(not np.isfinite(x) or x < 0 or x > 1 for x in state):
        raise ValueError("Initial state outside domain")
    return tuple(min(int(x * bins), bins - 1) for x in state)


def realize_step(certificate: dict, state: tuple[int, ...],
                 action: tuple[float, ...]) -> tuple[int, ...]:
    d, bins = len(certificate["domain_low"]), certificate["bins_per_dim"]
    if len(state) != d or len(action) != d:
        raise ValueError("Abstract state/action dimension mismatch")
    result = []
    for coordinate, control in zip(state, action):
        if not isinstance(coordinate, int) or not 0 <= coordinate < bins:
            raise ValueError("Invalid abstract state")
        if not np.isfinite(control) or not 0 <= control <= 1:
            raise ValueError("Action outside domain")
        segments = certificate["axis_transitions"][coordinate]["segments"]
        cuts = [s["action_high"] for s in segments[:-1]]
        result.append(segments[bisect_right(cuts, control)]["target"])
    return tuple(result)


def realization_output(certificate: dict, state: tuple[int, ...]) -> tuple[float, ...]:
    bins = certificate["bins_per_dim"]
    if len(state) != len(certificate["domain_low"]) or any(
            not isinstance(i, int) or not 0 <= i < bins for i in state):
        raise ValueError("Invalid abstract state")
    return tuple((i + 0.5) / bins for i in state)


def _profile(config: ClosedRealizationConfig, index: int) -> dict:
    d = config.state_dims[index]
    system = benchmark_cover_system(d)
    started = time.perf_counter()
    certificate = certified_closed_realization(
        system, config.bins_per_dim, config.packing_bins, config.epsilon)
    runtime = time.perf_counter() - started
    return {"profile": index, "seed": config.seed + index * 1009,
            "state_dim": d, "action_dim": d,
            "parameter_count": _parameter_count(system),
            "system": system.to_dict(), "certificate": certificate,
            "verification": verify_closed_realization(system, certificate),
            "runtime_seconds": runtime}


def _summary(profiles: list[dict]) -> dict:
    return {"schema": "ncd.continuous-closed-realization-summary.v1",
            "profiles": len(profiles),
            "largest_state_dim": max(p["state_dim"] for p in profiles),
            "largest_realization": max(p["certificate"]["upper_bound"] for p in profiles),
            "total_initial_cells_verified": sum(p["verification"]["initial_cells_verified"] for p in profiles),
            "total_transition_segments_verified": sum(p["verification"]["transition_segments_verified"] for p in profiles),
            "total_packing_pairs_verified": sum(p["verification"]["packing_pairs_verified"] for p in profiles),
            "scope": "uniform infinite-horizon epsilon simulation of frozen separable ReLU dynamics with continuous controls"}


def run_closed_realizations(output: Path, config: ClosedRealizationConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    started = time.perf_counter()
    profiles = []
    for index in range(len(config.state_dims)):
        profile = _profile(config, index)
        profiles.append(profile)
        directory = output / "profiles" / f"profile_{index:03d}"
        save_json(directory / "system.json", profile["system"])
        save_json(directory / "profile.json", profile)
    summary = _summary(profiles)
    summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    artifacts = {p.relative_to(output).as_posix(): digest(p)
                 for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    save_json(output / "manifest.json", {
        "schema": "ncd.continuous-closed-realization-manifest.v1",
        "artifacts": artifacts})
    return summary


def _without_runtime(profile: dict) -> str:
    value = json.loads(json.dumps(profile))
    value.pop("runtime_seconds", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_closed_realizations(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.continuous-closed-realization-manifest.v1":
        raise ValueError("Unsupported realization manifest")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Realization artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Realization integrity failure: {relative}")
    config = ClosedRealizationConfig(**read_json(output / "config.json"))
    config.validate()
    replayed = []
    for index in range(len(config.state_dims)):
        directory = output / "profiles" / f"profile_{index:03d}"
        stored = read_json(directory / "profile.json")
        system = ContinuousReLUSystem.from_dict(read_json(directory / "system.json"))
        if system.to_dict() != stored["system"]:
            raise ValueError("Stored system mismatch")
        verify_closed_realization(system, stored["certificate"])
        profile = _profile(config, index)
        if _without_runtime(stored) != _without_runtime(profile):
            raise ValueError(f"Realization replay mismatch: {index}")
        replayed.append(profile)
    summary = read_json(output / "summary.json")
    summary.pop("runtime_seconds", None)
    if summary != _summary(replayed):
        raise ValueError("Realization summary mismatch")
    return {"status": "verified", "profiles_replayed": len(replayed),
            "largest_state_dim": max(config.state_dims),
            "largest_realization": max(p["certificate"]["upper_bound"] for p in replayed),
            "initial_cells_verified": sum(p["verification"]["initial_cells_verified"] for p in replayed),
            "transition_segments_verified": sum(p["verification"]["transition_segments_verified"] for p in replayed),
            "packing_pairs_verified": sum(p["verification"]["packing_pairs_verified"] for p in replayed)}

