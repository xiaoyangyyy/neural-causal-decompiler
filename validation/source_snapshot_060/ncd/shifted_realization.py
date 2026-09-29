"""Nine-center inductive realization of the separable continuous ReLU benchmark."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
import json
import time
import numpy as np

from .continuous_cover import benchmark_cover_system, _parameter_count
from .transition_overlap_lower import _parameters
from .continuous_separation import (
    ContinuousReLUSystem, _abs_difference_bounds, _digest, _network_interval,
)
from .io import digest, read_json, save_json


@dataclass
class ShiftedRealizationConfig:
    seed: int = 14701
    state_dims: tuple[int, ...] = (1, 2)
    epsilon: float = 0.101
    relation_radius: float = 0.1005
    packing_bins: int = 5

    @classmethod
    def quick(cls, seed: int = 14701) -> "ShiftedRealizationConfig":
        return cls(seed=seed, state_dims=(1,))

    def validate(self) -> None:
        self.state_dims = tuple(int(x) for x in self.state_dims)
        self.epsilon = float(self.epsilon)
        self.relation_radius = float(self.relation_radius)
        self.packing_bins = int(self.packing_bins)
        if (not self.state_dims or any(d < 1 for d in self.state_dims)
                or not np.isfinite(self.epsilon) or self.epsilon <= 0
                or not np.isfinite(self.relation_radius)
                or self.relation_radius < 0.1 or self.relation_radius > self.epsilon
                or self.packing_bins < 2
                or 1 / (self.packing_bins - 1) <= 2 * self.epsilon):
            raise ValueError("Invalid shifted-realization configuration")


def _centers() -> list[float]:
    return [i / 10 for i in range(1, 10)]


def _axis_proof(radius: float) -> tuple[list[dict], list[dict], list[dict]]:
    system = benchmark_cover_system(1)
    centers = _centers()
    cell_boundaries = [0.0]
    cell_boundaries += [(left + right) / 2 for left, right in zip(centers[:-1], centers[1:])]
    cell_boundaries += [1.0]
    initial, observations, transitions = [], [], []
    for index, center in enumerate(centers):
        low, high = cell_boundaries[index:index + 2]
        cell_radius = max(abs(low - center), abs(high - center))
        initial.append({"state": index, "low": low, "high": high,
                        "state_distance_upper": cell_radius})
        relation_low = max(0.0, center - radius)
        relation_high = min(1.0, center + radius)
        output_low, output_high = _network_interval(
            system.observation, np.array([relation_low]), np.array([relation_high]))
        nominal_output = system.observation([center])
        observation_bound = _abs_difference_bounds(
            output_low, output_high, nominal_output, nominal_output)[1]
        observations.append({"state": index, "low": relation_low,
                             "high": relation_high,
                             "output_error_upper": observation_bound})
        cuts = sorted({
            ((left + right) / 2 - 0.5 * center) / 0.4
            for left, right in zip(centers[:-1], centers[1:])
            if 0 < ((left + right) / 2 - 0.5 * center) / 0.4 < 1
        })
        action_bounds = [0.0, *cuts, 1.0]
        segments = []
        for action_low, action_high in zip(action_bounds[:-1], action_bounds[1:]):
            midpoint = (action_low + action_high) / 2
            nominal = float(system.step([center], [midpoint])[0])
            target = min(range(len(centers)), key=lambda j: abs(nominal - centers[j]))
            next_low, next_high = _network_interval(
                system.transition, np.array([relation_low, action_low]),
                np.array([relation_high, action_high]))
            target_center = np.array([centers[target]])
            error = _abs_difference_bounds(
                next_low, next_high, target_center, target_center)[1]
            segments.append({"action_low": action_low, "action_high": action_high,
                             "target": target, "state_error_upper": error,
                             "next_low": float(next_low[0]),
                             "next_high": float(next_high[0])})
        transitions.append({"source": index, "segments": segments})
    return initial, observations, transitions


def _packing(system: ContinuousReLUSystem, bins: int) -> tuple[list[list[float]], list[dict]]:
    axis = np.linspace(0.0, 1.0, bins).tolist()
    points = [list(p) for p in product(axis, repeat=system.state_dim)]
    outputs = [_network_interval(system.observation, np.asarray(p), np.asarray(p))
               for p in points]
    pairs = []
    for left, right in combinations(range(len(points)), 2):
        distance = _abs_difference_bounds(*outputs[left], *outputs[right])[0]
        pairs.append({"left": left, "right": right,
                      "initial_output_distance_lower": distance})
    return points, pairs


def certified_shifted_realization(system: ContinuousReLUSystem,
                                  config: ShiftedRealizationConfig) -> dict:
    if system.to_dict() != benchmark_cover_system(system.state_dim).to_dict():
        raise ValueError("Shifted product proof requires the frozen separable benchmark")
    config.validate()
    _parameters(benchmark_cover_system(1))  # Exact unit-domain invariance.
    initial, observations, transitions = _axis_proof(config.relation_radius)
    points, pairs = _packing(system, config.packing_bins)
    certificate = {
        "schema": "ncd.shifted-realization.v1",
        "system_sha256": _digest(system.to_dict()),
        "domain_low": [0.0] * system.state_dim,
        "domain_high": [1.0] * system.state_dim,
        "action_lower": [0.0] * system.action_dim,
        "action_upper": [1.0] * system.action_dim,
        "epsilon": config.epsilon, "relation_radius": config.relation_radius,
        "packing_bins": config.packing_bins, "centers": _centers(),
        "axis_initial": initial, "axis_observations": observations,
        "axis_transitions": transitions,
        "packing_points": points, "packing_pairs": pairs,
        "lower_bound": len(points),
        "upper_bound": len(_centers()) ** system.state_dim,
        "minimal": len(points) == len(_centers()) ** system.state_dim,
        "claim": "inductive product realization for every continuous action sequence and every time",
        "boundary": "frozen separable ReLU dynamics; minimum state count not closed",
    }
    verify_shifted_realization(system, certificate)
    return certificate


def verify_shifted_realization(system: ContinuousReLUSystem,
                               certificate: dict) -> dict:
    if certificate.get("schema") != "ncd.shifted-realization.v1":
        raise ValueError("Unsupported shifted-realization certificate")
    if (system.to_dict() != benchmark_cover_system(system.state_dim).to_dict()
            or certificate.get("system_sha256") != _digest(system.to_dict())):
        raise ValueError("Shifted-realization system mismatch")
    d = system.state_dim
    config = ShiftedRealizationConfig(
        state_dims=(d,), epsilon=certificate["epsilon"],
        relation_radius=certificate["relation_radius"],
        packing_bins=certificate["packing_bins"])
    config.validate()
    if (certificate.get("domain_low") != [0.0] * d
            or certificate.get("domain_high") != [1.0] * d
            or certificate.get("action_lower") != [0.0] * d
            or certificate.get("action_upper") != [1.0] * d
            or certificate.get("centers") != _centers()
            or certificate.get("claim") !=
            "inductive product realization for every continuous action sequence and every time"
            or certificate.get("boundary") !=
            "frozen separable ReLU dynamics; minimum state count not closed"):
        raise ValueError("Shifted-realization domain or claim mismatch")
    _parameters(benchmark_cover_system(1))  # Exact unit-domain invariance.
    initial, observations, transitions = _axis_proof(config.relation_radius)
    if certificate.get("axis_initial") != initial:
        raise ValueError("Shifted initial cover failed replay")
    if certificate.get("axis_observations") != observations:
        raise ValueError("Shifted relation observation failed replay")
    if certificate.get("axis_transitions") != transitions:
        raise ValueError("Shifted action partition failed replay")
    if any(cell["state_distance_upper"] > config.relation_radius for cell in initial):
        raise ValueError("Shifted initial relation does not cover unit domain")
    if any(cell["output_error_upper"] > config.epsilon for cell in observations):
        raise ValueError("Shifted output relation exceeds tolerance")
    if any(segment["state_error_upper"] > config.relation_radius
           for entry in transitions for segment in entry["segments"]):
        raise ValueError("Shifted transition relation is not forward invariant")
    points, pairs = _packing(system, config.packing_bins)
    if (certificate.get("packing_points") != points
            or certificate.get("packing_pairs") != pairs
            or any(pair["initial_output_distance_lower"] <= 2 * config.epsilon
                   for pair in pairs)):
        raise ValueError("Shifted packing lower proof failed")
    lower, upper = len(points), 9 ** d
    if (certificate.get("lower_bound") != lower
            or certificate.get("upper_bound") != upper
            or certificate.get("minimal") != (lower == upper)):
        raise ValueError("Shifted aggregate state bounds mismatch")
    return {"status": "verified", "state_dim": d,
            "initial_cells_verified": len(initial),
            "observation_relations_verified": len(observations),
            "action_segments_verified": sum(len(e["segments"]) for e in transitions),
            "packing_pairs_verified": len(pairs),
            "lower_bound": lower, "upper_bound": upper,
            "horizon": "unbounded"}


def shifted_initial(certificate: dict, state: tuple[float, ...]) -> tuple[int, ...]:
    d = len(certificate["domain_low"])
    if len(state) != d or any(not np.isfinite(x) or x < 0 or x > 1 for x in state):
        raise ValueError("Initial state outside domain")
    centers = certificate["centers"]
    cuts = [(x + y) / 2 for x, y in zip(centers[:-1], centers[1:])]
    return tuple(bisect_right(cuts, x) for x in state)


def shifted_step(certificate: dict, state: tuple[int, ...],
                 action: tuple[float, ...]) -> tuple[int, ...]:
    d = len(certificate["domain_low"])
    if len(state) != d or len(action) != d:
        raise ValueError("Abstract state/action dimension mismatch")
    result = []
    for index, control in zip(state, action):
        if not isinstance(index, int) or not 0 <= index < 9:
            raise ValueError("Invalid shifted abstract state")
        if not np.isfinite(control) or not 0 <= control <= 1:
            raise ValueError("Action outside declared unit domain")
        segments = certificate["axis_transitions"][index]["segments"]
        cuts = [s["action_high"] for s in segments[:-1]]
        result.append(segments[bisect_right(cuts, control)]["target"])
    return tuple(result)


def shifted_output(certificate: dict, state: tuple[int, ...]) -> tuple[float, ...]:
    if len(state) != len(certificate["domain_low"]) or any(
            not isinstance(i, int) or not 0 <= i < 9 for i in state):
        raise ValueError("Invalid shifted abstract state")
    return tuple(certificate["centers"][i] for i in state)


def _profile(config: ShiftedRealizationConfig, index: int) -> dict:
    d = config.state_dims[index]
    system = benchmark_cover_system(d)
    started = time.perf_counter()
    certificate = certified_shifted_realization(system, config)
    runtime = time.perf_counter() - started
    return {"profile": index, "seed": config.seed + index * 1009,
            "state_dim": d, "action_dim": d,
            "parameter_count": _parameter_count(system),
            "system": system.to_dict(), "certificate": certificate,
            "verification": verify_shifted_realization(system, certificate),
            "runtime_seconds": runtime}


def _summary(profiles: list[dict]) -> dict:
    return {"schema": "ncd.shifted-realization-summary.v1",
            "profiles": len(profiles),
            "largest_state_dim": max(p["state_dim"] for p in profiles),
            "largest_realization": max(p["certificate"]["upper_bound"] for p in profiles),
            "total_initial_cells_verified": sum(p["verification"]["initial_cells_verified"] for p in profiles),
            "total_observation_relations_verified": sum(p["verification"]["observation_relations_verified"] for p in profiles),
            "total_action_segments_verified": sum(p["verification"]["action_segments_verified"] for p in profiles),
            "total_packing_pairs_verified": sum(p["verification"]["packing_pairs_verified"] for p in profiles),
            "scope": "unbounded-horizon shifted-grid realization for frozen separable ReLU dynamics"}


def run_shifted_realizations(output: Path, config: ShiftedRealizationConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    profiles = []
    started = time.perf_counter()
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
        "schema": "ncd.shifted-realization-manifest.v1", "artifacts": artifacts})
    return summary


def _without_runtime(profile: dict) -> str:
    value = json.loads(json.dumps(profile))
    value.pop("runtime_seconds", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_shifted_realizations(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.shifted-realization-manifest.v1":
        raise ValueError("Unsupported shifted-realization manifest")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Shifted-realization artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Shifted-realization integrity failure: {relative}")
    config = ShiftedRealizationConfig(**read_json(output / "config.json"))
    config.validate()
    replayed = []
    for index in range(len(config.state_dims)):
        directory = output / "profiles" / f"profile_{index:03d}"
        stored = read_json(directory / "profile.json")
        system = ContinuousReLUSystem.from_dict(read_json(directory / "system.json"))
        if system.to_dict() != stored["system"]:
            raise ValueError("Stored shifted system mismatch")
        verify_shifted_realization(system, stored["certificate"])
        profile = _profile(config, index)
        if _without_runtime(stored) != _without_runtime(profile):
            raise ValueError(f"Shifted-realization replay mismatch: {index}")
        replayed.append(profile)
    summary = read_json(output / "summary.json")
    summary.pop("runtime_seconds", None)
    if summary != _summary(replayed):
        raise ValueError("Shifted-realization summary mismatch")
    return {"status": "verified", "profiles_replayed": len(replayed),
            "largest_state_dim": max(config.state_dims),
            "largest_realization": max(p["certificate"]["upper_bound"] for p in replayed),
            "initial_cells_verified": sum(p["verification"]["initial_cells_verified"] for p in replayed),
            "observation_relations_verified": sum(p["verification"]["observation_relations_verified"] for p in replayed),
            "action_segments_verified": sum(p["verification"]["action_segments_verified"] for p in replayed),
            "packing_pairs_verified": sum(p["verification"]["packing_pairs_verified"] for p in replayed)}

