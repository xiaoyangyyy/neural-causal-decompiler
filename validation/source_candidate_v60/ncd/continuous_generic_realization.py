"""Generic grid simulation certificates for bounded continuous ReLU systems."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
import json
import time
import numpy as np

from .continuous_scale import _tuple_network
from .continuous_separation import (
    ContinuousReLUSystem, _abs_difference_bounds, _digest, _network_interval,
)
from .io import digest, read_json, save_json


@dataclass
class GridRealizationConfig:
    seed: int = 11701
    state_bins: int = 10
    action_bins: int = 10
    packing_bins: int = 5
    epsilon: float = 0.12
    relation_radius: float = 0.115

    def validate(self) -> None:
        self.state_bins = int(self.state_bins)
        self.action_bins = int(self.action_bins)
        self.packing_bins = int(self.packing_bins)
        self.epsilon = float(self.epsilon)
        self.relation_radius = float(self.relation_radius)
        if (self.state_bins < 2 or self.action_bins < 1 or self.packing_bins < 2
                or not np.isfinite(self.epsilon) or self.epsilon <= 0
                or not np.isfinite(self.relation_radius) or self.relation_radius <= 0
                or 1 / (2 * self.state_bins) > self.relation_radius):
            raise ValueError("Invalid grid-realization configuration")


def benchmark_coupled_system() -> ContinuousReLUSystem:
    """2D system with cross-coordinate coupling and an active ReLU phase boundary."""
    d = 2
    first = np.zeros((3 * d, 2 * d))
    first[:2 * d, :] = np.eye(2 * d)
    for i in range(d):
        first[2 * d + i, i] = 1.0
        first[2 * d + i, (i + 1) % d] = -1.0
    bias = np.r_[np.ones(2 * d), np.zeros(d)]
    second = np.zeros((d, 3 * d))
    for i in range(d):
        second[i, i] = 0.25
        second[i, (i + 1) % d] = 0.10
        second[i, d + i] = 0.20
        second[i, 2 * d + i] = 0.05
    transition = _tuple_network(
        [first, second], [bias, np.full(d, -0.45)])
    observation = _tuple_network(
        [np.eye(d), np.eye(d)], [np.ones(d), -np.ones(d)])
    return ContinuousReLUSystem(
        d, d, transition, observation, ("control_0", "control_1"))


def _geometry(system: ContinuousReLUSystem, config: GridRealizationConfig):
    d, action_dim = system.state_dim, system.action_dim
    state_axis = [(i + 0.5) / config.state_bins for i in range(config.state_bins)]
    states = [list(p) for p in product(state_axis, repeat=d)]
    state_indices = list(product(range(config.state_bins), repeat=d))
    action_indices = list(product(range(config.action_bins), repeat=action_dim))
    return states, state_indices, action_indices


def _recompute(system: ContinuousReLUSystem, config: GridRealizationConfig) -> dict:
    config.validate()
    states, state_indices, action_indices = _geometry(system, config)
    d, action_dim = system.state_dim, system.action_dim
    epsilon, radius, bins, action_bins = (config.epsilon, config.relation_radius,
                                        config.state_bins, config.action_bins)
    initial = []
    relations = []
    for index, (point, coordinate) in enumerate(zip(states, state_indices)):
        low = [i / bins for i in coordinate]
        high = [(i + 1) / bins for i in coordinate]
        cell_radius = max(abs(x - r) for x, r in zip(low + high, point + point))
        initial.append({"state": index, "low": low, "high": high,
                        "state_distance_upper": cell_radius})
        relation_low = np.maximum(0.0, np.asarray(point) - radius)
        relation_high = np.minimum(1.0, np.asarray(point) + radius)
        output_low, output_high = _network_interval(
            system.observation, relation_low, relation_high)
        nominal = system.observation(point)
        output_error = _abs_difference_bounds(
            output_low, output_high, nominal, nominal)[1]
        relations.append({"state": index, "low": relation_low.tolist(),
                          "high": relation_high.tolist(),
                          "output_error_upper": output_error})
    transitions = []
    worst_bound = 0.0
    failures = []
    for source, point in enumerate(states):
        state_low = np.maximum(0.0, np.asarray(point) - radius)
        state_high = np.minimum(1.0, np.asarray(point) + radius)
        for action_index in action_indices:
            action_low = np.asarray([i / action_bins for i in action_index])
            action_high = np.asarray([(i + 1) / action_bins for i in action_index])
            midpoint = (action_low + action_high) / 2
            nominal = system.step(point, midpoint)
            target_axis = tuple(min(max(int(value * bins), 0), bins - 1)
                                for value in nominal)
            target = sum(value * bins ** (d - 1 - axis)
                         for axis, value in enumerate(target_axis))
            next_low, next_high = _network_interval(
                system.transition, np.r_[state_low, action_low],
                np.r_[state_high, action_high])
            output_center = np.asarray(states[target])
            error = _abs_difference_bounds(
                next_low, next_high, output_center, output_center)[1]
            inside = bool(np.all(next_low >= 0.0) and np.all(next_high <= 1.0))
            worst_bound = max(worst_bound, error)
            entry = {"source": source, "action_index": list(action_index),
                     "target": target, "state_error_upper": error,
                     "next_low": next_low.tolist(), "next_high": next_high.tolist()}
            transitions.append(entry)
            if (error > radius or not inside) and len(failures) < 20:
                failures.append({"source": source, "action_index": list(action_index),
                                 "error": error, "inside_domain": inside})
    packing_axis = np.linspace(0.0, 1.0, config.packing_bins).tolist()
    packing_points = [list(p) for p in product(packing_axis, repeat=d)]
    outputs = [_network_interval(system.observation, np.asarray(p), np.asarray(p))
               for p in packing_points]
    pairs = []
    for left, right in combinations(range(len(packing_points)), 2):
        distance = _abs_difference_bounds(*outputs[left], *outputs[right])[0]
        if distance > 2 * epsilon:
            pairs.append({"left": left, "right": right,
                          "initial_output_distance_lower": distance})
    complete_packing = len(pairs) == len(packing_points) * (len(packing_points) - 1) // 2
    initial_ok = all(cell["state_distance_upper"] <= radius for cell in initial)
    observation_ok = all(item["output_error_upper"] <= epsilon for item in relations)
    transition_ok = not failures
    certified = initial_ok and observation_ok and transition_ok
    return {
        "schema": "ncd.generic-grid-realization.v1",
        "system_sha256": _digest(system.to_dict()),
        "state_dim": d, "action_dim": action_dim,
        "domain_low": [0.0] * d, "domain_high": [1.0] * d,
        "action_lower": [0.0] * action_dim,
        "action_upper": [1.0] * action_dim,
        "epsilon": epsilon, "relation_radius": radius, "state_bins": bins,
        "action_bins": action_bins, "packing_bins": config.packing_bins,
        "representatives": states,
        "initial_cells": initial, "relation_output": relations,
        "transitions": transitions,
        "packing_points": packing_points, "packing_pairs": pairs,
        "packing_complete": complete_packing,
        "lower_bound": len(packing_points) if complete_packing else None,
        "upper_bound": len(states) if certified else None,
        "status": "certified" if certified else "unresolved",
        "worst_transition_bound": worst_bound,
        "failures": failures,
        "claim": "infinite-horizon epsilon simulation if and only if status is certified",
    }


def certified_grid_realization(system: ContinuousReLUSystem,
                               config: GridRealizationConfig) -> dict:
    """Return a checked upper certificate or an explicit unresolved result."""
    certificate = _recompute(system, config)
    verify_grid_realization(system, certificate)
    return certificate


def verify_grid_realization(system: ContinuousReLUSystem, certificate: dict) -> dict:
    """Check the stored machine independently of its construction heuristic."""
    if certificate.get("schema") != "ncd.generic-grid-realization.v1":
        raise ValueError("Unsupported grid-realization certificate")
    config = GridRealizationConfig(
        state_bins=certificate["state_bins"], action_bins=certificate["action_bins"],
        packing_bins=certificate["packing_bins"], epsilon=certificate["epsilon"],
        relation_radius=certificate["relation_radius"])
    config.validate()
    d, action_dim = system.state_dim, system.action_dim
    epsilon, radius = config.epsilon, config.relation_radius
    bins, action_bins = config.state_bins, config.action_bins
    if (certificate.get("system_sha256") != _digest(system.to_dict())
            or certificate.get("state_dim") != d
            or certificate.get("action_dim") != action_dim
            or certificate.get("domain_low") != [0.0] * d
            or certificate.get("domain_high") != [1.0] * d
            or certificate.get("action_lower") != [0.0] * action_dim
            or certificate.get("action_upper") != [1.0] * action_dim
            or certificate.get("claim") !=
            "infinite-horizon epsilon simulation if and only if status is certified"):
        raise ValueError("Grid-realization system, domain, or claim mismatch")
    centers = [(i + 0.5) / bins for i in range(bins)]
    states = [list(p) for p in product(centers, repeat=d)]
    if certificate.get("representatives") != states:
        raise ValueError("Invalid representative grid")
    if (len(certificate.get("initial_cells", [])) != len(states)
            or len(certificate.get("relation_output", [])) != len(states)):
        raise ValueError("Incomplete initial or observation proof")
    initial_ok, observation_ok = True, True
    for index, point in enumerate(states):
        coordinate = tuple(int(value * bins) for value in point)
        low = [i / bins for i in coordinate]
        high = [(i + 1) / bins for i in coordinate]
        cell_radius = max(abs(x - r) for x, r in zip(low + high, point + point))
        expected_cell = {"state": index, "low": low, "high": high,
                         "state_distance_upper": cell_radius}
        if certificate["initial_cells"][index] != expected_cell:
            raise ValueError("Initial-cell geometry or radius mismatch")
        initial_ok &= cell_radius <= radius
        relation_low = np.maximum(0.0, np.asarray(point) - radius)
        relation_high = np.minimum(1.0, np.asarray(point) + radius)
        output_low, output_high = _network_interval(
            system.observation, relation_low, relation_high)
        nominal = system.observation(point)
        output_error = _abs_difference_bounds(
            output_low, output_high, nominal, nominal)[1]
        expected_relation = {"state": index, "low": relation_low.tolist(),
                             "high": relation_high.tolist(),
                             "output_error_upper": output_error}
        if certificate["relation_output"][index] != expected_relation:
            raise ValueError("Observation relation bound mismatch")
        observation_ok &= output_error <= epsilon
    action_indices = list(product(range(action_bins), repeat=action_dim))
    expected_count = len(states) * len(action_indices)
    transitions = certificate.get("transitions", [])
    if len(transitions) != expected_count:
        raise ValueError("Incomplete state-action transition table")
    failures = []
    worst = 0.0
    for source, point in enumerate(states):
        state_low = np.maximum(0.0, np.asarray(point) - radius)
        state_high = np.minimum(1.0, np.asarray(point) + radius)
        for action_offset, action_index in enumerate(action_indices):
            entry = transitions[source * len(action_indices) + action_offset]
            if (entry.get("source") != source
                    or entry.get("action_index") != list(action_index)
                    or not isinstance(entry.get("target"), int)
                    or not 0 <= entry["target"] < len(states)):
                raise ValueError("Transition table indexing or target mismatch")
            control_low = np.asarray([i / action_bins for i in action_index])
            control_high = np.asarray([(i + 1) / action_bins for i in action_index])
            next_low, next_high = _network_interval(
                system.transition, np.r_[state_low, control_low],
                np.r_[state_high, control_high])
            target = np.asarray(states[entry["target"]])
            error = _abs_difference_bounds(next_low, next_high, target, target)[1]
            inside = bool(np.all(next_low >= 0.0) and np.all(next_high <= 1.0))
            expected_entry = {
                "source": source, "action_index": list(action_index),
                "target": entry["target"], "state_error_upper": error,
                "next_low": next_low.tolist(), "next_high": next_high.tolist()}
            if entry != expected_entry:
                raise ValueError("Stored transition interval or bound mismatch")
            worst = max(worst, error)
            if (error > radius or not inside) and len(failures) < 20:
                failures.append({"source": source, "action_index": list(action_index),
                                 "error": error, "inside_domain": inside})
    packing_axis = np.linspace(0.0, 1.0, config.packing_bins).tolist()
    points = [list(p) for p in product(packing_axis, repeat=d)]
    if certificate.get("packing_points") != points:
        raise ValueError("Packing point set mismatch")
    outputs = [_network_interval(system.observation, np.asarray(p), np.asarray(p))
               for p in points]
    pairs = []
    for left, right in combinations(range(len(points)), 2):
        distance = _abs_difference_bounds(*outputs[left], *outputs[right])[0]
        if distance > 2 * epsilon:
            pairs.append({"left": left, "right": right,
                          "initial_output_distance_lower": distance})
    complete = len(pairs) == len(points) * (len(points) - 1) // 2
    certified = initial_ok and observation_ok and not failures
    if (certificate.get("packing_pairs") != pairs
            or certificate.get("packing_complete") != complete
            or certificate.get("lower_bound") != (len(points) if complete else None)
            or certificate.get("upper_bound") != (len(states) if certified else None)
            or certificate.get("status") != ("certified" if certified else "unresolved")
            or certificate.get("worst_transition_bound") != worst
            or certificate.get("failures") != failures):
        raise ValueError("Grid-realization aggregate proof mismatch")
    return {"status": "certified" if certified else "unresolved",
            "state_dim": d, "states": len(states),
            "initial_cells_verified": len(states),
            "relation_observations_verified": len(states),
            "transition_boxes_verified": len(transitions),
            "packing_pairs_verified": len(pairs),
            "lower_bound": len(points) if complete else None,
            "upper_bound": len(states) if certified else None,
            "worst_transition_bound": worst,
            "horizon": "unbounded" if certified else None}

def grid_initial(certificate: dict, state: tuple[float, ...]) -> int:
    if certificate.get("status") != "certified":
        raise ValueError("No certified executable realization")
    d, bins = certificate["state_dim"], certificate["state_bins"]
    if len(state) != d or any(not np.isfinite(x) or x < 0 or x > 1 for x in state):
        raise ValueError("Initial state outside domain")
    index = tuple(min(int(x * bins), bins - 1) for x in state)
    return sum(value * bins ** (d - 1 - axis) for axis, value in enumerate(index))


def grid_step(certificate: dict, state: int, action: tuple[float, ...]) -> int:
    if certificate.get("status") != "certified":
        raise ValueError("No certified executable realization")
    d, a = certificate["state_dim"], certificate["action_dim"]
    if not isinstance(state, int) or not 0 <= state < len(certificate["representatives"]):
        raise ValueError("Invalid abstract state")
    if len(action) != a or any(not np.isfinite(x) or x < 0 or x > 1 for x in action):
        raise ValueError("Action outside domain")
    bins = certificate["action_bins"]
    coordinate = tuple(min(int(x * bins), bins - 1) for x in action)
    offset = sum(value * bins ** (a - 1 - axis)
                 for axis, value in enumerate(coordinate))
    return certificate["transitions"][state * bins ** a + offset]["target"]


def grid_output(system: ContinuousReLUSystem, certificate: dict, state: int) -> list[float]:
    if certificate.get("status") != "certified":
        raise ValueError("No certified executable realization")
    if not isinstance(state, int) or not 0 <= state < len(certificate["representatives"]):
        raise ValueError("Invalid abstract state")
    return system.observation(certificate["representatives"][state]).tolist()


def run_generic_grid(output: Path, system: ContinuousReLUSystem,
                     config: GridRealizationConfig) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    save_json(output / "system.json", system.to_dict())
    started = time.perf_counter()
    certificate = certified_grid_realization(system, config)
    runtime = time.perf_counter() - started
    save_json(output / "certificate.json", certificate)
    summary = verify_grid_realization(system, certificate)
    summary["runtime_seconds"] = runtime
    save_json(output / "summary.json", summary)
    artifacts = {p.relative_to(output).as_posix(): digest(p)
                 for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    save_json(output / "manifest.json", {
        "schema": "ncd.generic-grid-realization-manifest.v1",
        "artifacts": artifacts})
    return summary


def verify_generic_grid(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.generic-grid-realization-manifest.v1":
        raise ValueError("Unsupported generic-grid manifest")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Generic-grid artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Generic-grid integrity failure: {relative}")
    system = ContinuousReLUSystem.from_dict(read_json(output / "system.json"))
    config = GridRealizationConfig(**read_json(output / "config.json"))
    stored = read_json(output / "certificate.json")
    verification = verify_grid_realization(system, stored)
    regenerated = certified_grid_realization(system, config)
    if json.dumps(regenerated, sort_keys=True) != json.dumps(stored, sort_keys=True):
        raise ValueError("Generic-grid deterministic replay mismatch")
    summary = read_json(output / "summary.json")
    summary.pop("runtime_seconds", None)
    if summary != verification:
        raise ValueError("Generic-grid summary mismatch")
    return verification

