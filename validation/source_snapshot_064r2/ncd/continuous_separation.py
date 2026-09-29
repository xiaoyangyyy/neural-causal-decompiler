"""Certified finite-horizon separation bounds for continuous ReLU systems.

The oracle treats the initial states as fixed and the intervention sequence as
the continuous optimization domain. A feasible intervention gives a lower
bound; directed-rounding interval propagation over a branch-and-bound cover
gives the upper bound. Failure to close the interval remains unresolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Iterable
from itertools import combinations

import numpy as np

from .io import digest as file_digest, read_json, save_json
from .approximate_finite import chromatic_number


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReLUMLP:
    """A float64 MLP with ReLU after every layer except the last."""

    weights: tuple[tuple[tuple[float, ...], ...], ...]
    biases: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("An MLP needs matching non-empty weights and biases")
        previous = None
        for weight, bias in zip(self.weights, self.biases):
            array = np.asarray(weight, dtype=float)
            if array.ndim != 2 or array.shape[0] != len(bias) or not array.size:
                raise ValueError("Invalid affine layer dimensions")
            if previous is not None and array.shape[1] != previous:
                raise ValueError("Adjacent MLP dimensions do not match")
            previous = array.shape[0]
            if not np.isfinite(array).all() or not np.isfinite(np.asarray(bias)).all():
                raise ValueError("MLP parameters must be finite")

    @property
    def input_dim(self) -> int:
        return len(self.weights[0][0])

    @property
    def output_dim(self) -> int:
        return len(self.biases[-1])

    def __call__(self, value: Iterable[float]) -> np.ndarray:
        current = np.asarray(tuple(value), dtype=np.float64)
        if current.shape != (self.input_dim,):
            raise ValueError("Invalid MLP input dimension")
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = np.asarray(weight, dtype=np.float64) @ current + np.asarray(bias, dtype=np.float64)
            if index + 1 < len(self.weights):
                current = np.maximum(current, 0.0)
        return current

    def to_dict(self) -> dict:
        return {"weights": [[list(row) for row in layer] for layer in self.weights],
                "biases": [list(layer) for layer in self.biases]}

    @classmethod
    def from_dict(cls, value: dict) -> "ReLUMLP":
        return cls(
            tuple(tuple(tuple(float(x) for x in row) for row in layer) for layer in value["weights"]),
            tuple(tuple(float(x) for x in layer) for layer in value["biases"]),
        )


@dataclass(frozen=True)
class ContinuousReLUSystem:
    """Deterministic recurrent neural system with continuous controls."""

    state_dim: int
    action_dim: int
    transition: ReLUMLP
    observation: ReLUMLP
    action_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("State and action dimensions must be positive")
        if self.transition.input_dim != self.state_dim + self.action_dim:
            raise ValueError("Transition input dimension mismatch")
        if self.transition.output_dim != self.state_dim or self.observation.input_dim != self.state_dim:
            raise ValueError("System network dimensions do not match")
        if self.action_names and len(self.action_names) != self.action_dim:
            raise ValueError("Action name count mismatch")

    def step(self, state: Iterable[float], action: Iterable[float]) -> np.ndarray:
        state_array = np.asarray(tuple(state), dtype=np.float64)
        action_array = np.asarray(tuple(action), dtype=np.float64)
        return self.transition(np.concatenate((state_array, action_array)))

    def response(self, state: Iterable[float], actions: np.ndarray) -> np.ndarray:
        current = np.asarray(tuple(state), dtype=np.float64)
        values = [self.observation(current)]
        for action in np.asarray(actions, dtype=np.float64):
            current = self.step(current, action)
            values.append(self.observation(current))
        return np.asarray(values)

    def to_dict(self) -> dict:
        return {"schema": "ncd.continuous-relu-system.v1", "state_dim": self.state_dim,
                "action_dim": self.action_dim, "action_names": list(self.action_names),
                "transition": self.transition.to_dict(), "observation": self.observation.to_dict()}

    @classmethod
    def from_dict(cls, value: dict) -> "ContinuousReLUSystem":
        if value.get("schema") != "ncd.continuous-relu-system.v1":
            raise ValueError("Unsupported continuous neural system")
        return cls(int(value["state_dim"]), int(value["action_dim"]),
                   ReLUMLP.from_dict(value["transition"]), ReLUMLP.from_dict(value["observation"]),
                   tuple(value.get("action_names", ())))


def _lower(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _upper(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def _affine_interval(weight: tuple[tuple[float, ...], ...], bias: tuple[float, ...],
                     low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Affine interval arithmetic with outward rounding after every operation."""
    result_low = np.empty(len(bias), dtype=np.float64)
    result_high = np.empty(len(bias), dtype=np.float64)
    for row, (coefficients, offset) in enumerate(zip(weight, bias)):
        lo = _lower(offset)
        hi = _upper(offset)
        for coefficient, left, right in zip(coefficients, low, high):
            if coefficient >= 0:
                term_low = _lower(coefficient * float(left))
                term_high = _upper(coefficient * float(right))
            else:
                term_low = _lower(coefficient * float(right))
                term_high = _upper(coefficient * float(left))
            lo = _lower(lo + term_low)
            hi = _upper(hi + term_high)
        result_low[row], result_high[row] = lo, hi
    return result_low, result_high


def _network_interval(network: ReLUMLP, low: np.ndarray,
                      high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        low, high = _affine_interval(weight, bias, low, high)
        if index + 1 < len(network.weights):
            low, high = np.maximum(low, 0.0), np.maximum(high, 0.0)
    return low, high


def _abs_difference_bounds(left_low: np.ndarray, left_high: np.ndarray,
                           right_low: np.ndarray, right_high: np.ndarray) -> tuple[float, float]:
    difference_low = np.nextafter(left_low - right_high, -np.inf)
    difference_high = np.nextafter(left_high - right_low, np.inf)
    upper = float(np.max(np.maximum(np.abs(difference_low), np.abs(difference_high))))
    lower_per_coordinate = np.where(difference_low > 0, difference_low,
                                    np.where(difference_high < 0, -difference_high, 0.0))
    return float(np.max(lower_per_coordinate)), _upper(upper)


def response_distance_bounds(system: ContinuousReLUSystem, left: Iterable[float],
                             right: Iterable[float], action_low: np.ndarray,
                             action_high: np.ndarray) -> tuple[float, float]:
    """Sound bounds for max-time L-infinity response distance on one action box."""
    left_low = left_high = np.asarray(tuple(left), dtype=np.float64)
    right_low = right_high = np.asarray(tuple(right), dtype=np.float64)
    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)
    if action_low.shape != action_high.shape or action_low.ndim != 2 or action_low.shape[1] != system.action_dim:
        raise ValueError("Action box shape mismatch")
    if np.any(action_low > action_high):
        raise ValueError("Empty action box")
    # The two runs share the intervention word. Reflexivity is therefore exact,
    # while two independent interval traces would lose that dependency.
    if np.array_equal(left_low, right_low):
        return 0.0, 0.0
    overall_lower, overall_upper = 0.0, 0.0
    for time in range(len(action_low) + 1):
        yll, ylh = _network_interval(system.observation, left_low, left_high)
        yrl, yrh = _network_interval(system.observation, right_low, right_high)
        lower, upper = _abs_difference_bounds(yll, ylh, yrl, yrh)
        overall_lower, overall_upper = max(overall_lower, lower), max(overall_upper, upper)
        if time < len(action_low):
            left_low, left_high = _network_interval(
                system.transition, np.concatenate((left_low, action_low[time])),
                np.concatenate((left_high, action_high[time])))
            right_low, right_high = _network_interval(
                system.transition, np.concatenate((right_low, action_low[time])),
                np.concatenate((right_high, action_high[time])))
    return overall_lower, overall_upper


def _shared_network_difference_interval(
    network: ReLUMLP,
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
    difference_low: np.ndarray,
    difference_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a relational difference while every paired ReLU is stable.

    The two executions retain the same symbolic intervention. Affine biases and
    shared-action terms therefore cancel from the difference. A hidden neuron
    is accepted only when both executions are provably active or both are
    provably inactive over the complete input box.
    """
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        next_left_low, next_left_high = _affine_interval(weight, bias, left_low, left_high)
        next_right_low, next_right_high = _affine_interval(weight, bias, right_low, right_high)
        zeros = tuple(0.0 for _ in bias)
        next_difference_low, next_difference_high = _affine_interval(
            weight, zeros, difference_low, difference_high)
        if index + 1 < len(network.weights):
            active = (next_left_low >= 0.0) & (next_right_low >= 0.0)
            inactive = (next_left_high <= 0.0) & (next_right_high <= 0.0)
            if not np.all(active | inactive):
                raise ValueError("Relational ReLU stability could not be certified")
            next_left_low, next_left_high = np.maximum(next_left_low, 0.0), np.maximum(next_left_high, 0.0)
            next_right_low, next_right_high = np.maximum(next_right_low, 0.0), np.maximum(next_right_high, 0.0)
            next_difference_low = np.where(inactive, 0.0, next_difference_low)
            next_difference_high = np.where(inactive, 0.0, next_difference_high)
        left_low, left_high = next_left_low, next_left_high
        right_low, right_high = next_right_low, next_right_high
        difference_low, difference_high = next_difference_low, next_difference_high
    return left_low, left_high, right_low, right_high, difference_low, difference_high


def relational_response_distance_bounds(
    system: ContinuousReLUSystem,
    left: Iterable[float],
    right: Iterable[float],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[float, float]:
    """Sound paired bound that preserves shared controls through stable ReLUs."""
    left_low = left_high = np.asarray(tuple(left), dtype=np.float64)
    right_low = right_high = np.asarray(tuple(right), dtype=np.float64)
    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)
    if (left_low.shape != (system.state_dim,) or right_low.shape != (system.state_dim,)
            or action_low.shape != action_high.shape or action_low.ndim != 2
            or action_low.shape[1] != system.action_dim or np.any(action_low > action_high)):
        raise ValueError("Invalid relational response domain")
    difference_low = np.nextafter(left_low - right_high, -np.inf)
    difference_high = np.nextafter(left_high - right_low, np.inf)
    upper_bound = 0.0
    for time_index in range(len(action_low) + 1):
        _, _, _, _, output_difference_low, output_difference_high = _shared_network_difference_interval(
            system.observation, left_low, left_high, right_low, right_high,
            difference_low, difference_high)
        upper_bound = max(
            upper_bound,
            _upper(float(np.max(np.maximum(np.abs(output_difference_low),
                                           np.abs(output_difference_high))))),
        )
        if time_index < len(action_low):
            zero_action = np.zeros(system.action_dim, dtype=float)
            (left_low, left_high, right_low, right_high,
             difference_low, difference_high) = _shared_network_difference_interval(
                system.transition,
                np.concatenate((left_low, action_low[time_index])),
                np.concatenate((left_high, action_high[time_index])),
                np.concatenate((right_low, action_low[time_index])),
                np.concatenate((right_high, action_high[time_index])),
                np.concatenate((difference_low, zero_action)),
                np.concatenate((difference_high, zero_action)),
            )
    return 0.0, upper_bound


def _leaf(system: ContinuousReLUSystem, left: tuple[float, ...], right: tuple[float, ...],
          low: np.ndarray, high: np.ndarray, bound_method: str = "independent-ibp") -> dict:
    center = (low + high) / 2.0
    witness_lower, witness_upper = response_distance_bounds(system, left, right, center, center)
    actual_method = bound_method
    if bound_method == "relational-stable":
        try:
            _, bound = relational_response_distance_bounds(system, left, right, low, high)
        except ValueError:
            actual_method = "independent-ibp"
            _, bound = response_distance_bounds(system, left, right, low, high)
    elif bound_method == "independent-ibp":
        _, bound = response_distance_bounds(system, left, right, low, high)
    else:
        raise ValueError("Unknown continuous bound method")
    return {"kind": "leaf", "low": low.tolist(), "high": high.tolist(),
            "upper_bound": bound, "witness_actions": center.tolist(),
            "witness_lower_bound": witness_lower, "witness_upper_bound": witness_upper,
            "bound_method": actual_method}


def certified_separation(system: ContinuousReLUSystem, left: Iterable[float], right: Iterable[float],
                         action_lower: Iterable[float], action_upper: Iterable[float], horizon: int,
                         epsilon: float, max_leaves: int = 256, target_gap: float = 1e-5,
                         bound_method: str = "independent-ibp",
                         split_strategy: str = "widest") -> dict:
    """Bound separation and return a replayable branch-and-bound certificate."""
    left = tuple(float(x) for x in left)
    right = tuple(float(x) for x in right)
    lower_action = np.asarray(tuple(action_lower), dtype=np.float64)
    upper_action = np.asarray(tuple(action_upper), dtype=np.float64)
    if len(left) != system.state_dim or len(right) != system.state_dim:
        raise ValueError("Initial state dimension mismatch")
    if lower_action.shape != (system.action_dim,) or upper_action.shape != (system.action_dim,):
        raise ValueError("Action bound dimension mismatch")
    if (horizon < 0 or epsilon < 0 or max_leaves < 1 or target_gap < 0
            or np.any(lower_action > upper_action) or split_strategy not in ("widest", "best-bound")):
        raise ValueError("Invalid separation configuration")
    root = _leaf(system, left, right, np.tile(lower_action, (horizon, 1)),
                 np.tile(upper_action, (horizon, 1)), bound_method)
    leaves = [root]
    while len(leaves) < max_leaves:
        lower_bound = max(leaf["witness_lower_bound"] for leaf in leaves)
        upper_bound = max(leaf["upper_bound"] for leaf in leaves)
        if lower_bound > 2 * epsilon or upper_bound <= 2 * epsilon or upper_bound - lower_bound <= target_gap:
            break
        candidate = max(leaves, key=lambda leaf: leaf["upper_bound"])
        low, high = np.asarray(candidate["low"]), np.asarray(candidate["high"])
        widths = (high - low).reshape(-1)
        if not np.any(widths > 0):
            break
        if split_strategy == "widest":
            dimensions = [int(np.argmax(widths))]
        else:
            dimensions = [index for index, width in enumerate(widths) if width > 0]
        best = None
        for dimension in dimensions:
            split = float((low.reshape(-1)[dimension] + high.reshape(-1)[dimension]) / 2.0)
            first_high, second_low = high.copy(), low.copy()
            first_high.reshape(-1)[dimension] = split
            second_low.reshape(-1)[dimension] = split
            first = _leaf(system, left, right, low, first_high, bound_method)
            second = _leaf(system, left, right, second_low, high, bound_method)
            proposal = (max(first["upper_bound"], second["upper_bound"]), dimension,
                        split, first, second)
            if best is None or proposal[:2] < best[:2]:
                best = proposal
        _, dimension, split, first, second = best
        candidate.clear()
        candidate.update({"kind": "split", "dimension": dimension, "value": split,
                          "low": low.tolist(), "high": high.tolist(), "children": [first, second]})
        leaves.remove(candidate)
        leaves.extend((first, second))
    lower_bound = max(leaf["witness_lower_bound"] for leaf in leaves)
    upper_bound = max(leaf["upper_bound"] for leaf in leaves)
    witness = max(leaves, key=lambda leaf: leaf["witness_lower_bound"])["witness_actions"]
    status = "separated" if lower_bound > 2 * epsilon else "certified-within-epsilon" if upper_bound <= 2 * epsilon else "unresolved"
    return {"schema": "ncd.continuous-separation-certificate.v1", "system_sha256": _digest(system.to_dict()),
            "left": list(left), "right": list(right), "action_lower": lower_action.tolist(),
            "action_upper": upper_action.tolist(), "horizon": horizon, "epsilon": epsilon,
            "max_leaves": max_leaves, "target_gap": target_gap, "lower_bound": lower_bound,
            "upper_bound": upper_bound, "status": status, "witness_actions": witness,
            "leaves": len(leaves), "bound_method": bound_method,
            "split_strategy": split_strategy, "tree": root}


def verify_separation_certificate(system: ContinuousReLUSystem, certificate: dict) -> dict:
    """Independently check domain coverage and every numerical bound."""
    if certificate.get("schema") != "ncd.continuous-separation-certificate.v1":
        raise ValueError("Unsupported separation certificate")
    if certificate.get("system_sha256") != _digest(system.to_dict()):
        raise ValueError("Certificate is for a different neural system")
    left, right = tuple(certificate["left"]), tuple(certificate["right"])
    horizon = int(certificate["horizon"])
    root_low = np.tile(np.asarray(certificate["action_lower"], dtype=float), (horizon, 1))
    root_high = np.tile(np.asarray(certificate["action_upper"], dtype=float), (horizon, 1))
    verified_leaves: list[tuple[float, float, list]] = []

    def visit(node: dict, expected_low: np.ndarray, expected_high: np.ndarray) -> None:
        low, high = np.asarray(node["low"], dtype=float), np.asarray(node["high"], dtype=float)
        if not np.array_equal(low, expected_low) or not np.array_equal(high, expected_high):
            raise ValueError("Certificate tree does not cover its declared box")
        if node["kind"] == "split":
            dimension, value = int(node["dimension"]), float(node["value"])
            if not 0 <= dimension < low.size or not low.reshape(-1)[dimension] < value < high.reshape(-1)[dimension]:
                raise ValueError("Invalid branch split")
            first_high, second_low = high.copy(), low.copy()
            first_high.reshape(-1)[dimension] = value
            second_low.reshape(-1)[dimension] = value
            if len(node.get("children", ())) != 2:
                raise ValueError("A split must have two children")
            visit(node["children"][0], low, first_high)
            visit(node["children"][1], second_low, high)
            return
        if node["kind"] != "leaf":
            raise ValueError("Unknown certificate tree node")
        actions = np.asarray(node["witness_actions"], dtype=float)
        if actions.shape != low.shape or np.any(actions < low) or np.any(actions > high):
            raise ValueError("Witness lies outside its certified box")
        witness_lower, witness_upper = response_distance_bounds(system, left, right, actions, actions)
        method = node.get("bound_method", "independent-ibp")
        if method == "relational-stable":
            _, upper_bound = relational_response_distance_bounds(system, left, right, low, high)
        elif method == "independent-ibp":
            _, upper_bound = response_distance_bounds(system, left, right, low, high)
        else:
            raise ValueError("Unknown leaf bound method")
        for key, actual in (("witness_lower_bound", witness_lower), ("witness_upper_bound", witness_upper),
                            ("upper_bound", upper_bound)):
            if float(node[key]) != actual:
                raise ValueError(f"Invalid leaf {key}")
        verified_leaves.append((witness_lower, upper_bound, node["witness_actions"]))

    visit(certificate["tree"], root_low, root_high)
    lower_bound = max(value[0] for value in verified_leaves)
    upper_bound = max(value[1] for value in verified_leaves)
    epsilon = float(certificate["epsilon"])
    status = "separated" if lower_bound > 2 * epsilon else "certified-within-epsilon" if upper_bound <= 2 * epsilon else "unresolved"
    declared_witness = np.asarray(certificate.get("witness_actions"), dtype=float)
    if (declared_witness.shape != root_low.shape or np.any(declared_witness < root_low)
            or np.any(declared_witness > root_high)):
        raise ValueError("Aggregate witness lies outside the intervention domain")
    declared_lower, _ = response_distance_bounds(system, left, right, declared_witness, declared_witness)
    if declared_lower != lower_bound:
        raise ValueError("Aggregate witness does not attain the declared lower bound")
    expected = {"lower_bound": lower_bound, "upper_bound": upper_bound, "status": status,
                "leaves": len(verified_leaves)}
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(f"Invalid aggregate certificate field: {key}")
    return {"status": status, "lower_bound": lower_bound, "upper_bound": upper_bound,
            "leaves_verified": len(verified_leaves)}


def certified_incompatibility_lower_bound(
    system: ContinuousReLUSystem, states: Iterable[Iterable[float]],
    action_lower: Iterable[float], action_upper: Iterable[float], horizon: int,
    epsilon: float, max_leaves: int = 256,
) -> dict:
    """Certify a chromatic lower bound from continuous separation queries."""
    states = tuple(tuple(float(x) for x in state) for state in states)
    if not states:
        raise ValueError("At least one state is required")
    pair_certificates = []
    adjacency = [set() for _ in states]
    for left, right in combinations(range(len(states)), 2):
        certificate = certified_separation(system, states[left], states[right], action_lower,
                                           action_upper, horizon, epsilon, max_leaves)
        pair_certificates.append({"left": left, "right": right, "certificate": certificate})
        if certificate["status"] == "separated":
            adjacency[left].add(right)
            adjacency[right].add(left)
    bound, coloring = chromatic_number(adjacency)
    return {"schema": "ncd.continuous-incompatibility-lower.v1",
            "system_sha256": _digest(system.to_dict()), "states": [list(state) for state in states],
            "action_lower": list(action_lower), "action_upper": list(action_upper),
            "horizon": horizon, "epsilon": epsilon, "max_leaves": max_leaves,
            "pair_certificates": pair_certificates, "bound": bound,
            "chromatic_coloring": coloring,
            "claim": "every epsilon-faithful encoding of the declared states needs at least bound states"}


def verify_incompatibility_lower_bound(system: ContinuousReLUSystem, certificate: dict) -> dict:
    if certificate.get("schema") != "ncd.continuous-incompatibility-lower.v1":
        raise ValueError("Unsupported continuous incompatibility certificate")
    if certificate.get("system_sha256") != _digest(system.to_dict()):
        raise ValueError("Incompatibility certificate system mismatch")
    states = certificate["states"]
    expected_pairs = set(combinations(range(len(states)), 2))
    actual_pairs = set()
    adjacency = [set() for _ in states]
    for item in certificate["pair_certificates"]:
        pair = (int(item["left"]), int(item["right"]))
        if pair not in expected_pairs or pair in actual_pairs:
            raise ValueError("Invalid or duplicate continuous state pair")
        pair_certificate = item["certificate"]
        if (pair_certificate["left"] != states[pair[0]] or pair_certificate["right"] != states[pair[1]]
                or pair_certificate["action_lower"] != certificate["action_lower"]
                or pair_certificate["action_upper"] != certificate["action_upper"]
                or pair_certificate["horizon"] != certificate["horizon"]
                or pair_certificate["epsilon"] != certificate["epsilon"]):
            raise ValueError("Pair certificate scope mismatch")
        result = verify_separation_certificate(system, pair_certificate)
        if result["status"] == "separated":
            adjacency[pair[0]].add(pair[1])
            adjacency[pair[1]].add(pair[0])
        actual_pairs.add(pair)
    if actual_pairs != expected_pairs:
        raise ValueError("Continuous pair certificate set is incomplete")
    bound, coloring = chromatic_number(adjacency)
    if certificate.get("bound") != bound or certificate.get("chromatic_coloring") != coloring:
        raise ValueError("Invalid continuous chromatic lower bound")
    return {"status": "verified", "states": len(states), "pairs": len(actual_pairs),
            "edges": sum(len(row) for row in adjacency) // 2, "lower_bound": bound}


def _candidate_upper_error(system: ContinuousReLUSystem, states: list[list[float]],
                           action_lower: np.ndarray, action_upper: np.ndarray,
                           candidate: dict) -> tuple[float, int]:
    outputs = [np.asarray(value, dtype=float) for value in candidate["outputs"]]
    encoding = [int(value) for value in candidate["encoding"]]
    if len(encoding) != len(states) or not outputs or any(not 0 <= value < len(outputs) for value in encoding):
        raise ValueError("Invalid continuous candidate encoding")
    if len(candidate.get("transition_trees", ())) != len(outputs):
        raise ValueError("A transition tree is required for every candidate state")
    maximum_error = 0.0
    leaves_verified = 0

    def output_error(low: np.ndarray, high: np.ndarray, center: np.ndarray) -> float:
        if low.shape != center.shape:
            raise ValueError("Candidate output dimension mismatch")
        return _upper(float(np.max(np.maximum(np.abs(low - center), np.abs(high - center)))))

    for state_index, state in enumerate(states):
        state_array = np.asarray(state, dtype=float)
        observed_low, observed_high = _network_interval(system.observation, state_array, state_array)
        maximum_error = max(maximum_error, output_error(observed_low, observed_high,
                                                         outputs[encoding[state_index]]))

    for source, tree in enumerate(candidate["transition_trees"]):
        members = [np.asarray(states[index], dtype=float) for index, value in enumerate(encoding) if value == source]

        def visit(node: dict, expected_low: np.ndarray, expected_high: np.ndarray) -> None:
            nonlocal maximum_error, leaves_verified
            low, high = np.asarray(node["low"], dtype=float), np.asarray(node["high"], dtype=float)
            if not np.array_equal(low, expected_low) or not np.array_equal(high, expected_high):
                raise ValueError("Candidate transition tree does not cover the action box")
            if node["kind"] == "split":
                dimension, value = int(node["dimension"]), float(node["value"])
                if not 0 <= dimension < low.size or not low[dimension] < value < high[dimension]:
                    raise ValueError("Invalid candidate action split")
                first_high, second_low = high.copy(), low.copy()
                first_high[dimension], second_low[dimension] = value, value
                if len(node.get("children", ())) != 2:
                    raise ValueError("A candidate split must have two children")
                visit(node["children"][0], low, first_high)
                visit(node["children"][1], second_low, high)
                return
            if node["kind"] != "leaf" or not 0 <= int(node["target"]) < len(outputs):
                raise ValueError("Invalid candidate transition leaf")
            target = outputs[int(node["target"])]
            for state in members:
                next_low, next_high = _network_interval(
                    system.transition, np.concatenate((state, low)), np.concatenate((state, high)))
                observed_low, observed_high = _network_interval(system.observation, next_low, next_high)
                maximum_error = max(maximum_error, output_error(observed_low, observed_high, target))
            leaves_verified += 1

        visit(tree, action_lower, action_upper)
    return maximum_error, leaves_verified


def certify_continuous_horizon_one_realization(system: ContinuousReLUSystem,
                                                lower_certificate: dict,
                                                candidate: dict) -> dict:
    """Combine a chromatic lower proof with a uniform H=1 candidate upper proof."""
    lower = verify_incompatibility_lower_bound(system, lower_certificate)
    if int(lower_certificate["horizon"]) != 1:
        raise ValueError("The continuous candidate verifier currently supports horizon one")
    maximum_error, leaves = _candidate_upper_error(
        system, lower_certificate["states"], np.asarray(lower_certificate["action_lower"], dtype=float),
        np.asarray(lower_certificate["action_upper"], dtype=float), candidate)
    epsilon = float(lower_certificate["epsilon"])
    if maximum_error > epsilon:
        raise ValueError("Continuous candidate exceeds epsilon")
    upper = len(candidate["outputs"])
    return {"schema": "ncd.continuous-realization-certificate.v1",
            "system_sha256": _digest(system.to_dict()), "lower": lower_certificate,
            "candidate": candidate, "lower_bound": lower["lower_bound"], "upper_bound": upper,
            "maximum_verified_error": maximum_error, "transition_leaves": leaves,
            "minimal": lower["lower_bound"] == upper,
            "scope": "declared finite initial-state collection, continuous action box, horizon one"}


def verify_continuous_realization(system: ContinuousReLUSystem, certificate: dict) -> dict:
    if certificate.get("schema") != "ncd.continuous-realization-certificate.v1":
        raise ValueError("Unsupported continuous realization certificate")
    if certificate.get("system_sha256") != _digest(system.to_dict()):
        raise ValueError("Continuous realization system mismatch")
    lower_certificate = certificate["lower"]
    lower = verify_incompatibility_lower_bound(system, lower_certificate)
    maximum_error, leaves = _candidate_upper_error(
        system, lower_certificate["states"], np.asarray(lower_certificate["action_lower"], dtype=float),
        np.asarray(lower_certificate["action_upper"], dtype=float), certificate["candidate"])
    epsilon = float(lower_certificate["epsilon"])
    upper = len(certificate["candidate"]["outputs"])
    expected = {"lower_bound": lower["lower_bound"], "upper_bound": upper,
                "maximum_verified_error": maximum_error, "transition_leaves": leaves,
                "minimal": lower["lower_bound"] == upper}
    if maximum_error > epsilon:
        raise ValueError("Continuous candidate exceeds epsilon")
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(f"Invalid continuous realization field: {key}")
    return {"status": "verified", **expected}


@dataclass
class ContinuousSeparationConfig:
    horizon: int = 1
    full_leaves: int = 64
    limited_leaves: int = 4

    @classmethod
    def quick(cls) -> "ContinuousSeparationConfig":
        return cls(full_leaves=32)

    def validate(self) -> None:
        if self.horizon != 1 or self.full_leaves < 16 or not 1 <= self.limited_leaves < 16:
            raise ValueError("This frozen benchmark requires horizon=1, full_leaves>=16, limited_leaves<16")


def benchmark_continuous_system() -> ContinuousReLUSystem:
    """Stable neural dynamics with a continuous demand intervention."""
    transition = ReLUMLP(
        (((1.0, 0.0), (0.0, 1.0)), ((0.5, 0.4),)),
        ((0.0, 0.0), (0.0,)),
    )
    observation = ReLUMLP((((1.0,),), ((1.0,),)), ((0.0,), (0.0,)))
    return ContinuousReLUSystem(1, 1, transition, observation, ("demand",))


def _benchmark_queries(config: ContinuousSeparationConfig) -> list[dict]:
    return [
        {"name": "separated", "left": (0.1,), "right": (0.9,), "epsilon": 0.3,
         "max_leaves": config.full_leaves, "expected": "separated"},
        {"name": "reflexive", "left": (0.2,), "right": (0.2,), "epsilon": 0.0,
         "max_leaves": config.full_leaves, "expected": "certified-within-epsilon"},
        {"name": "within", "left": (0.2,), "right": (0.25,), "epsilon": 0.03,
         "max_leaves": config.full_leaves, "expected": "certified-within-epsilon"},
        {"name": "budget_limited", "left": (0.2,), "right": (0.25,), "epsilon": 0.03,
         "max_leaves": config.limited_leaves, "expected": "unresolved"},
    ]


def run_continuous_separation(output: Path, config: ContinuousSeparationConfig) -> dict:
    """Create a deterministic continuous-oracle benchmark and its certificates."""
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    system = benchmark_continuous_system()
    save_json(output / "config.json", asdict(config))
    save_json(output / "system.json", system.to_dict())
    cases = []
    for specification in _benchmark_queries(config):
        certificate = certified_separation(
            system, specification["left"], specification["right"], (0.0,), (1.0,),
            config.horizon, specification["epsilon"], specification["max_leaves"])
        verification = verify_separation_certificate(system, certificate)
        if verification["status"] != specification["expected"]:
            raise RuntimeError(f"Unexpected benchmark status for {specification['name']}")
        case = {"name": specification["name"], "expected": specification["expected"],
                "certificate": certificate, "verification": verification}
        cases.append(case)
        save_json(output / "cases" / specification["name"] / "certificate.json", certificate)
    lower_certificate = certified_incompatibility_lower_bound(
        system, ((0.1,), (0.2,), (0.25,), (0.9,)), (0.0,), (1.0,),
        config.horizon, 0.03, config.full_leaves)
    lower_verification = verify_incompatibility_lower_bound(system, lower_certificate)
    save_json(output / "incompatibility_lower_bound.json", lower_certificate)
    realization_lower = certified_incompatibility_lower_bound(
        system, ((0.1,), (0.2,), (0.25,), (0.9,)), (0.0,), (1.0,),
        config.horizon, 0.3, config.full_leaves)
    candidate = {
        "outputs": [[0.23], [0.82]],
        "encoding": [0, 0, 0, 1],
        "transition_trees": [
            {"kind": "leaf", "low": [0.0], "high": [1.0], "target": 0},
            {"kind": "split", "low": [0.0], "high": [1.0], "dimension": 0,
             "value": 0.1875, "children": [
                 {"kind": "leaf", "low": [0.0], "high": [0.1875], "target": 0},
                 {"kind": "leaf", "low": [0.1875], "high": [1.0], "target": 1},
             ]},
        ],
    }
    realization = certify_continuous_horizon_one_realization(system, realization_lower, candidate)
    realization_verification = verify_continuous_realization(system, realization)
    save_json(output / "continuous_realization.json", realization)
    summary = {
        "schema": "ncd.continuous-separation-summary.v1",
        "cases": len(cases),
        "statuses": {status: sum(case["certificate"]["status"] == status for case in cases)
                     for status in ("separated", "certified-within-epsilon", "unresolved")},
        "all_certificates_verified": True,
        "declared_states": lower_verification["states"],
        "incompatibility_edges": lower_verification["edges"],
        "realization_complexity_lower_bound": lower_verification["lower_bound"],
        "closed_realization": {
            "epsilon": 0.3,
            "lower_bound": realization_verification["lower_bound"],
            "upper_bound": realization_verification["upper_bound"],
            "minimal": realization_verification["minimal"],
            "maximum_verified_error": realization_verification["maximum_verified_error"],
        },
        "metric": "maximum over time of L-infinity observation distance",
        "domain": {"action": "demand", "lower": 0.0, "upper": 1.0, "horizon": config.horizon},
        "scope": "fixed initial-state pairs; bounded continuous intervention words; finite horizon",
        "runtime_seconds": time.perf_counter() - started,
    }
    save_json(output / "summary.json", summary)
    artifacts = {path.relative_to(output).as_posix(): file_digest(path)
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    save_json(output / "manifest.json", {"schema": "ncd.continuous-separation-manifest.v1",
                                          "artifacts": artifacts})
    return summary


def verify_continuous_separation(output: Path) -> dict:
    """Check artifact integrity, certificate coverage, and frozen benchmark semantics."""
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.continuous-separation-manifest.v1":
        raise ValueError("Unsupported continuous separation manifest")
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Continuous separation artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if file_digest(output / relative) != expected:
            raise ValueError(f"Continuous artifact integrity failure: {relative}")
    config = ContinuousSeparationConfig(**read_json(output / "config.json"))
    config.validate()
    system = ContinuousReLUSystem.from_dict(read_json(output / "system.json"))
    if system.to_dict() != benchmark_continuous_system().to_dict():
        raise ValueError("Continuous benchmark system mismatch")
    statuses = {status: 0 for status in ("separated", "certified-within-epsilon", "unresolved")}
    leaves = 0
    for specification in _benchmark_queries(config):
        certificate = read_json(output / "cases" / specification["name"] / "certificate.json")
        result = verify_separation_certificate(system, certificate)
        if result["status"] != specification["expected"]:
            raise ValueError(f"Continuous benchmark semantic mismatch: {specification['name']}")
        statuses[result["status"]] += 1
        leaves += result["leaves_verified"]
    lower_result = verify_incompatibility_lower_bound(
        system, read_json(output / "incompatibility_lower_bound.json"))
    realization_result = verify_continuous_realization(
        system, read_json(output / "continuous_realization.json"))
    summary = read_json(output / "summary.json")
    if (summary.get("statuses") != statuses or not summary.get("all_certificates_verified")
            or summary.get("realization_complexity_lower_bound") != lower_result["lower_bound"]
            or summary.get("incompatibility_edges") != lower_result["edges"]
            or summary.get("closed_realization", {}).get("lower_bound") != realization_result["lower_bound"]
            or summary.get("closed_realization", {}).get("upper_bound") != realization_result["upper_bound"]
            or summary.get("closed_realization", {}).get("minimal") != realization_result["minimal"]):
        raise ValueError("Continuous separation summary mismatch")
    return {"status": "verified", "certificates_verified": sum(statuses.values()),
            "statuses": statuses, "leaves_verified": leaves,
            "incompatibility_lower_bound": lower_result,
            "continuous_realization": realization_result}
