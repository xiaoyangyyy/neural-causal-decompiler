"""Certified response separation for continuous initial-state regions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json
import time

import numpy as np

from .continuous_separation import (
    ContinuousReLUSystem,
    _abs_difference_bounds,
    _digest,
    _network_interval,
    _shared_network_difference_interval,
    _upper,
)
from .continuous_scale import _state_pairs, train_scaled_dynamics
from .io import digest, read_json, save_json


def _box(value: Iterable[float], dimension: int, name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name} box endpoint")
    return result


def _region_domain(
    system: ContinuousReLUSystem,
    left_low: Iterable[float],
    left_high: Iterable[float],
    right_low: Iterable[float],
    right_high: Iterable[float],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_low = _box(left_low, system.state_dim, "left-low")
    left_high = _box(left_high, system.state_dim, "left-high")
    right_low = _box(right_low, system.state_dim, "right-low")
    right_high = _box(right_high, system.state_dim, "right-high")
    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)
    if (action_low.shape != action_high.shape or action_low.ndim != 2
            or action_low.shape[1] != system.action_dim
            or not np.isfinite(action_low).all() or not np.isfinite(action_high).all()
            or np.any(left_low > left_high) or np.any(right_low > right_high)
            or np.any(action_low > action_high)):
        raise ValueError("Invalid continuous region domain")
    return left_low, left_high, right_low, right_high, action_low, action_high


def region_response_distance_bounds(
    system: ContinuousReLUSystem,
    left_low: Iterable[float],
    left_high: Iterable[float],
    right_low: Iterable[float],
    right_high: Iterable[float],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[float, float]:
    """Bound max-time L-infinity distance over two state boxes and one action box."""
    (left_low, left_high, right_low, right_high,
     action_low, action_high) = _region_domain(
        system, left_low, left_high, right_low, right_high, action_low, action_high)
    overall_lower, overall_upper = 0.0, 0.0
    for time_index in range(len(action_low) + 1):
        output_left_low, output_left_high = _network_interval(
            system.observation, left_low, left_high)
        output_right_low, output_right_high = _network_interval(
            system.observation, right_low, right_high)
        lower, upper = _abs_difference_bounds(
            output_left_low, output_left_high, output_right_low, output_right_high)
        overall_lower = max(overall_lower, lower)
        overall_upper = max(overall_upper, upper)
        if time_index < len(action_low):
            left_low, left_high = _network_interval(
                system.transition,
                np.concatenate((left_low, action_low[time_index])),
                np.concatenate((left_high, action_high[time_index])),
            )
            right_low, right_high = _network_interval(
                system.transition,
                np.concatenate((right_low, action_low[time_index])),
                np.concatenate((right_high, action_high[time_index])),
            )
    return overall_lower, overall_upper


def relational_region_response_distance_bounds(
    system: ContinuousReLUSystem,
    left_low: Iterable[float],
    left_high: Iterable[float],
    right_low: Iterable[float],
    right_high: Iterable[float],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[float, float]:
    """Preserve shared controls when every paired ReLU phase is certified stable."""
    (left_low, left_high, right_low, right_high,
     action_low, action_high) = _region_domain(
        system, left_low, left_high, right_low, right_high, action_low, action_high)
    difference_low = np.nextafter(left_low - right_high, -np.inf)
    difference_high = np.nextafter(left_high - right_low, np.inf)
    overall_lower, overall_upper = 0.0, 0.0
    for time_index in range(len(action_low) + 1):
        (_, _, _, _, output_difference_low,
         output_difference_high) = _shared_network_difference_interval(
            system.observation, left_low, left_high, right_low, right_high,
            difference_low, difference_high)
        lower_per_coordinate = np.where(
            output_difference_low > 0.0,
            output_difference_low,
            np.where(output_difference_high < 0.0, -output_difference_high, 0.0),
        )
        upper = float(np.max(np.maximum(
            np.abs(output_difference_low), np.abs(output_difference_high))))
        overall_lower = max(overall_lower, float(np.max(lower_per_coordinate)))
        overall_upper = max(overall_upper, _upper(upper))
        if time_index < len(action_low):
            zero_action = np.zeros(system.action_dim, dtype=np.float64)
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
    return overall_lower, overall_upper


def _flatten_domain(
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate((left_low, right_low, action_low.reshape(-1))),
        np.concatenate((left_high, right_high, action_high.reshape(-1))),
    )


def _unpack_domain(
    low: np.ndarray,
    high: np.ndarray,
    state_dim: int,
    action_dim: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_low, left_high = low[:state_dim], high[:state_dim]
    right_low, right_high = low[state_dim:2 * state_dim], high[state_dim:2 * state_dim]
    action_low = low[2 * state_dim:].reshape(horizon, action_dim)
    action_high = high[2 * state_dim:].reshape(horizon, action_dim)
    return left_low, left_high, right_low, right_high, action_low, action_high


def _region_leaf(
    system: ContinuousReLUSystem,
    low: np.ndarray,
    high: np.ndarray,
    horizon: int,
    bound_method: str,
) -> dict:
    domain = _unpack_domain(low, high, system.state_dim, system.action_dim, horizon)
    actual_method = bound_method
    if bound_method == "relational-stable":
        try:
            _, upper_bound = relational_region_response_distance_bounds(system, *domain)
        except ValueError:
            actual_method = "independent-ibp"
            _, upper_bound = region_response_distance_bounds(system, *domain)
    elif bound_method == "independent-ibp":
        _, upper_bound = region_response_distance_bounds(system, *domain)
    else:
        raise ValueError("Unknown continuous region bound method")
    return {
        "kind": "leaf",
        "low": low.tolist(),
        "high": high.tolist(),
        "upper_bound": upper_bound,
        "bound_method": actual_method,
    }


def _robust_witness_bound(
    system: ContinuousReLUSystem,
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
    actions: np.ndarray,
    bound_method: str,
) -> tuple[float, str]:
    actual_method = bound_method
    if bound_method == "relational-stable":
        try:
            lower, _ = relational_region_response_distance_bounds(
                system, left_low, left_high, right_low, right_high, actions, actions)
        except ValueError:
            actual_method = "independent-ibp"
            lower, _ = region_response_distance_bounds(
                system, left_low, left_high, right_low, right_high, actions, actions)
    elif bound_method == "independent-ibp":
        lower, _ = region_response_distance_bounds(
            system, left_low, left_high, right_low, right_high, actions, actions)
    else:
        raise ValueError("Unknown continuous region witness method")
    return lower, actual_method


def certified_region_separation(
    system: ContinuousReLUSystem,
    left_low: Iterable[float],
    left_high: Iterable[float],
    right_low: Iterable[float],
    right_high: Iterable[float],
    action_lower: Iterable[float],
    action_upper: Iterable[float],
    horizon: int,
    epsilon: float,
    max_leaves: int = 256,
    bound_method: str = "relational-stable",
    split_strategy: str = "widest",
) -> dict:
    """Certify uniform closeness or robust separation for two initial-state boxes."""
    action_lower = _box(action_lower, system.action_dim, "action-lower")
    action_upper = _box(action_upper, system.action_dim, "action-upper")
    if (horizon < 0 or epsilon < 0 or max_leaves < 1
            or split_strategy not in ("widest", "best-bound")):
        raise ValueError("Invalid continuous region configuration")
    action_low = np.tile(action_lower, (horizon, 1))
    action_high = np.tile(action_upper, (horizon, 1))
    (left_low, left_high, right_low, right_high,
     action_low, action_high) = _region_domain(
        system, left_low, left_high, right_low, right_high, action_low, action_high)
    root_low, root_high = _flatten_domain(
        left_low, left_high, right_low, right_high, action_low, action_high)
    root = _region_leaf(system, root_low, root_high, horizon, bound_method)
    leaves = [root]

    center_actions = (action_low + action_high) / 2.0
    witness_actions = center_actions
    robust_lower_bound, witness_bound_method = _robust_witness_bound(
        system, left_low, left_high, right_low, right_high, center_actions, bound_method)

    while len(leaves) < max_leaves:
        upper_bound = max(leaf["upper_bound"] for leaf in leaves)
        if robust_lower_bound > 2 * epsilon or upper_bound <= 2 * epsilon:
            break
        candidate = max(leaves, key=lambda leaf: leaf["upper_bound"])
        low = np.asarray(candidate["low"], dtype=np.float64)
        high = np.asarray(candidate["high"], dtype=np.float64)
        widths = high - low
        if not np.any(widths > 0):
            break
        dimensions = ([int(np.argmax(widths))] if split_strategy == "widest"
                      else [index for index, width in enumerate(widths) if width > 0])
        best = None
        for dimension in dimensions:
            split = float((low[dimension] + high[dimension]) / 2.0)
            first_high, second_low = high.copy(), low.copy()
            first_high[dimension] = split
            second_low[dimension] = split
            first = _region_leaf(system, low, first_high, horizon, bound_method)
            second = _region_leaf(system, second_low, high, horizon, bound_method)
            proposal = (max(first["upper_bound"], second["upper_bound"]),
                        dimension, split, first, second)
            if best is None or proposal[:2] < best[:2]:
                best = proposal
        _, dimension, split, first, second = best
        candidate.clear()
        candidate.update({
            "kind": "split",
            "dimension": dimension,
            "value": split,
            "low": low.tolist(),
            "high": high.tolist(),
            "children": [first, second],
        })
        leaves.remove(candidate)
        leaves.extend((first, second))

        if dimension >= 2 * system.state_dim:
            for leaf in (first, second):
                leaf_low = np.asarray(leaf["low"], dtype=np.float64)
                leaf_high = np.asarray(leaf["high"], dtype=np.float64)
                actions = ((leaf_low[2 * system.state_dim:]
                            + leaf_high[2 * system.state_dim:]) / 2.0).reshape(
                                horizon, system.action_dim)
                lower, actual_witness_method = _robust_witness_bound(
                    system, left_low, left_high, right_low, right_high, actions, bound_method)
                if lower > robust_lower_bound:
                    robust_lower_bound, witness_actions = lower, actions
                    witness_bound_method = actual_witness_method

    upper_bound = max(leaf["upper_bound"] for leaf in leaves)
    status = ("robustly-separated" if robust_lower_bound > 2 * epsilon
              else "certified-within-epsilon" if upper_bound <= 2 * epsilon
              else "unresolved")
    return {
        "schema": "ncd.continuous-region-certificate.v1",
        "system_sha256": _digest(system.to_dict()),
        "left_low": left_low.tolist(),
        "left_high": left_high.tolist(),
        "right_low": right_low.tolist(),
        "right_high": right_high.tolist(),
        "action_lower": action_lower.tolist(),
        "action_upper": action_upper.tolist(),
        "horizon": horizon,
        "epsilon": epsilon,
        "max_leaves": max_leaves,
        "bound_method": bound_method,
        "split_strategy": split_strategy,
        "robust_lower_bound": robust_lower_bound,
        "upper_bound": upper_bound,
        "status": status,
        "witness_actions": witness_actions.tolist(),
        "witness_bound_method": witness_bound_method,
        "leaves": len(leaves),
        "tree": root,
        "semantics": {
            "robustly-separated": "one stored action word separates every cross-region state pair",
            "certified-within-epsilon": "every cross-region state pair is within tolerance for every action word",
        },
    }


def verify_region_separation_certificate(
    system: ContinuousReLUSystem,
    certificate: dict,
) -> dict:
    """Independently recompute a continuous initial-region certificate."""
    if certificate.get("schema") != "ncd.continuous-region-certificate.v1":
        raise ValueError("Unsupported continuous region certificate")
    if certificate.get("system_sha256") != _digest(system.to_dict()):
        raise ValueError("Region certificate is for a different system")
    horizon = int(certificate["horizon"])
    epsilon = float(certificate["epsilon"])
    action_lower = _box(certificate["action_lower"], system.action_dim, "action-lower")
    action_upper = _box(certificate["action_upper"], system.action_dim, "action-upper")
    action_low = np.tile(action_lower, (horizon, 1))
    action_high = np.tile(action_upper, (horizon, 1))
    (left_low, left_high, right_low, right_high,
     action_low, action_high) = _region_domain(
        system,
        certificate["left_low"], certificate["left_high"],
        certificate["right_low"], certificate["right_high"],
        action_low, action_high,
    )
    root_low, root_high = _flatten_domain(
        left_low, left_high, right_low, right_high, action_low, action_high)
    verified_upper: list[float] = []

    def visit(node: dict, expected_low: np.ndarray, expected_high: np.ndarray) -> None:
        low = np.asarray(node["low"], dtype=np.float64)
        high = np.asarray(node["high"], dtype=np.float64)
        if not np.array_equal(low, expected_low) or not np.array_equal(high, expected_high):
            raise ValueError("Region certificate tree does not cover its declared box")
        if node["kind"] == "split":
            dimension = int(node["dimension"])
            value = float(node["value"])
            if (not 0 <= dimension < low.size
                    or not low[dimension] < value < high[dimension]
                    or len(node.get("children", ())) != 2):
                raise ValueError("Invalid continuous region branch split")
            first_high, second_low = high.copy(), low.copy()
            first_high[dimension] = value
            second_low[dimension] = value
            visit(node["children"][0], low, first_high)
            visit(node["children"][1], second_low, high)
            return
        if node["kind"] != "leaf":
            raise ValueError("Unknown continuous region tree node")
        domain = _unpack_domain(
            low, high, system.state_dim, system.action_dim, horizon)
        method = node.get("bound_method")
        if method == "relational-stable":
            _, upper = relational_region_response_distance_bounds(system, *domain)
        elif method == "independent-ibp":
            _, upper = region_response_distance_bounds(system, *domain)
        else:
            raise ValueError("Unknown continuous region leaf method")
        if float(node["upper_bound"]) != upper:
            raise ValueError("Invalid continuous region leaf upper bound")
        verified_upper.append(upper)

    visit(certificate["tree"], root_low, root_high)
    witness_actions = np.asarray(certificate["witness_actions"], dtype=np.float64)
    if (witness_actions.shape != action_low.shape
            or np.any(witness_actions < action_low)
            or np.any(witness_actions > action_high)):
        raise ValueError("Region witness lies outside the action domain")
    witness_bound_method = certificate.get("witness_bound_method")
    if witness_bound_method == "relational-stable":
        robust_lower_bound, _ = relational_region_response_distance_bounds(
            system, left_low, left_high, right_low, right_high,
            witness_actions, witness_actions)
    elif witness_bound_method == "independent-ibp":
        robust_lower_bound, _ = region_response_distance_bounds(
            system, left_low, left_high, right_low, right_high,
            witness_actions, witness_actions)
    else:
        raise ValueError("Unknown continuous region witness method")
    upper_bound = max(verified_upper)
    status = ("robustly-separated" if robust_lower_bound > 2 * epsilon
              else "certified-within-epsilon" if upper_bound <= 2 * epsilon
              else "unresolved")
    expected = {
        "robust_lower_bound": robust_lower_bound,
        "upper_bound": upper_bound,
        "status": status,
        "leaves": len(verified_upper),
    }
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(f"Invalid continuous region aggregate field: {key}")
    return {
        "status": status,
        "robust_lower_bound": robust_lower_bound,
        "upper_bound": upper_bound,
        "leaves_verified": len(verified_upper),
    }


@dataclass
class ContinuousRegionConfig:
    seed: int = 8701
    state_dims: tuple[int, ...] = (8, 32, 64)
    action_dims: tuple[int, ...] = (2, 3, 4)
    horizons: tuple[int, ...] = (3, 5, 10)
    region_radii: tuple[float, ...] = (0.001, 0.001, 0.001)
    leaf_budgets: tuple[int, ...] = (8, 8, 8)
    training_samples: int = 512

    @classmethod
    def quick(cls, seed: int = 8701) -> "ContinuousRegionConfig":
        return cls(
            seed=seed,
            state_dims=(8,),
            action_dims=(2,),
            horizons=(3,),
            region_radii=(0.001,),
            leaf_budgets=(8,),
            training_samples=256,
        )

    def validate(self) -> None:
        self.state_dims = tuple(int(value) for value in self.state_dims)
        self.action_dims = tuple(int(value) for value in self.action_dims)
        self.horizons = tuple(int(value) for value in self.horizons)
        self.region_radii = tuple(float(value) for value in self.region_radii)
        self.leaf_budgets = tuple(int(value) for value in self.leaf_budgets)
        lengths = {
            len(self.state_dims),
            len(self.action_dims),
            len(self.horizons),
            len(self.region_radii),
            len(self.leaf_budgets),
        }
        if lengths != {len(self.state_dims)} or not self.state_dims:
            raise ValueError("Region profile arrays must have matching non-zero lengths")
        if (any(value < 1 for value in self.state_dims)
                or any(value < 1 for value in self.action_dims)
                or any(value < 1 for value in self.horizons)
                or any(value <= 0 for value in self.region_radii)
                or any(value < 1 for value in self.leaf_budgets)
                or self.training_samples < 32):
            raise ValueError("Invalid continuous region profile configuration")


def run_region_profile(config: ContinuousRegionConfig, index: int) -> dict:
    state_dim = config.state_dims[index]
    action_dim = config.action_dims[index]
    horizon = config.horizons[index]
    radius = config.region_radii[index]
    leaf_budget = config.leaf_budgets[index]
    seed = config.seed + index * 1009
    system, training = train_scaled_dynamics(
        state_dim, action_dim, config.training_samples, seed)
    pairs = _state_pairs(state_dim, seed + 17)
    results = {}
    region_specs = {}
    for name, (left_center, right_center, epsilon) in pairs.items():
        left_center_array = np.asarray(left_center, dtype=np.float64)
        right_center_array = np.asarray(right_center, dtype=np.float64)
        left_low = left_center_array - radius
        left_high = left_center_array + radius
        right_low = right_center_array - radius
        right_high = right_center_array + radius
        region_specs[name] = {
            "left_low": left_low.tolist(),
            "left_high": left_high.tolist(),
            "right_low": right_low.tolist(),
            "right_high": right_high.tolist(),
            "epsilon": epsilon,
        }
        result = {"epsilon": epsilon}
        runtimes = {}
        for method, bound_method in (
            ("relational", "relational-stable"),
            ("independent", "independent-ibp"),
        ):
            started = time.perf_counter()
            certificate = certified_region_separation(
                system,
                left_low,
                left_high,
                right_low,
                right_high,
                (0.0,) * action_dim,
                (1.0,) * action_dim,
                horizon,
                epsilon,
                max_leaves=leaf_budget,
                bound_method=bound_method,
                split_strategy="widest",
            )
            runtimes[method] = time.perf_counter() - started
            result[method] = certificate
            result[method + "_verification"] = verify_region_separation_certificate(
                system, certificate)
        result["runtime_seconds"] = runtimes
        results[name] = result
    return {
        "profile": index,
        "seed": seed,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "region_radius": radius,
        "leaf_budget": leaf_budget,
        "training": training,
        "system": system.to_dict(),
        "regions": region_specs,
        "results": results,
    }


def summarize_continuous_regions(profiles: list[dict]) -> dict:
    methods = ("relational", "independent")
    return {
        "schema": "ncd.continuous-region-summary.v1",
        "profiles": len(profiles),
        "largest_state_dim": max(profile["state_dim"] for profile in profiles),
        "largest_horizon": max(profile["horizon"] for profile in profiles),
        "all_models_fit_max_error_below_1e-9": all(
            profile["training"]["fit_max_error"] < 1e-9 for profile in profiles),
        "statuses": {
            method: {
                status: sum(
                    result[method]["status"] == status
                    for profile in profiles
                    for result in profile["results"].values()
                )
                for status in (
                    "robustly-separated",
                    "certified-within-epsilon",
                    "unresolved",
                )
            }
            for method in methods
        },
        "scope": (
            "learned stable ReLU dynamics; continuous initial-state boxes; "
            "complete action boxes; finite horizons"
        ),
    }


def run_continuous_regions(output: Path, config: ContinuousRegionConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    profiles = []
    started = time.perf_counter()
    for index in range(len(config.state_dims)):
        profile = run_region_profile(config, index)
        profiles.append(profile)
        directory = output / "profiles" / f"profile_{index:03d}"
        save_json(directory / "system.json", profile["system"])
        save_json(directory / "profile.json", profile)
    summary = summarize_continuous_regions(profiles)
    summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    artifacts = {
        path.relative_to(output).as_posix(): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    save_json(output / "manifest.json", {
        "schema": "ncd.continuous-region-manifest.v1",
        "artifacts": artifacts,
    })
    return summary


def _without_region_runtime(profile: dict) -> str:
    profile = json.loads(json.dumps(profile))
    for result in profile["results"].values():
        result.pop("runtime_seconds", None)
    return json.dumps(
        profile, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_continuous_regions(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.continuous-region-manifest.v1":
        raise ValueError("Unsupported continuous region manifest")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("Continuous region artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Continuous region integrity failure: {relative}")

    config = ContinuousRegionConfig(**read_json(output / "config.json"))
    config.validate()
    replayed_profiles = []
    certificates = 0
    for index in range(len(config.state_dims)):
        directory = output / "profiles" / f"profile_{index:03d}"
        stored = read_json(directory / "profile.json")
        system = ContinuousReLUSystem.from_dict(read_json(directory / "system.json"))
        if system.to_dict() != stored["system"]:
            raise ValueError("Continuous region stored system mismatch")
        for result in stored["results"].values():
            for method in ("relational", "independent"):
                verify_region_separation_certificate(system, result[method])
                certificates += 1
        replayed = run_region_profile(config, index)
        if _without_region_runtime(replayed) != _without_region_runtime(stored):
            raise ValueError(f"Continuous region deterministic replay mismatch: {index}")
        replayed_profiles.append(replayed)

    replayed_summary = summarize_continuous_regions(replayed_profiles)
    stored_summary = read_json(output / "summary.json")
    stored_summary.pop("runtime_seconds", None)
    if replayed_summary != stored_summary:
        raise ValueError("Continuous region summary mismatch")
    return {
        "status": "verified",
        "profiles_replayed": len(replayed_profiles),
        "certificates_verified": certificates,
        "largest_state_dim": max(config.state_dims),
        "largest_horizon": max(config.horizons),
    }
