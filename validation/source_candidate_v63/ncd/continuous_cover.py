"""Certified finite-horizon behavioral covers of bounded continuous state domains."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
import json
import time

import numpy as np

from .continuous_regions import (
    certified_region_separation,
    verify_region_separation_certificate,
)
from .continuous_scale import _tuple_network
from .continuous_separation import (
    ContinuousReLUSystem,
    certified_separation,
    verify_separation_certificate,
    _digest,
)
from .io import digest, read_json, save_json


@dataclass
class ContinuousCoverConfig:
    seed: int = 9701
    state_dims: tuple[int, ...] = (1, 2)
    bins_per_dim: int = 5
    horizon: int = 3
    epsilon: float = 0.101

    @classmethod
    def quick(cls, seed: int = 9701) -> "ContinuousCoverConfig":
        return cls(seed=seed, state_dims=(1,))

    def validate(self) -> None:
        self.state_dims = tuple(int(value) for value in self.state_dims)
        self.bins_per_dim = int(self.bins_per_dim)
        self.horizon = int(self.horizon)
        self.epsilon = float(self.epsilon)
        if (not self.state_dims or any(value < 1 for value in self.state_dims)
                or self.bins_per_dim < 2 or self.horizon < 1 or self.epsilon <= 0):
            raise ValueError("Invalid continuous cover configuration")
        cell_radius = 1.0 / (2.0 * self.bins_per_dim)
        packing_spacing = 1.0 / (self.bins_per_dim - 1)
        if cell_radius > self.epsilon or packing_spacing <= 2.0 * self.epsilon:
            raise ValueError("Grid does not support the declared matching cover and packing")


def benchmark_cover_system(state_dim: int) -> ContinuousReLUSystem:
    """Coordinate-wise contractive ReLU dynamics x'=0.5x+0.4a with identity output."""
    input_dim = 2 * state_dim
    first_weight = np.eye(input_dim)
    second_weight = np.zeros((state_dim, input_dim), dtype=np.float64)
    for index in range(state_dim):
        second_weight[index, index] = 0.5
        second_weight[index, state_dim + index] = 0.4
    transition = _tuple_network(
        [first_weight, second_weight],
        [np.ones(input_dim), np.full(state_dim, -0.9)],
    )
    observation = _tuple_network(
        [np.eye(state_dim), np.eye(state_dim)],
        [np.ones(state_dim), -np.ones(state_dim)],
    )
    return ContinuousReLUSystem(
        state_dim,
        state_dim,
        transition,
        observation,
        tuple(f"control_{index}" for index in range(state_dim)),
    )


def _parameter_count(system: ContinuousReLUSystem) -> int:
    return sum(
        np.asarray(layer).size for network in (system.transition, system.observation)
        for layer in network.weights
    ) + sum(
        len(layer) for network in (system.transition, system.observation)
        for layer in network.biases
    )


def certified_behavioral_cover(
    system: ContinuousReLUSystem,
    bins_per_dim: int,
    horizon: int,
    epsilon: float,
) -> dict:
    """Construct matching cover and packing certificates on the unit state cube."""
    dimension = system.state_dim
    if system.action_dim != dimension:
        raise ValueError("The frozen cover benchmark requires matching state and action dimensions")
    boundaries = np.linspace(0.0, 1.0, bins_per_dim + 1)
    packing_axis = np.linspace(0.0, 1.0, bins_per_dim)
    action_lower = (0.0,) * system.action_dim
    action_upper = (1.0,) * system.action_dim

    cells = []
    for index in product(range(bins_per_dim), repeat=dimension):
        low = np.asarray([boundaries[value] for value in index], dtype=np.float64)
        high = np.asarray([boundaries[value + 1] for value in index], dtype=np.float64)
        representative = (low + high) / 2.0
        certificate = certified_region_separation(
            system,
            low,
            high,
            representative,
            representative,
            action_lower,
            action_upper,
            horizon,
            epsilon / 2.0,
            max_leaves=1,
            bound_method="relational-stable",
            split_strategy="widest",
        )
        if certificate["status"] != "certified-within-epsilon":
            raise RuntimeError(f"Cover cell did not certify: {index}")
        verify_region_separation_certificate(system, certificate)
        cells.append({
            "index": list(index),
            "low": low.tolist(),
            "high": high.tolist(),
            "representative": representative.tolist(),
            "certificate": certificate,
        })

    packing_points = [
        list(point)
        for point in product(packing_axis.tolist(), repeat=dimension)
    ]
    pair_certificates = []
    for left_index, right_index in combinations(range(len(packing_points)), 2):
        certificate = certified_separation(
            system,
            packing_points[left_index],
            packing_points[right_index],
            action_lower,
            action_upper,
            horizon,
            epsilon,
            max_leaves=1,
            bound_method="relational-stable",
        )
        if certificate["status"] != "separated":
            raise RuntimeError("Packing pair did not separate")
        verify_separation_certificate(system, certificate)
        pair_certificates.append({
            "left": left_index,
            "right": right_index,
            "certificate": certificate,
        })

    lower_bound = len(packing_points)
    upper_bound = len(cells)
    return {
        "schema": "ncd.continuous-behavioral-cover.v1",
        "system_sha256": _digest(system.to_dict()),
        "domain_low": [0.0] * dimension,
        "domain_high": [1.0] * dimension,
        "action_lower": list(action_lower),
        "action_upper": list(action_upper),
        "horizon": horizon,
        "epsilon": epsilon,
        "bins_per_dim": bins_per_dim,
        "grid_boundaries": boundaries.tolist(),
        "cells": cells,
        "packing_points": packing_points,
        "packing_pair_certificates": pair_certificates,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "minimal": lower_bound == upper_bound,
        "claim": (
            "minimum number of epsilon response representatives for the declared "
            "finite-horizon unit-cube domain and complete action box"
        ),
        "boundary": (
            "finite-horizon behavioral cover; not a transition-closed quotient "
            "or an infinite-horizon realization"
        ),
    }


def verify_behavioral_cover(
    system: ContinuousReLUSystem,
    certificate: dict,
) -> dict:
    """Verify exact grid coverage, every cell upper proof, and complete packing."""
    if certificate.get("schema") != "ncd.continuous-behavioral-cover.v1":
        raise ValueError("Unsupported continuous behavioral cover")
    if certificate.get("system_sha256") != _digest(system.to_dict()):
        raise ValueError("Behavioral cover is for a different system")
    dimension = system.state_dim
    bins = int(certificate["bins_per_dim"])
    horizon = int(certificate["horizon"])
    epsilon = float(certificate["epsilon"])
    if (certificate.get("domain_low") != [0.0] * dimension
            or certificate.get("domain_high") != [1.0] * dimension
            or certificate.get("action_lower") != [0.0] * system.action_dim
            or certificate.get("action_upper") != [1.0] * system.action_dim):
        raise ValueError("Unexpected behavioral cover domain")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    if certificate.get("grid_boundaries") != boundaries.tolist():
        raise ValueError("Invalid behavioral cover grid")

    cells = certificate.get("cells", [])
    expected_indices = set(product(range(bins), repeat=dimension))
    actual_indices = {tuple(cell["index"]) for cell in cells}
    if actual_indices != expected_indices or len(cells) != len(expected_indices):
        raise ValueError("Behavioral cover cells do not tile the unit cube")
    cell_certificates = 0
    for cell in cells:
        index = tuple(cell["index"])
        low = [float(boundaries[value]) for value in index]
        high = [float(boundaries[value + 1]) for value in index]
        representative = [(left + right) / 2.0 for left, right in zip(low, high)]
        if (cell.get("low") != low or cell.get("high") != high
                or cell.get("representative") != representative):
            raise ValueError("Invalid behavioral cover cell geometry")
        region_certificate = cell["certificate"]
        if (region_certificate.get("left_low") != low
                or region_certificate.get("left_high") != high
                or region_certificate.get("right_low") != representative
                or region_certificate.get("right_high") != representative
                or region_certificate.get("action_lower") != certificate["action_lower"]
                or region_certificate.get("action_upper") != certificate["action_upper"]
                or region_certificate.get("epsilon") != epsilon / 2.0
                or region_certificate.get("horizon") != horizon):
            raise ValueError("Cell proof does not match its declared geometry")
        result = verify_region_separation_certificate(system, region_certificate)
        if result["status"] != "certified-within-epsilon":
            raise ValueError("Behavioral cover cell is not certified")
        cell_certificates += 1

    packing_axis = np.linspace(0.0, 1.0, bins).tolist()
    expected_points = [
        list(point) for point in product(packing_axis, repeat=dimension)
    ]
    points = certificate.get("packing_points")
    if points != expected_points:
        raise ValueError("Invalid behavioral packing points")
    expected_pairs = set(combinations(range(len(points)), 2))
    pair_entries = certificate.get("packing_pair_certificates", [])
    actual_pairs = {(int(entry["left"]), int(entry["right"])) for entry in pair_entries}
    if actual_pairs != expected_pairs or len(pair_entries) != len(expected_pairs):
        raise ValueError("Behavioral packing pair set is incomplete")
    for entry in pair_entries:
        left = int(entry["left"])
        right = int(entry["right"])
        pair_certificate = entry["certificate"]
        if (pair_certificate.get("left") != points[left]
                or pair_certificate.get("right") != points[right]
                or pair_certificate.get("action_lower") != certificate["action_lower"]
                or pair_certificate.get("action_upper") != certificate["action_upper"]
                or pair_certificate.get("epsilon") != epsilon
                or pair_certificate.get("horizon") != horizon):
            raise ValueError("Packing proof does not match its declared pair")
        result = verify_separation_certificate(system, pair_certificate)
        if result["status"] != "separated":
            raise ValueError("Behavioral packing pair is not separated")

    lower_bound = len(points)
    upper_bound = len(cells)
    expected = {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "minimal": lower_bound == upper_bound,
    }
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(f"Invalid behavioral cover aggregate field: {key}")
    return {
        "status": "verified",
        "state_dim": dimension,
        "cells_verified": cell_certificates,
        "packing_pairs_verified": len(pair_entries),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "minimal": lower_bound == upper_bound,
    }


def run_cover_profile(config: ContinuousCoverConfig, index: int) -> dict:
    dimension = config.state_dims[index]
    system = benchmark_cover_system(dimension)
    started = time.perf_counter()
    certificate = certified_behavioral_cover(
        system, config.bins_per_dim, config.horizon, config.epsilon)
    runtime = time.perf_counter() - started
    verification = verify_behavioral_cover(system, certificate)
    return {
        "profile": index,
        "seed": config.seed + index * 1009,
        "state_dim": dimension,
        "action_dim": dimension,
        "horizon": config.horizon,
        "epsilon": config.epsilon,
        "bins_per_dim": config.bins_per_dim,
        "parameter_count": _parameter_count(system),
        "system": system.to_dict(),
        "certificate": certificate,
        "verification": verification,
        "runtime_seconds": runtime,
    }


def summarize_behavioral_covers(profiles: list[dict]) -> dict:
    return {
        "schema": "ncd.continuous-behavioral-cover-summary.v1",
        "profiles": len(profiles),
        "largest_state_dim": max(profile["state_dim"] for profile in profiles),
        "largest_cover": max(profile["certificate"]["upper_bound"] for profile in profiles),
        "all_minimal": all(profile["certificate"]["minimal"] for profile in profiles),
        "total_cells_verified": sum(
            profile["verification"]["cells_verified"] for profile in profiles),
        "total_packing_pairs_verified": sum(
            profile["verification"]["packing_pairs_verified"] for profile in profiles),
        "scope": (
            "exact finite-horizon epsilon behavioral cover number on declared "
            "continuous unit cubes; complete bounded action words"
        ),
    }


def run_continuous_covers(output: Path, config: ContinuousCoverConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    profiles = []
    started = time.perf_counter()
    for index in range(len(config.state_dims)):
        profile = run_cover_profile(config, index)
        profiles.append(profile)
        directory = output / "profiles" / f"profile_{index:03d}"
        save_json(directory / "system.json", profile["system"])
        save_json(directory / "profile.json", profile)
    summary = summarize_behavioral_covers(profiles)
    summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    artifacts = {
        path.relative_to(output).as_posix(): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    save_json(output / "manifest.json", {
        "schema": "ncd.continuous-behavioral-cover-manifest.v1",
        "artifacts": artifacts,
    })
    return summary


def _without_cover_runtime(profile: dict) -> str:
    profile = json.loads(json.dumps(profile))
    profile.pop("runtime_seconds", None)
    return json.dumps(
        profile, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_continuous_covers(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.continuous-behavioral-cover-manifest.v1":
        raise ValueError("Unsupported continuous behavioral cover manifest")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("Continuous behavioral cover artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Continuous behavioral cover integrity failure: {relative}")

    config = ContinuousCoverConfig(**read_json(output / "config.json"))
    config.validate()
    replayed_profiles = []
    certificates = 0
    for index in range(len(config.state_dims)):
        directory = output / "profiles" / f"profile_{index:03d}"
        stored = read_json(directory / "profile.json")
        system = ContinuousReLUSystem.from_dict(read_json(directory / "system.json"))
        if system.to_dict() != stored["system"]:
            raise ValueError("Continuous behavioral cover stored system mismatch")
        verification = verify_behavioral_cover(system, stored["certificate"])
        certificates += verification["cells_verified"]
        certificates += verification["packing_pairs_verified"]
        replayed = run_cover_profile(config, index)
        if _without_cover_runtime(replayed) != _without_cover_runtime(stored):
            raise ValueError(f"Continuous behavioral cover replay mismatch: {index}")
        replayed_profiles.append(replayed)
    replayed_summary = summarize_behavioral_covers(replayed_profiles)
    stored_summary = read_json(output / "summary.json")
    stored_summary.pop("runtime_seconds", None)
    if replayed_summary != stored_summary:
        raise ValueError("Continuous behavioral cover summary mismatch")
    return {
        "status": "verified",
        "profiles_replayed": len(replayed_profiles),
        "certificates_verified": certificates,
        "largest_state_dim": max(config.state_dims),
        "largest_cover": max(
            profile["certificate"]["upper_bound"] for profile in replayed_profiles),
    }
