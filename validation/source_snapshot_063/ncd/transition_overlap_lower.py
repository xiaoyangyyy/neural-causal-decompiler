"""Exact-rational transition-aware lower certificates for scalar affine ReLU systems."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
import json
import numpy as np

from .continuous_cover import benchmark_cover_system
from .continuous_separation import (
    ContinuousReLUSystem, ReLUMLP, _affine_interval, _digest,
)
from .io import digest, read_json, save_json


@dataclass
class TransitionLowerConfig:
    epsilon: float = 0.101
    max_states: int = 20

    def validate(self) -> None:
        self.epsilon = float(self.epsilon)
        self.max_states = int(self.max_states)
        if (not np.isfinite(self.epsilon) or self.epsilon <= 0
                or self.max_states < 1):
            raise ValueError("Invalid transition-lower configuration")


def _affine_coefficients(network: ReLUMLP) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Prove all hidden units active on the unit cube, then extract exact affine map."""
    if len(network.weights) != 2:
        raise ValueError("Transition-overlap theorem requires a two-layer ReLU MLP")
    low, high = _affine_interval(
        network.weights[0], network.biases[0],
        np.zeros(network.input_dim), np.ones(network.input_dim))
    if not np.all(low > 0.0):
        raise ValueError("Hidden ReLU phase is not strictly active on the domain")
    first = [[Fraction(value) for value in row] for row in network.weights[0]]
    second = [[Fraction(value) for value in row] for row in network.weights[1]]
    first_bias = [Fraction(value) for value in network.biases[0]]
    second_bias = [Fraction(value) for value in network.biases[1]]
    coefficients = [
        [sum((second[o][h] * first[h][i] for h in range(len(first))), Fraction(0))
         for i in range(network.input_dim)]
        for o in range(network.output_dim)]
    bias = [
        second_bias[o] + sum(
            (second[o][h] * first_bias[h] for h in range(len(first))), Fraction(0))
        for o in range(network.output_dim)]
    return coefficients, bias


def _parameters(system: ContinuousReLUSystem) -> tuple[Fraction, Fraction, Fraction]:
    if system.state_dim != 1 or system.action_dim != 1:
        raise ValueError("Transition-overlap theorem currently requires one state and one action")
    transition, bias = _affine_coefficients(system.transition)
    observation, output_bias = _affine_coefficients(system.observation)
    if observation != [[Fraction(1)]] or output_bias != [Fraction(0)]:
        raise ValueError("Transition-overlap theorem requires identity observation")
    lam, beta = transition[0]
    offset = bias[0]
    if lam <= 0 or beta <= 0 or offset < 0 or offset + lam + beta > 1:
        raise ValueError("Affine dynamics must preserve the unit state interval")
    return lam, beta, offset


def _rational(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _proof(system: ContinuousReLUSystem, config: TransitionLowerConfig) -> dict:
    config.validate()
    lam, beta, offset = _parameters(system)
    epsilon = Fraction(config.epsilon)
    exclusions = []
    for states in range(1, config.max_states + 1):
        cover_capacity = 2 * epsilon * states
        overlap_budget = cover_capacity - 1
        forced_overlap = lam / states
        sweep_span = beta + forced_overlap
        if cover_capacity < 1:
            reason = "output-cover-length"
        elif overlap_budget < forced_overlap and sweep_span > 2 * epsilon:
            reason = "deterministic-transition-overlap"
        else:
            break
        exclusions.append({
            "states": states, "reason": reason,
            "output_cover_capacity": _rational(cover_capacity),
            "overlap_budget": _rational(overlap_budget),
            "forced_target_overlap": _rational(forced_overlap),
            "moving_window_span": _rational(sweep_span),
        })
    return {
        "schema": "ncd.transition-overlap-lower.v1",
        "system_sha256": _digest(system.to_dict()),
        "state_domain": [0.0, 1.0], "action_domain": [0.0, 1.0],
        "epsilon": config.epsilon, "max_states": config.max_states,
        "lambda": _rational(lam), "beta": _rational(beta),
        "offset": _rational(offset),
        "epsilon_exact_float": _rational(epsilon),
        "excluded_sizes": exclusions,
        "lower_bound": len(exclusions) + 1,
        "status": "certified-lower-bound",
        "claim": "minimum deterministic intervention-labelled finite realization size on the complete unit interval",
        "scope": "all initial states, all continuous action words, all times, arbitrary encoder and abstract outputs",
    }


def certified_transition_lower(system: ContinuousReLUSystem,
                               config: TransitionLowerConfig) -> dict:
    certificate = _proof(system, config)
    verify_transition_lower(system, certificate)
    return certificate


def verify_transition_lower(system: ContinuousReLUSystem, certificate: dict) -> dict:
    """Check each excluded size from extracted network coefficients and exact fractions."""
    if certificate.get("schema") != "ncd.transition-overlap-lower.v1":
        raise ValueError("Unsupported transition-aware lower certificate")
    config = TransitionLowerConfig(
        epsilon=certificate["epsilon"], max_states=certificate["max_states"])
    config.validate()
    lam, beta, offset = _parameters(system)
    epsilon = Fraction(config.epsilon)
    if (certificate.get("system_sha256") != _digest(system.to_dict())
            or certificate.get("state_domain") != [0.0, 1.0]
            or certificate.get("action_domain") != [0.0, 1.0]
            or certificate.get("lambda") != _rational(lam)
            or certificate.get("beta") != _rational(beta)
            or certificate.get("offset") != _rational(offset)
            or certificate.get("epsilon_exact_float") != _rational(epsilon)
            or certificate.get("status") != "certified-lower-bound"
            or certificate.get("claim") !=
            "minimum deterministic intervention-labelled finite realization size on the complete unit interval"
            or certificate.get("scope") !=
            "all initial states, all continuous action words, all times, arbitrary encoder and abstract outputs"):
        raise ValueError("Transition-lower domain or extracted affine map mismatch")
    entries = certificate.get("excluded_sizes")
    if not isinstance(entries, list) or len(entries) > config.max_states:
        raise ValueError("Invalid transition-lower exclusion list")
    if certificate.get("lower_bound") != len(entries) + 1:
        raise ValueError("Transition-lower count does not follow exclusions")
    transition_aware = 0
    for states, entry in enumerate(entries, start=1):
        capacity = 2 * epsilon * states
        budget = capacity - 1
        forced = lam / states
        sweep = beta + forced
        if capacity < 1:
            reason = "output-cover-length"
        elif budget < forced and sweep > 2 * epsilon:
            reason = "deterministic-transition-overlap"
            transition_aware += 1
        else:
            raise ValueError(f"Size {states} is not excluded by the theorem")
        expected = {
            "states": states, "reason": reason,
            "output_cover_capacity": _rational(capacity),
            "overlap_budget": _rational(budget),
            "forced_target_overlap": _rational(forced),
            "moving_window_span": _rational(sweep),
        }
        if entry != expected:
            raise ValueError("Stored exact-rational inequality mismatch")
    return {"status": "verified", "lower_bound": len(entries) + 1,
            "excluded_sizes": len(entries),
            "transition_aware_exclusions": transition_aware,
            "arithmetic": "exact-rational"}

def run_transition_lower(output: Path, system: ContinuousReLUSystem,
                         config: TransitionLowerConfig) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    save_json(output / "system.json", system.to_dict())
    certificate = certified_transition_lower(system, config)
    save_json(output / "certificate.json", certificate)
    summary = verify_transition_lower(system, certificate)
    save_json(output / "summary.json", summary)
    artifacts = {p.relative_to(output).as_posix(): digest(p)
                 for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    save_json(output / "manifest.json", {
        "schema": "ncd.transition-overlap-lower-manifest.v1",
        "artifacts": artifacts})
    return summary


def verify_transition_lower_run(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.transition-overlap-lower-manifest.v1":
        raise ValueError("Unsupported transition-lower manifest")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Transition-lower artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Transition-lower integrity failure: {relative}")
    system = ContinuousReLUSystem.from_dict(read_json(output / "system.json"))
    config = TransitionLowerConfig(**read_json(output / "config.json"))
    stored = read_json(output / "certificate.json")
    result = verify_transition_lower(system, stored)
    regenerated = certified_transition_lower(system, config)
    if json.dumps(stored, sort_keys=True) != json.dumps(regenerated, sort_keys=True):
        raise ValueError("Transition-lower deterministic replay mismatch")
    if result != read_json(output / "summary.json"):
        raise ValueError("Transition-lower summary mismatch")
    return result


def benchmark_transition_lower_system() -> ContinuousReLUSystem:
    return benchmark_cover_system(1)

