"""Exact-rational multi-switch lower certificates for scalar affine ReLU systems."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
import json

from .continuous_separation import ContinuousReLUSystem, _digest
from .transition_overlap_lower import _parameters, _rational, benchmark_transition_lower_system
from .io import digest, read_json, save_json


@dataclass
class MultiSwitchConfig:
    epsilon: float = 0.101
    max_states: int = 20

    def validate(self) -> None:
        self.epsilon = float(self.epsilon)
        self.max_states = int(self.max_states)
        if not 0 < self.epsilon < 1 or self.max_states < 1:
            raise ValueError("Invalid multi-switch configuration")


def _line(lam: Fraction, beta: Fraction, epsilon: Fraction, states: int) -> dict:
    output_width = 2 * epsilon
    coverage = states * output_width
    redundancy = coverage - 1
    moving_width = lam / states
    target_capacity = output_width - moving_width
    target_count = ceil(beta / target_capacity) if target_capacity > 0 else None
    forced_redundancy = ((target_count - 1) * moving_width
                         if target_count is not None else None)
    if coverage < 1:
        reason = "output-cover-length"
    elif target_capacity <= 0:
        reason = "source-window-too-wide"
    elif target_count > states:
        reason = "insufficient-targets"
    elif redundancy < forced_redundancy:
        reason = "multi-switch-overlap"
    else:
        reason = "unresolved"
    return {
        "states": states, "reason": reason,
        "output_cover_capacity": _rational(coverage),
        "overlap_budget": _rational(redundancy),
        "source_window_width": _rational(moving_width),
        "single_target_shift_capacity": _rational(target_capacity),
        "required_target_count": target_count,
        "forced_total_overlap": (
            _rational(forced_redundancy) if forced_redundancy is not None else None),
    }


def _construct(system: ContinuousReLUSystem, config: MultiSwitchConfig) -> dict:
    config.validate()
    lam, beta, offset = _parameters(system)
    epsilon = Fraction(config.epsilon)
    exclusions = []
    first_unresolved = None
    for states in range(1, config.max_states + 1):
        line = _line(lam, beta, epsilon, states)
        if line["reason"] == "unresolved":
            first_unresolved = states
            break
        exclusions.append(line)
    return {
        "schema": "ncd.multiswitch-lower.v1",
        "system_sha256": _digest(system.to_dict()),
        "state_domain": [0.0, 1.0], "action_domain": [0.0, 1.0],
        "epsilon": config.epsilon, "max_states": config.max_states,
        "lambda": _rational(lam), "beta": _rational(beta),
        "offset": _rational(offset),
        "epsilon_exact_float": _rational(epsilon),
        "excluded_sizes": exclusions,
        "first_unresolved_size": first_unresolved,
        "lower_bound": len(exclusions) + 1,
        "status": "certified-lower-bound",
        "claim": "all smaller deterministic intervention-labelled finite realizations are impossible",
        "scope": "complete unit initial domain, continuous unit actions, arbitrary encoder and outputs, unbounded horizon",
    }


def certified_multiswitch_lower(system: ContinuousReLUSystem,
                                config: MultiSwitchConfig) -> dict:
    certificate = _construct(system, config)
    verify_multiswitch_lower(system, certificate)
    return certificate


def verify_multiswitch_lower(system: ContinuousReLUSystem, certificate: dict) -> dict:
    """Validate every excluded size independently using exact rational arithmetic."""
    if certificate.get("schema") != "ncd.multiswitch-lower.v1":
        raise ValueError("Unsupported multi-switch certificate")
    config = MultiSwitchConfig(
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
            "all smaller deterministic intervention-labelled finite realizations are impossible"
            or certificate.get("scope") !=
            "complete unit initial domain, continuous unit actions, arbitrary encoder and outputs, unbounded horizon"):
        raise ValueError("Multi-switch system or declared theorem scope mismatch")
    entries = certificate.get("excluded_sizes")
    if not isinstance(entries, list) or len(entries) > config.max_states:
        raise ValueError("Invalid multi-switch exclusion list")
    if certificate.get("lower_bound") != len(entries) + 1:
        raise ValueError("Multi-switch lower bound does not follow exclusions")
    multi_switch = 0
    for states, entry in enumerate(entries, start=1):
        verified = _line(lam, beta, epsilon, states)
        if verified["reason"] == "unresolved" or entry != verified:
            raise ValueError("A stored exact-rational multi-switch inequality failed")
        multi_switch += verified["reason"] == "multi-switch-overlap"
    next_size = len(entries) + 1
    if len(entries) < config.max_states:
        if _line(lam, beta, epsilon, next_size)["reason"] != "unresolved":
            raise ValueError("The first unresolved size is not actually unresolved")
        if certificate.get("first_unresolved_size") != next_size:
            raise ValueError("First unresolved size mismatch")
    elif certificate.get("first_unresolved_size") is not None:
        raise ValueError("Unresolved size lies outside the search budget")
    return {"status": "verified", "lower_bound": next_size,
            "excluded_sizes": len(entries),
            "multi_switch_exclusions": multi_switch,
            "arithmetic": "exact-rational"}


def run_multiswitch_lower(output: Path, system: ContinuousReLUSystem,
                          config: MultiSwitchConfig) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    save_json(output / "system.json", system.to_dict())
    certificate = certified_multiswitch_lower(system, config)
    save_json(output / "certificate.json", certificate)
    summary = verify_multiswitch_lower(system, certificate)
    save_json(output / "summary.json", summary)
    artifacts = {p.relative_to(output).as_posix(): digest(p)
                 for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    save_json(output / "manifest.json", {
        "schema": "ncd.multiswitch-lower-manifest.v1", "artifacts": artifacts})
    return summary


def verify_multiswitch_run(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.multiswitch-lower-manifest.v1":
        raise ValueError("Unsupported multi-switch manifest")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Multi-switch artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Multi-switch integrity failure: {relative}")
    system = ContinuousReLUSystem.from_dict(read_json(output / "system.json"))
    config = MultiSwitchConfig(**read_json(output / "config.json"))
    stored = read_json(output / "certificate.json")
    result = verify_multiswitch_lower(system, stored)
    replayed = certified_multiswitch_lower(system, config)
    if json.dumps(stored, sort_keys=True) != json.dumps(replayed, sort_keys=True):
        raise ValueError("Multi-switch deterministic replay mismatch")
    if result != read_json(output / "summary.json"):
        raise ValueError("Multi-switch summary mismatch")
    return result


def benchmark_multiswitch_system() -> ContinuousReLUSystem:
    return benchmark_transition_lower_system()

