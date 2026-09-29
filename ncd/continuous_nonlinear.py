"""Certified separation stress tests with control-dependent ReLU phase crossings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np

from .continuous_separation import ContinuousReLUSystem, ReLUMLP, certified_separation, verify_separation_certificate
from .continuous_scale import ACTION_NAMES, _sample_lower, _tuple_network
from .io import digest, read_json, save_json


@dataclass
class NonlinearContinuousConfig:
    seed: int = 5701
    state_dims: tuple[int, ...] = (16, 32, 64)
    action_dims: tuple[int, ...] = (2, 2, 3)
    horizons: tuple[int, ...] = (1, 2, 3)
    leaf_budgets: tuple[int, ...] = (80, 64, 24)
    near_epsilons: tuple[float, ...] = (0.03, 0.08, 0.12)
    training_samples: int = 1024
    sample_words: int = 64

    @classmethod
    def quick(cls, seed: int = 5701) -> "NonlinearContinuousConfig":
        return cls(seed=seed, state_dims=(16,), action_dims=(2,), horizons=(1,),
                   leaf_budgets=(80,), near_epsilons=(0.03,),
                   training_samples=256, sample_words=16)

    def validate(self) -> None:
        self.state_dims = tuple(int(x) for x in self.state_dims)
        self.action_dims = tuple(int(x) for x in self.action_dims)
        self.horizons = tuple(int(x) for x in self.horizons)
        self.leaf_budgets = tuple(int(x) for x in self.leaf_budgets)
        self.near_epsilons = tuple(float(x) for x in self.near_epsilons)
        lengths = {len(self.state_dims), len(self.action_dims), len(self.horizons),
                   len(self.leaf_budgets), len(self.near_epsilons)}
        if lengths != {len(self.state_dims)} or not self.state_dims:
            raise ValueError("Nonlinear profile arrays must have matching non-zero lengths")
        if any(d < 3 for d in self.state_dims) or any(not 2 <= a <= len(ACTION_NAMES) for a in self.action_dims):
            raise ValueError("Invalid nonlinear dimensions")
        if any(h < 1 for h in self.horizons) or any(b < 2 for b in self.leaf_budgets):
            raise ValueError("Invalid nonlinear horizons or budgets")
        if any(e <= 0 for e in self.near_epsilons):
            raise ValueError("Nonlinear epsilon values must be positive")
        if self.training_samples < 32 or self.sample_words < 1:
            raise ValueError("Invalid nonlinear sample counts")


def train_phase_crossing_dynamics(state_dim: int, action_dim: int, samples: int,
                                  seed: int) -> tuple[ContinuousReLUSystem, dict]:
    """Fit an output layer on fixed nonlinear features with known phase crossings."""
    rng = np.random.default_rng(seed)
    input_dim = state_dim + action_dim
    hidden = input_dim + 3
    first_weight = np.zeros((hidden, input_dim), dtype=float)
    first_bias = np.zeros(hidden, dtype=float)
    first_weight[:input_dim] = np.eye(input_dim)
    first_bias[:input_dim] = 1.0
    # The first two gates cross at different controls for nearby latent states.
    first_weight[input_dim, state_dim - 1] = 1.0
    first_weight[input_dim, state_dim] = 1.0
    first_bias[input_dim] = -0.8
    first_weight[input_dim + 1, state_dim - 1] = 1.0
    first_weight[input_dim + 1, state_dim] = 1.0
    first_bias[input_dim + 1] = -1.0
    first_weight[input_dim + 2, state_dim - 2] = 1.0
    first_weight[input_dim + 2, state_dim + 1] = -0.8
    first_bias[input_dim + 2] = -0.2

    teacher_second = np.zeros((state_dim, hidden), dtype=float)
    np.fill_diagonal(teacher_second[:, :state_dim], 0.42)
    for row in range(state_dim):
        teacher_second[row, (row - 1) % state_dim] += 0.04
    teacher_second[0, state_dim - 1] += 0.10
    teacher_second[:, state_dim:state_dim + action_dim] = rng.uniform(
        0.005, 0.02, size=(state_dim, action_dim))
    teacher_second[0, input_dim] = 0.25
    teacher_second[0, input_dim + 1] = -0.10
    teacher_second[1, input_dim + 2] = 0.08
    teacher_bias = rng.uniform(0.01, 0.03, size=state_dim)
    teacher_bias -= teacher_second[:, :input_dim] @ np.ones(input_dim)

    states = rng.uniform(0.15, 0.55, size=(samples, state_dim))
    actions = rng.uniform(0.0, 1.0, size=(samples, action_dim))
    inputs = np.concatenate((states, actions), axis=1)
    features = np.maximum(inputs @ first_weight.T + first_bias, 0.0)
    targets = features @ teacher_second.T + teacher_bias
    design = np.concatenate((features, np.ones((samples, 1))), axis=1)
    fitted, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    fitted = np.round(fitted, 12)
    second_weight = fitted[:-1].T
    second_bias = fitted[-1]
    transition = _tuple_network([first_weight, second_weight], [first_bias, second_bias])

    output_dim = min(4, state_dim)
    observation_first = np.eye(state_dim)
    observation_second = np.zeros((output_dim, state_dim))
    observation_second[:, :output_dim] = np.eye(output_dim)
    observation = _tuple_network([observation_first, observation_second],
                                 [np.ones(state_dim), -np.ones(output_dim)])
    system = ContinuousReLUSystem(state_dim, action_dim, transition, observation,
                                  ACTION_NAMES[:action_dim])
    predictions = np.asarray([system.step(state, action) for state, action in zip(states, actions)])
    error = predictions - targets
    return system, {
        "seed": seed, "samples": samples, "state_dim": state_dim,
        "action_dim": action_dim, "hidden_width": hidden,
        "parameter_count": sum(np.asarray(layer).size for layer in transition.weights)
        + sum(len(layer) for layer in transition.biases)
        + sum(np.asarray(layer).size for layer in observation.weights)
        + sum(len(layer) for layer in observation.biases),
        "fit_rmse": float(np.sqrt(np.mean(error ** 2))),
        "fit_max_error": float(np.max(np.abs(error))),
        "phase_crossing_gates": 3,
        "certified_gate_preactivation_intervals": [
            [-0.65, 0.75], [-0.85, 0.55], [-0.85, 0.35]],
    }


def _pairs(state_dim: int, seed: int, near_epsilon: float = 0.03) -> dict:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.25, 0.4, size=state_dim)
    base[-1] = 0.35
    base[-2] = 0.35
    near = base.copy(); near[-1] = 0.45
    far = base.copy(); far[-1] = 0.95
    return {"near": (tuple(base), tuple(near), near_epsilon),
            "far": (tuple(base), tuple(far), 0.08)}


def _leaf_methods(node: dict) -> dict[str, int]:
    if node["kind"] == "leaf":
        method = node.get("bound_method", "independent-ibp")
        return {"relational-stable": int(method == "relational-stable"),
                "independent-ibp": int(method == "independent-ibp")}
    totals = {"relational-stable": 0, "independent-ibp": 0}
    for child in node["children"]:
        values = _leaf_methods(child)
        for key in totals:
            totals[key] += values[key]
    return totals


def run_nonlinear_profile(config: NonlinearContinuousConfig, index: int) -> dict:
    state_dim, action_dim = config.state_dims[index], config.action_dims[index]
    horizon, budget = config.horizons[index], config.leaf_budgets[index]
    seed = config.seed + index * 1009
    system, training = train_phase_crossing_dynamics(
        state_dim, action_dim, config.training_samples, seed)
    pairs = _pairs(state_dim, seed + 19, config.near_epsilons[index])
    lower, upper = (0.0,) * action_dim, (1.0,) * action_dim
    results = {}
    for pair_index, (name, (left, right, epsilon)) in enumerate(pairs.items()):
        methods = {
            "hybrid_best": ("relational-stable", "best-bound"),
            "hybrid_widest": ("relational-stable", "widest"),
            "independent_best": ("independent-ibp", "best-bound"),
            "independent_widest": ("independent-ibp", "widest"),
        }
        certificates, verification, runtimes, leaf_methods = {}, {}, {}, {}
        for method, (bound_method, split_strategy) in methods.items():
            started = time.perf_counter()
            certificate = certified_separation(
                system, left, right, lower, upper, horizon, epsilon,
                max_leaves=budget, bound_method=bound_method, split_strategy=split_strategy)
            runtimes[method] = time.perf_counter() - started
            certificates[method] = certificate
            verification[method] = verify_separation_certificate(system, certificate)
            leaf_methods[method] = _leaf_methods(certificate["tree"])
        results[name] = {
            "epsilon": epsilon,
            **certificates,
            "verification": verification,
            "leaf_methods": leaf_methods,
            "sampling": _sample_lower(system, left, right, horizon, config.sample_words,
                                      seed + 211 + pair_index),
            "runtime_seconds": runtimes,
        }
    return {
        "profile": index, "seed": seed, "state_dim": state_dim, "action_dim": action_dim,
        "horizon": horizon, "leaf_budget": budget, "training": training,
        "system": system.to_dict(),
        "pairs": {name: {"left": list(value[0]), "right": list(value[1]), "epsilon": value[2]}
                  for name, value in pairs.items()},
        "results": results,
    }


def summarize_nonlinear(profiles: list[dict]) -> dict:
    methods = ("hybrid_best", "hybrid_widest", "independent_best", "independent_widest")
    return {
        "schema": "ncd.continuous-nonlinear-summary.v1",
        "profiles": len(profiles),
        "largest_state_dim": max(p["state_dim"] for p in profiles),
        "largest_horizon": max(p["horizon"] for p in profiles),
        "all_models_fit_max_error_below_1e-9": all(p["training"]["fit_max_error"] < 1e-9 for p in profiles),
        "all_training_domains_certify_phase_crossings": all(
            all(interval[0] < 0 < interval[1]
                for interval in profile["training"]["certified_gate_preactivation_intervals"])
            for profile in profiles),
        "statuses": {method: {status: sum(
            result[method]["status"] == status for profile in profiles for result in profile["results"].values())
            for status in ("separated", "certified-within-epsilon", "unresolved")} for method in methods},
        "hybrid_leaf_methods": {hybrid: {method: sum(
            result["leaf_methods"][hybrid][method]
            for profile in profiles for result in profile["results"].values())
            for method in ("relational-stable", "independent-ibp")}
            for hybrid in ("hybrid_best", "hybrid_widest")},
        "scope": "learned phase-crossing ReLU dynamics; fixed state pairs; complete action boxes; finite horizons",
    }


def run_continuous_nonlinear(output: Path, config: NonlinearContinuousConfig) -> dict:
    config.validate(); output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); save_json(output / "config.json", asdict(config))
    profiles = []
    started = time.perf_counter()
    for index in range(len(config.state_dims)):
        profile = run_nonlinear_profile(config, index); profiles.append(profile)
        directory = output / "profiles" / f"profile_{index:03d}"
        save_json(directory / "system.json", profile["system"])
        save_json(directory / "profile.json", profile)
    summary = summarize_nonlinear(profiles); summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    artifacts = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    save_json(output / "manifest.json", {"schema": "ncd.continuous-nonlinear-manifest.v1", "artifacts": artifacts})
    return summary


def _without_runtime(value: dict) -> str:
    value = json.loads(json.dumps(value))
    for result in value["results"].values(): result.pop("runtime_seconds", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_continuous_nonlinear(output: Path) -> dict:
    output = Path(output); manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.continuous-nonlinear-manifest.v1":
        raise ValueError("Unsupported nonlinear manifest")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]): raise ValueError("Nonlinear artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected: raise ValueError(f"Nonlinear integrity failure: {relative}")
    config = NonlinearContinuousConfig(**read_json(output / "config.json")); config.validate()
    profiles = []; certificates = 0
    for index in range(len(config.state_dims)):
        directory = output / "profiles" / f"profile_{index:03d}"
        stored = read_json(directory / "profile.json")
        system = ContinuousReLUSystem.from_dict(read_json(directory / "system.json"))
        if system.to_dict() != stored["system"]: raise ValueError("Nonlinear stored system mismatch")
        for result in stored["results"].values():
            for method in ("hybrid_best", "hybrid_widest", "independent_best", "independent_widest"):
                verify_separation_certificate(system, result[method]); certificates += 1
        replayed = run_nonlinear_profile(config, index)
        if _without_runtime(replayed) != _without_runtime(stored):
            raise ValueError(f"Nonlinear deterministic replay mismatch: {index}")
        profiles.append(replayed)
    replayed_summary = summarize_nonlinear(profiles)
    stored_summary = read_json(output / "summary.json"); stored_summary.pop("runtime_seconds", None)
    if replayed_summary != stored_summary: raise ValueError("Nonlinear summary mismatch")
    return {"status": "verified", "profiles_replayed": len(profiles),
            "certificates_verified": certificates, "largest_state_dim": max(config.state_dims),
            "largest_horizon": max(config.horizons)}
