"""Scaling study for certified continuous neural-state separation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import html
import json
import time

import numpy as np

from .continuous_separation import (
    ContinuousReLUSystem,
    ReLUMLP,
    certified_separation,
    verify_separation_certificate,
)
from .io import digest, read_json, save_json


ACTION_NAMES = ("demand", "incident", "signal", "closure", "external_inflow")


@dataclass
class ContinuousScaleConfig:
    seed: int = 4701
    state_dims: tuple[int, ...] = (8, 32, 64, 128)
    action_dims: tuple[int, ...] = (2, 3, 4, 5)
    horizons: tuple[int, ...] = (3, 5, 10, 20)
    training_samples: int = 2048
    independent_leaves: int = 8
    sample_words: int = 64

    @classmethod
    def quick(cls, seed: int = 4701) -> "ContinuousScaleConfig":
        return cls(seed=seed, state_dims=(8, 16), action_dims=(2, 2), horizons=(3, 5),
                   training_samples=256, independent_leaves=4, sample_words=16)

    def validate(self) -> None:
        self.state_dims = tuple(int(value) for value in self.state_dims)
        self.action_dims = tuple(int(value) for value in self.action_dims)
        self.horizons = tuple(int(value) for value in self.horizons)
        if not self.state_dims or not (len(self.state_dims) == len(self.action_dims) == len(self.horizons)):
            raise ValueError("Scale profiles must have matching non-empty dimensions")
        if any(value < 2 for value in self.state_dims) or any(not 1 <= value <= len(ACTION_NAMES) for value in self.action_dims):
            raise ValueError("Invalid continuous scale dimensions")
        if any(value < 1 for value in self.horizons) or self.training_samples < 32:
            raise ValueError("Invalid horizon or training sample count")
        if self.independent_leaves < 1 or self.sample_words < 1:
            raise ValueError("Invalid scale evaluation budget")


def _tuple_network(weights: list[np.ndarray], biases: list[np.ndarray]) -> ReLUMLP:
    return ReLUMLP(
        tuple(tuple(tuple(float(x) for x in row) for row in layer) for layer in weights),
        tuple(tuple(float(x) for x in layer) for layer in biases),
    )


def train_scaled_dynamics(state_dim: int, action_dim: int, samples: int, seed: int) -> tuple[ContinuousReLUSystem, dict]:
    """Fit a stable affine traffic-like dynamics model represented by a ReLU MLP."""
    rng = np.random.default_rng(seed)
    teacher_a = np.zeros((state_dim, state_dim), dtype=float)
    np.fill_diagonal(teacher_a, 0.45)
    for row in range(state_dim):
        teacher_a[row, (row - 1) % state_dim] = 0.05
    teacher_a[0, -1] = 0.30
    teacher_b = rng.uniform(0.01, 0.03, size=(state_dim, action_dim))
    teacher_c = rng.uniform(0.01, 0.03, size=state_dim)

    states = rng.uniform(0.1, 0.6, size=(samples, state_dim))
    actions = rng.uniform(0.0, 1.0, size=(samples, action_dim))
    targets = states @ teacher_a.T + actions @ teacher_b.T + teacher_c
    design = np.concatenate((states, actions, np.ones((samples, 1))), axis=1)
    fitted, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    fitted = np.round(fitted, 12)
    fitted_a = fitted[:state_dim].T
    fitted_b = fitted[state_dim:state_dim + action_dim].T
    fitted_c = fitted[-1]

    input_dim = state_dim + action_dim
    first_weight = np.eye(input_dim, dtype=float)
    first_bias = np.ones(input_dim, dtype=float)
    second_weight = np.concatenate((fitted_a, fitted_b), axis=1)
    second_bias = fitted_c - second_weight @ np.ones(input_dim)
    transition = _tuple_network([first_weight, second_weight], [first_bias, second_bias])

    output_dim = min(4, state_dim)
    observation_first = np.eye(state_dim, dtype=float)
    observation_second = np.zeros((output_dim, state_dim), dtype=float)
    observation_second[:, :output_dim] = np.eye(output_dim)
    observation = _tuple_network(
        [observation_first, observation_second],
        [np.ones(state_dim), -np.ones(output_dim)],
    )
    system = ContinuousReLUSystem(
        state_dim, action_dim, transition, observation, ACTION_NAMES[:action_dim])

    predictions = np.asarray([system.step(state, action) for state, action in zip(states, actions)])
    residual = predictions - targets
    training = {
        "seed": seed,
        "samples": samples,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_width": input_dim,
        "parameter_count": sum(np.asarray(layer).size for layer in transition.weights)
        + sum(len(layer) for layer in transition.biases)
        + sum(np.asarray(layer).size for layer in observation.weights)
        + sum(len(layer) for layer in observation.biases),
        "fit_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "fit_max_error": float(np.max(np.abs(residual))),
    }
    return system, training


def _state_pairs(state_dim: int, seed: int) -> dict[str, tuple[tuple[float, ...], tuple[float, ...], float]]:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.4, size=state_dim)
    near = base.copy()
    near[-1] += 0.05
    far = base.copy()
    far[-1] += 0.8
    return {
        "near": (tuple(base), tuple(near), 0.01),
        "far": (tuple(base), tuple(far), 0.10),
    }


def _sample_lower(system: ContinuousReLUSystem, left: tuple[float, ...], right: tuple[float, ...],
                  horizon: int, words: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    best = -1.0
    witness = None
    for _ in range(words):
        actions = rng.uniform(0.0, 1.0, size=(horizon, system.action_dim))
        distance = float(np.max(np.abs(system.response(left, actions) - system.response(right, actions))))
        if distance > best:
            best, witness = distance, actions.tolist()
    return {"words": words, "lower_bound_only": best, "witness_actions": witness,
            "certifies_absence": False}


def run_scale_profile(config: ContinuousScaleConfig, index: int) -> dict:
    state_dim = config.state_dims[index]
    action_dim = config.action_dims[index]
    horizon = config.horizons[index]
    seed = config.seed + index * 1009
    system, training = train_scaled_dynamics(state_dim, action_dim, config.training_samples, seed)
    pairs = _state_pairs(state_dim, seed + 17)
    action_lower, action_upper = (0.0,) * action_dim, (1.0,) * action_dim
    results = {}
    for pair_index, (name, (left, right, epsilon)) in enumerate(pairs.items()):
        started = time.perf_counter()
        relational = certified_separation(
            system, left, right, action_lower, action_upper, horizon, epsilon,
            max_leaves=1, bound_method="relational-stable")
        relational_seconds = time.perf_counter() - started
        relational_check = verify_separation_certificate(system, relational)
        if relational["tree"]["bound_method"] != "relational-stable":
            raise RuntimeError("Scale profile unexpectedly lost stable relational bounds")
        started = time.perf_counter()
        independent = certified_separation(
            system, left, right, action_lower, action_upper, horizon, epsilon,
            max_leaves=config.independent_leaves, bound_method="independent-ibp")
        independent_seconds = time.perf_counter() - started
        independent_check = verify_separation_certificate(system, independent)
        sampling = _sample_lower(system, left, right, horizon, config.sample_words,
                                 seed + 101 + pair_index)
        results[name] = {
            "epsilon": epsilon,
            "relational": relational,
            "relational_verification": relational_check,
            "independent": independent,
            "independent_verification": independent_check,
            "sampling": sampling,
            "runtime_seconds": {
                "relational": relational_seconds,
                "independent": independent_seconds,
            },
        }
    if results["near"]["relational"]["status"] != "certified-within-epsilon":
        raise RuntimeError("Relational scale case did not certify the near pair")
    if results["far"]["relational"]["status"] != "separated":
        raise RuntimeError("Relational scale case did not certify the far pair")
    return {
        "profile": index,
        "seed": seed,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "system": system.to_dict(),
        "training": training,
        "pairs": {name: {"left": list(value[0]), "right": list(value[1]), "epsilon": value[2]}
                  for name, value in pairs.items()},
        "results": results,
    }


def summarize_scale(profiles: list[dict]) -> dict:
    relational = [result["relational"] for profile in profiles for result in profile["results"].values()]
    independent = [result["independent"] for profile in profiles for result in profile["results"].values()]
    runtimes = [result["runtime_seconds"] for profile in profiles for result in profile["results"].values()]
    return {
        "schema": "ncd.continuous-scale-summary.v1",
        "profiles": len(profiles),
        "largest_state_dim": max(profile["state_dim"] for profile in profiles),
        "largest_action_dim": max(profile["action_dim"] for profile in profiles),
        "largest_horizon": max(profile["horizon"] for profile in profiles),
        "largest_parameter_count": max(profile["training"]["parameter_count"] for profile in profiles),
        "all_training_fit_max_error_below_1e-9": all(
            profile["training"]["fit_max_error"] < 1e-9 for profile in profiles),
        "relational_statuses": {status: sum(item["status"] == status for item in relational)
                                for status in ("separated", "certified-within-epsilon", "unresolved")},
        "independent_statuses": {status: sum(item["status"] == status for item in independent)
                                 for status in ("separated", "certified-within-epsilon", "unresolved")},
        "relational_mean_seconds": float(np.mean([item["relational"] for item in runtimes])),
        "independent_mean_seconds": float(np.mean([item["independent"] for item in runtimes])),
        "scope": "learned stable continuous ReLU dynamics; fixed state pairs; complete action boxes; finite horizons",
    }


def _report(summary: dict, profiles: list[dict]) -> str:
    rows = []
    for profile in profiles:
        near = profile["results"]["near"]
        rows.append(
            "<tr>"
            f"<td>{profile['state_dim']}</td><td>{profile['action_dim']}</td>"
            f"<td>{profile['horizon']}</td><td>{profile['training']['parameter_count']}</td>"
            f"<td>{near['relational']['upper_bound']:.6f}</td>"
            f"<td>{near['independent']['upper_bound']:.6f}</td>"
            f"<td>{html.escape(near['relational']['status'])}</td>"
            f"<td>{html.escape(near['independent']['status'])}</td></tr>"
        )
    return """<!doctype html><meta charset="utf-8"><title>Continuous certification scaling</title>
<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;line-height:1.5}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:7px;text-align:right}
th:first-child,td:first-child{text-align:left}</style>
<h1>Continuous certification scaling</h1>
<p>Stable relational propagation preserves shared controls through certified ReLU phases.</p>
<table><thead><tr><th>state dim</th><th>action dim</th><th>horizon</th><th>parameters</th>
<th>relational UB</th><th>independent UB</th><th>relational</th><th>independent</th></tr></thead>
<tbody>""" + "".join(rows) + "</tbody></table>" + (
        f"<p>Largest profile: {summary['largest_state_dim']} states, "
        f"horizon {summary['largest_horizon']}.</p>")


def run_continuous_scale(output: Path, config: ContinuousScaleConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    started = time.perf_counter()
    profiles = []
    for index in range(len(config.state_dims)):
        profile = run_scale_profile(config, index)
        profiles.append(profile)
        directory = output / "profiles" / f"profile_{index:03d}"
        save_json(directory / "system.json", profile["system"])
        save_json(directory / "profile.json", profile)
    summary = summarize_scale(profiles)
    summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    (output / "report.html").write_text(_report(summary, profiles), encoding="utf-8", newline="\n")
    artifacts = {path.relative_to(output).as_posix(): digest(path)
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    save_json(output / "manifest.json", {"schema": "ncd.continuous-scale-manifest.v1",
                                          "artifacts": artifacts})
    return summary


def _canonical_without_runtime(profile: dict) -> str:
    value = json.loads(json.dumps(profile))
    for result in value["results"].values():
        result.pop("runtime_seconds", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_continuous_scale(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.continuous-scale-manifest.v1":
        raise ValueError("Unsupported continuous scale manifest")
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Continuous scale artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Continuous scale integrity failure: {relative}")
    config = ContinuousScaleConfig(**read_json(output / "config.json"))
    config.validate()
    profiles = []
    certificates = 0
    for index in range(len(config.state_dims)):
        directory = output / "profiles" / f"profile_{index:03d}"
        stored = read_json(directory / "profile.json")
        system = ContinuousReLUSystem.from_dict(read_json(directory / "system.json"))
        if system.to_dict() != stored["system"]:
            raise ValueError("Stored scale system mismatch")
        for result in stored["results"].values():
            verify_separation_certificate(system, result["relational"])
            verify_separation_certificate(system, result["independent"])
            certificates += 2
        replayed = run_scale_profile(config, index)
        if _canonical_without_runtime(replayed) != _canonical_without_runtime(stored):
            raise ValueError(f"Continuous scale deterministic replay mismatch: {index}")
        profiles.append(replayed)
    replayed_summary = summarize_scale(profiles)
    stored_summary = read_json(output / "summary.json")
    for key in ("runtime_seconds", "relational_mean_seconds", "independent_mean_seconds"):
        stored_summary.pop(key, None)
        replayed_summary.pop(key, None)
    if replayed_summary != stored_summary:
        raise ValueError("Continuous scale summary mismatch")
    return {"status": "verified", "profiles_replayed": len(profiles),
            "certificates_verified": certificates,
            "largest_state_dim": max(config.state_dims), "largest_horizon": max(config.horizons)}
