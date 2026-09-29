"""Train phase-crossing queue dynamics, then certify the frozen ReLU network."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json

import numpy as np

from .continuous_scale import _tuple_network
from .continuous_separation import ContinuousReLUSystem, _digest
from .continuous_compositional_realization import certify_weighted, verify_weighted, value
from .io import read_json, save_json

STATE_KNOTS = (-0.1, 0.0, 0.1, 0.2)
ACTION_KNOTS = (-0.1, 0.0, 0.1, 0.2)
ALPHAS = (0.01, 0.001, 0.0001)


@dataclass
class NonlinearTrainingConfig:
    seed: int = 6101
    state_dim: int = 8
    train_samples: int = 2048
    selection_samples: int = 512
    test_samples: int = 1024
    rollout_cases: int = 64
    rollout_horizon: int = 10

    def validate(self) -> None:
        if (self.state_dim < 6 or self.train_samples < 128
                or self.selection_samples < 64 or self.test_samples < 64
                or self.rollout_cases < 1 or self.rollout_horizon < 1):
            raise ValueError("Invalid nonlinear training configuration")


def _teacher_coefficients(d: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(91000 + d)
    return {
        "constant": rng.uniform(0.078, 0.082, d),
        "self": rng.uniform(0.24, 0.28, d),
        "upstream": rng.uniform(0.04, 0.055, d),
        "demand": rng.uniform(0.105, 0.125, d),
        "signal": rng.uniform(0.035, 0.045, d),
        "queue_hinge": rng.uniform(0.022, 0.032, d),
        "control_hinge": rng.uniform(0.012, 0.020, d),
    }


def _teacher_step(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    d = states.shape[-1]
    c = _teacher_coefficients(d)
    upstream = np.roll(states, 1, axis=-1)
    demand = actions[..., 0, None]
    signal = actions[..., 1, None]
    return (c["constant"] + c["self"] * states + c["upstream"] * upstream
            + c["demand"] * demand - c["signal"] * signal
            + c["queue_hinge"] * np.maximum(upstream - states - 0.075, 0.0)
            + c["control_hinge"] * np.maximum(demand - signal - 0.125, 0.0))


def _samples(config: NonlinearTrainingConfig, tag: int, count: int):
    rng = np.random.default_rng(config.seed * 100 + tag)
    states = rng.uniform(0.0, 1.0, (count, config.state_dim))
    actions = rng.uniform(0.0, 1.0, (count, 2))
    targets = _teacher_step(states, actions)
    digest = hashlib.sha256(states.tobytes() + actions.tobytes() + targets.tobytes()).hexdigest()
    return states, actions, targets, digest


def _feature_layer(d: int):
    hidden = d + 2 + 4 * d + 4
    weights = np.zeros((hidden, d + 2))
    biases = np.zeros(hidden)
    weights[:d + 2, :] = np.eye(d + 2)
    state_base = d + 2
    for i in range(d):
        prev = (i - 1) % d
        for knot, threshold in enumerate(STATE_KNOTS):
            row = state_base + 4 * i + knot
            weights[row, prev] = 1.0
            weights[row, i] = -1.0
            biases[row] = -threshold
    action_base = state_base + 4 * d
    for knot, threshold in enumerate(ACTION_KNOTS):
        row = action_base + knot
        weights[row, d] = 1.0
        weights[row, d + 1] = -1.0
        biases[row] = -threshold
    return weights, biases


def _features(states: np.ndarray, actions: np.ndarray, weights, biases):
    joined = np.concatenate((states, actions), axis=1)
    return np.maximum(joined @ weights.T + biases, 0.0)


def _fit(d: int, features: np.ndarray, targets: np.ndarray, alpha: float,
         first_weights: np.ndarray, first_biases: np.ndarray) -> ContinuousReLUSystem:
    hidden = first_weights.shape[0]
    weights = np.zeros((d, hidden))
    biases = np.zeros(d)
    action_base = d + 2 + 4 * d
    for i in range(d):
        indices = [i, (i - 1) % d, d, d + 1]
        indices += list(range(d + 2 + 4 * i, d + 2 + 4 * (i + 1)))
        indices += list(range(action_base, action_base + 4))
        design = np.c_[features[:, indices], np.ones(len(features))]
        gram = design.T @ design
        penalty = np.eye(design.shape[1]) * alpha
        penalty[-1, -1] = 0.0
        fitted = np.linalg.solve(gram + penalty, design.T @ targets[:, i])
        fitted = np.round(fitted, 12)
        weights[i, indices] = fitted[:-1]
        biases[i] = fitted[-1]
    transition = _tuple_network([first_weights, weights], [first_biases, biases])
    observation = _tuple_network(
        [np.eye(4, d)], [np.zeros(4)])
    return ContinuousReLUSystem(
        d, 2, transition, observation, ("demand", "signal"))


def _predict(system: ContinuousReLUSystem, states: np.ndarray, actions: np.ndarray):
    current = np.concatenate((states, actions), axis=1)
    for layer, (weight, bias) in enumerate(zip(system.transition.weights,
                                                system.transition.biases)):
        current = current @ np.asarray(weight).T + np.asarray(bias)
        if layer + 1 < len(system.transition.weights):
            current = np.maximum(current, 0.0)
    return current


def _metrics(system: ContinuousReLUSystem, states, actions, targets):
    error = _predict(system, states, actions) - targets
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_error": float(np.max(np.abs(error)))}


def _rollout(config: NonlinearTrainingConfig, system: ContinuousReLUSystem):
    rng = np.random.default_rng(config.seed * 100 + 4)
    actual = rng.uniform(0.0, 1.0, (config.rollout_cases, config.state_dim))
    learned = actual.copy()
    errors = []
    for _ in range(config.rollout_horizon):
        actions = rng.uniform(0.0, 1.0, (config.rollout_cases, 2))
        actual = _teacher_step(actual, actions)
        learned = _predict(system, learned, actions)
        errors.append(np.max(np.abs(actual - learned), axis=1))
    return {"max_state_error": float(np.max(errors)),
            "mean_final_state_error": float(np.mean(errors[-1])),
            "teacher_domain_min": float(np.min(actual)),
            "teacher_domain_max": float(np.max(actual))}


def train_nonlinear(config: NonlinearTrainingConfig):
    """Training and hyperparameter choice use disjoint data; test is touched once."""
    config.validate()
    train_x, train_a, train_y, train_hash = _samples(config, 1, config.train_samples)
    select_x, select_a, select_y, select_hash = _samples(config, 2, config.selection_samples)
    test_x, test_a, test_y, test_hash = _samples(config, 3, config.test_samples)
    split_rows = [
        {row.tobytes() for row in np.concatenate((x, a), axis=1)}
        for x, a in ((train_x, train_a), (select_x, select_a), (test_x, test_a))]
    if any(split_rows[i] & split_rows[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Training, selection, and test input splits overlap")
    first_weights, first_biases = _feature_layer(config.state_dim)
    train_features = _features(train_x, train_a, first_weights, first_biases)
    candidates = []
    for alpha in ALPHAS:
        network = _fit(config.state_dim, train_features, train_y, alpha,
                       first_weights, first_biases)
        candidates.append((alpha, network,
                           _metrics(network, select_x, select_a, select_y)))
    alpha, system, selected = min(candidates, key=lambda item: item[2]["rmse"])
    result = {
        "config": asdict(config),
        "teacher": "ring queue with demand, signal, congestion and control hinges",
        "dictionary_state_knots": list(STATE_KNOTS),
        "dictionary_action_knots": list(ACTION_KNOTS),
        "candidate_ridge_alphas": list(ALPHAS),
        "chosen_ridge_alpha": alpha,
        "selection_candidates": [
            {"alpha": a, "rmse": metrics["rmse"], "max_error": metrics["max_error"]}
            for a, _, metrics in candidates],
        "training": _metrics(system, train_x, train_a, train_y),
        "selection": selected,
        "test": _metrics(system, test_x, test_a, test_y),
        "rollout_test": _rollout(config, system),
        "split_sha256": {"train": train_hash, "selection": select_hash, "test": test_hash},
        "splits_disjoint": True,
        "system_sha256": _digest(system.to_dict()),
    }
    return system, result


def phase_witnesses(system: ContinuousReLUSystem) -> dict[str, str]:
    d = system.state_dim
    base = [Q(2, 5)] * d + [Q(1, 2), Q(2, 5)]
    points = []
    for upstream in (Q(2, 5), Q(1, 2), Q(3, 5)):
        point = base.copy()
        point[d - 1] = upstream
        points.append(point)
    values = [value(system.transition, point)[0] for point in points]
    state_bend = values[1] * 2 - values[0] - values[2]
    points = []
    for demand in (Q(9, 20), Q(11, 20), Q(13, 20)):
        point = base.copy()
        point[d] = demand
        points.append(point)
    values = [value(system.transition, point)[0] for point in points]
    action_bend = values[1] * 2 - values[0] - values[2]
    return {"state_second_difference": str(state_bend),
            "control_second_difference": str(action_bend)}


def candidate_relation(d: int):
    return ((6, 5, 5, 5) + (1,) * (d - 5) + (3,),
            ("0.17",) * 4 + ("1.01",) * (d - 5) + ("0.4",))


def run_case(config: NonlinearTrainingConfig, output: Path) -> dict:
    system, training = train_nonlinear(config)
    bins, radii = candidate_relation(config.state_dim)
    certificate = certify_weighted(
        system, bins, radii, action_bins=128, epsilon="0.17", packing_axes=4)
    witnesses = phase_witnesses(system)
    if not all(abs(Q(value)) > Q(1, 10000) for value in witnesses.values()):
        raise RuntimeError("Frozen network did not exhibit both required phase crossings")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "system.json", system.to_dict())
    save_json(output / "training.json", training)
    save_json(output / "certificate.json", certificate)
    record = {"schema": "ncd.trained-nonlinear-global.v1",
              "training": training,
              "phase_witnesses": witnesses,
              "certificate_status": certificate["status"],
              "lower_bound": certificate["lower_bound"],
              "upper_bound": certificate["upper_bound"]}
    save_json(output / "record.json", record)
    return record


def verify_case(output: Path) -> dict:
    training = read_json(output / "training.json")
    config = NonlinearTrainingConfig(**training["config"])
    model, reproduced = train_nonlinear(config)
    stored_model = read_json(output / "system.json")
    if model.to_dict() != stored_model or training != reproduced:
        raise ValueError("Training or frozen model replay mismatch")
    certificate = read_json(output / "certificate.json")
    verify_weighted(model, certificate)
    witnesses = phase_witnesses(model)
    expected = {"schema": "ncd.trained-nonlinear-global.v1",
                "training": training, "phase_witnesses": witnesses,
                "certificate_status": certificate["status"],
                "lower_bound": certificate["lower_bound"],
                "upper_bound": certificate["upper_bound"]}
    if read_json(output / "record.json") != expected:
        raise ValueError("Nonlinear record replay mismatch")
    return expected



def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", type=int, default=6199)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.generate:
        config = NonlinearTrainingConfig(seed=args.seed, state_dim=args.dimension)
        if args.quick:
            config.train_samples = 256
            config.selection_samples = 64
            config.test_samples = 128
            config.rollout_cases = 8
            config.rollout_horizon = 4
        record = run_case(config, args.path)
    else:
        record = verify_case(args.path)
    print(json.dumps({
        "certificate_status": record["certificate_status"],
        "lower_bound": record["lower_bound"],
        "upper_bound": record["upper_bound"]}, sort_keys=True))


if __name__ == "__main__":
    main()

