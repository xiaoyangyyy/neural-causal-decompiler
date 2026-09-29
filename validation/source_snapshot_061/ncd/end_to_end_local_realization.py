"""Learn local ReLU directions, thresholds, and readouts; certify frozen dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .continuous_scale import _tuple_network
from .continuous_separation import ContinuousReLUSystem, _digest
from .continuous_compositional_realization import certify_weighted, verify_weighted
from .trained_nonlinear_realization import (
    NonlinearTrainingConfig, _samples, _teacher_step, _predict,
    phase_witnesses, candidate_relation,
)
from .io import read_json, save_json


@dataclass
class LocalLearningConfig:
    seed: int = 7101
    state_dim: int = 8
    train_samples: int = 2048
    selection_samples: int = 512
    test_samples: int = 1024
    epochs: int = 600
    rollout_cases: int = 64
    rollout_horizon: int = 10

    def validate(self) -> None:
        if (self.state_dim < 6 or self.train_samples < 128
                or self.selection_samples < 64 or self.test_samples < 64
                or self.epochs < 1 or self.rollout_cases < 1
                or self.rollout_horizon < 1):
            raise ValueError("Invalid local ReLU learning configuration")


def _base_config(config: LocalLearningConfig) -> NonlinearTrainingConfig:
    return NonlinearTrainingConfig(
        seed=config.seed, state_dim=config.state_dim,
        train_samples=config.train_samples,
        selection_samples=config.selection_samples,
        test_samples=config.test_samples,
        rollout_cases=config.rollout_cases,
        rollout_horizon=config.rollout_horizon)


def _local(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        states, torch.roll(states, 1, dims=1),
        actions[:, 0, None].expand_as(states),
        actions[:, 1, None].expand_as(states)), dim=-1)


def _as_network(d: int, first: np.ndarray, first_bias: np.ndarray,
                second: np.ndarray, direct: np.ndarray,
                output_bias: np.ndarray) -> ContinuousReLUSystem:
    hidden = 3 * d + 2
    w0 = np.zeros((hidden, d + 2))
    w0[:d + 2] = np.eye(d + 2)
    b0 = np.zeros(hidden)
    w1 = np.zeros((d, hidden))
    for i in range(d):
        w1[i, i] = direct[i, 0]
        w1[i, (i - 1) % d] += direct[i, 1]
        w1[i, d] = direct[i, 2]
        w1[i, d + 1] = direct[i, 3]
        for k in range(2):
            row = d + 2 + 2 * i + k
            w0[row, i] = first[i, k, 0]
            w0[row, (i - 1) % d] = first[i, k, 1]
            w0[row, d] = first[i, k, 2]
            w0[row, d + 1] = first[i, k, 3]
            b0[row] = first_bias[i, k]
            w1[i, row] = second[i, k]
    transition = _tuple_network([w0, w1], [b0, output_bias])
    observation = _tuple_network([np.eye(4, d)], [np.zeros(4)])
    return ContinuousReLUSystem(
        d, 2, transition, observation, ("demand", "signal"))


def _metrics(system: ContinuousReLUSystem, states, actions, targets) -> dict:
    error = _predict(system, states, actions) - targets
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_error": float(np.max(np.abs(error)))}


def _rollout(config: LocalLearningConfig, system: ContinuousReLUSystem) -> dict:
    rng = np.random.default_rng(config.seed * 100 + 4)
    teacher = rng.uniform(0.0, 1.0, (config.rollout_cases, config.state_dim))
    neural = teacher.copy()
    errors = []
    for _ in range(config.rollout_horizon):
        actions = rng.uniform(0.0, 1.0, (config.rollout_cases, 2))
        teacher = _teacher_step(teacher, actions)
        neural = _predict(system, neural, actions)
        errors.append(np.max(np.abs(teacher - neural), axis=1))
    return {"max_state_error": float(np.max(errors)),
            "mean_final_state_error": float(np.mean(errors[-1]))}


def train_local(config: LocalLearningConfig):
    config.validate()
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    base = _base_config(config)
    train_x, train_a, train_y, train_hash = _samples(base, 1, config.train_samples)
    val_x, val_a, val_y, val_hash = _samples(base, 2, config.selection_samples)
    test_x, test_a, test_y, test_hash = _samples(base, 3, config.test_samples)
    splits = [
        {row.tobytes() for row in np.concatenate((x, a), axis=1)}
        for x, a in ((train_x, train_a), (val_x, val_a), (test_x, test_a))]
    if any(splits[i] & splits[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Training, selection, and test inputs overlap")
    tensor = lambda x: torch.tensor(x, dtype=torch.float64)
    train_input = _local(tensor(train_x), tensor(train_a))
    val_input = _local(tensor(val_x), tensor(val_a))
    train_target = tensor(train_y)
    val_target = tensor(val_y)
    d = config.state_dim
    base_directions = np.array([[-1, 1, 0, 0], [0, 0, 1, -1]], dtype=float)
    initial_first = (np.tile(base_directions[None], (d, 1, 1))
                     + np.random.default_rng(config.seed).normal(0, 0.03, (d, 2, 4)))
    initial_bias = np.tile(np.array([-0.03, -0.06]), (d, 1))
    first = torch.nn.Parameter(tensor(initial_first))
    first_bias = torch.nn.Parameter(tensor(initial_bias))
    second = torch.nn.Parameter(tensor(np.tile([0.025, 0.015], (d, 1))))
    direct = torch.nn.Parameter(tensor(np.tile([0.25, 0.05, 0.11, -0.04], (d, 1))))
    output_bias = torch.nn.Parameter(torch.full((d,), 0.08, dtype=torch.float64))
    parameters = [first, first_bias, second, direct, output_bias]
    optimizer = torch.optim.Adam(parameters, lr=0.02)

    def predict(features: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("bdf,dhf->bdh", features, first) + first_bias
        return (torch.relu(hidden) * second).sum(-1) + (
            features * direct).sum(-1) + output_bias

    best_val = float("inf")
    best_epoch = -1
    best = None
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        loss = ((predict(train_input) - train_target) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(
                ((predict(val_input) - val_target) ** 2).mean()))
            if val_rmse < best_val:
                best_val = val_rmse
                best_epoch = epoch + 1
                best = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for parameter, selected in zip(parameters, best):
            parameter.copy_(selected)
    system = _as_network(
        d, first.detach().numpy(), first_bias.detach().numpy(),
        second.detach().numpy(), direct.detach().numpy(),
        output_bias.detach().numpy())
    direction_move = np.linalg.norm(first.detach().numpy() - initial_first, axis=-1)
    bias_move = np.abs(first_bias.detach().numpy() - initial_bias)
    training = {
        "config": asdict(config),
        "optimizer": "full-batch Adam 0.02; best selection checkpoint",
        "selected_epoch": best_epoch,
        "training": _metrics(system, train_x, train_a, train_y),
        "selection": _metrics(system, val_x, val_a, val_y),
        "test": _metrics(system, test_x, test_a, test_y),
        "rollout_test": _rollout(config, system),
        "split_sha256": {"train": train_hash, "selection": val_hash, "test": test_hash},
        "splits_disjoint": True,
        "trainable_parameters": 17 * d,
        "mean_hidden_direction_movement": float(np.mean(direction_move)),
        "min_hidden_direction_movement": float(np.min(direction_move)),
        "mean_hidden_bias_movement": float(np.mean(bias_move)),
        "min_hidden_bias_movement": float(np.min(bias_move)),
        "system_sha256": _digest(system.to_dict()),
    }
    return system, training


def run_case(config: LocalLearningConfig, output: Path) -> dict:
    system, training = train_local(config)
    bins, radii = candidate_relation(config.state_dim)
    certificate = certify_weighted(
        system, bins, radii, action_bins=128, epsilon="0.17", packing_axes=4)
    witnesses = phase_witnesses(system)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "system.json", system.to_dict())
    save_json(output / "training.json", training)
    save_json(output / "certificate.json", certificate)
    record = {
        "schema": "ncd.end-to-end-local-realization.v1",
        "training": training,
        "phase_witnesses": witnesses,
        "certificate_status": certificate["status"],
        "lower_bound": certificate["lower_bound"],
        "upper_bound": certificate["upper_bound"],
    }
    save_json(output / "record.json", record)
    return record


def verify_case(output: Path) -> dict:
    training = read_json(output / "training.json")
    config = LocalLearningConfig(**training["config"])
    model, reproduced = train_local(config)
    if model.to_dict() != read_json(output / "system.json") or training != reproduced:
        raise ValueError("End-to-end local training or model replay mismatch")
    certificate = read_json(output / "certificate.json")
    verify_weighted(model, certificate)
    expected = {
        "schema": "ncd.end-to-end-local-realization.v1",
        "training": training, "phase_witnesses": phase_witnesses(model),
        "certificate_status": certificate["status"],
        "lower_bound": certificate["lower_bound"],
        "upper_bound": certificate["upper_bound"],
    }
    if read_json(output / "record.json") != expected:
        raise ValueError("End-to-end local record replay mismatch")
    return expected


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", type=int, default=7199)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.generate:
        config = LocalLearningConfig(seed=args.seed, state_dim=args.dimension)
        if args.quick:
            config.train_samples = 256
            config.selection_samples = 64
            config.test_samples = 128
            config.epochs = 100
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

