"""Certified finite realizations of learned quantized traffic dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import html
import json
import math
import time
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans

from .certified_finite import (
    FiniteInterventionalSystem,
    ResponseOracle,
    certificate_from_recovery,
    compare_partition,
    oracle_minimize,
    recover_from_responses,
    verify_certificate,
)
from .io import digest, read_json, save_json

ACTIONS = ("normal", "demand_up", "incident_link1", "signal_red_link2", "closure_link3")


@dataclass
class TrafficConfig:
    seed: int = 3701
    cases: int = 2
    links: int = 4
    codebook_states: int = 10
    trajectories: int = 256
    horizon: int = 20
    hidden: int = 48
    epochs: int = 600
    learning_rate: float = 0.03
    budget_fraction: float = 0.5

    @classmethod
    def quick(cls, seed: int = 3701) -> "TrafficConfig":
        return cls(seed=seed, cases=1, codebook_states=7, trajectories=64, horizon=12, hidden=24, epochs=300)

    def validate(self) -> None:
        if self.cases < 1 or self.links != 4 or self.codebook_states < 3:
            raise ValueError("Invalid traffic benchmark dimensions")
        if self.trajectories < self.codebook_states or self.horizon < 2 or self.epochs < 1:
            raise ValueError("Insufficient traffic training configuration")
        if not 0 < self.learning_rate < 1 or not 0 < self.budget_fraction < 1:
            raise ValueError("Invalid traffic optimization configuration")


def traffic_step(state: np.ndarray, action: int) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    if state.shape != (4,) or not 0 <= action < len(ACTIONS):
        raise ValueError("Invalid traffic state or action")
    capacity = np.full(4, 0.22)
    demand = 0.055
    if action == 1:
        demand = 0.13
    elif action == 2:
        capacity[1] = 0.065
    elif action == 3:
        capacity[2] = 0.045
    elif action == 4:
        capacity[3] = 0.008
    downstream_room = np.maximum(0.0, 1.0 - np.roll(state, -1))
    outflow = np.minimum(np.minimum(state, capacity), 0.32 * downstream_room)
    inflow = np.roll(outflow, 1)
    inflow[0] += demand
    next_state = state + inflow - outflow
    next_state *= 0.995
    return np.clip(next_state, 0.0, 1.0)


def _traffic_samples(config: TrafficConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(config.trajectories):
        state = rng.uniform(0.03, 0.82, size=4)
        for _ in range(config.horizon):
            values.append(state.copy())
            state = traffic_step(state, int(rng.integers(0, len(ACTIONS))))
    return np.asarray(values)


class _TransitionNet(nn.Module):
    def __init__(self, inputs: int, hidden: int, outputs: int):
        super().__init__()
        self.first = nn.Linear(inputs, hidden)
        self.second = nn.Linear(hidden, outputs)

    def forward(self, value):
        return self.second(torch.relu(self.first(value)))


@dataclass(frozen=True)
class TrafficNeuralTransducer:
    codebook: tuple[tuple[float, ...], ...]
    first_weight: tuple[tuple[float, ...], ...]
    first_bias: tuple[float, ...]
    second_weight: tuple[tuple[float, ...], ...]
    second_bias: tuple[float, ...]

    def step(self, state: int, action: int) -> int:
        count = len(self.codebook)
        value = np.zeros(count + len(ACTIONS), dtype=float)
        value[state] = 1.0
        value[count + action] = 1.0
        hidden = np.maximum(np.asarray(self.first_weight) @ value + np.asarray(self.first_bias), 0.0)
        logits = np.asarray(self.second_weight) @ hidden + np.asarray(self.second_bias)
        return int(np.argmax(logits))

    def observe(self, state: int) -> int:
        mean_density = float(np.mean(self.codebook[state]))
        return 0 if mean_density < 0.36 else 1 if mean_density < 0.64 else 2

    def enumerate_system(self) -> FiniteInterventionalSystem:
        count = len(self.codebook)
        return FiniteInterventionalSystem(
            tuple(f"traffic_{i}" for i in range(count)),
            ACTIONS,
            tuple(tuple(self.step(state, action) for action in range(len(ACTIONS))) for state in range(count)),
            tuple(self.observe(state) for state in range(count)),
            self.codebook,
        )

    def to_dict(self) -> dict:
        return {
            "schema": "ncd.traffic-neural-transducer.v1",
            "codebook": [list(row) for row in self.codebook],
            "first_weight": [list(row) for row in self.first_weight],
            "first_bias": list(self.first_bias),
            "second_weight": [list(row) for row in self.second_weight],
            "second_bias": list(self.second_bias),
            "actions": list(ACTIONS),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TrafficNeuralTransducer":
        if value.get("schema") != "ncd.traffic-neural-transducer.v1" or tuple(value["actions"]) != ACTIONS:
            raise ValueError("Unsupported traffic neural transducer")
        return cls(
            tuple(tuple(float(x) for x in row) for row in value["codebook"]),
            tuple(tuple(float(x) for x in row) for row in value["first_weight"]),
            tuple(float(x) for x in value["first_bias"]),
            tuple(tuple(float(x) for x in row) for row in value["second_weight"]),
            tuple(float(x) for x in value["second_bias"]),
        )


def train_traffic_transducer(config: TrafficConfig, seed: int) -> tuple[TrafficNeuralTransducer, dict]:
    # CPU reductions may otherwise choose different parallel reduction orders
    # across fresh processes, changing recorded losses in their final bits.
    torch.set_num_threads(1)
    samples = _traffic_samples(config, seed)
    kmeans = KMeans(
        n_clusters=config.codebook_states,
        n_init=10,
        random_state=seed,
    ).fit(samples)
    codebook = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    count = len(codebook)
    teacher = np.zeros((count, len(ACTIONS)), dtype=np.int64)
    for state in range(count):
        for action in range(len(ACTIONS)):
            target = traffic_step(codebook[state], action)
            teacher[state, action] = int(np.argmin(np.sum((codebook - target) ** 2, axis=1)))
    features = np.zeros((count * len(ACTIONS), count + len(ACTIONS)), dtype=np.float32)
    labels = np.zeros(count * len(ACTIONS), dtype=np.int64)
    row = 0
    for state in range(count):
        for action in range(len(ACTIONS)):
            features[row, state] = 1.0
            features[row, count + action] = 1.0
            labels[row] = teacher[state, action]
            row += 1

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = _TransitionNet(features.shape[1], config.hidden, count)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    with torch.no_grad():
        initial_loss = float(nn.functional.cross_entropy(model(x), y))
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x)
        prediction = logits.argmax(dim=1)
        accuracy = float((prediction == y).float().mean())
        final_loss = float(nn.functional.cross_entropy(logits, y))
    if accuracy < 1.0:
        raise RuntimeError(f"Traffic transition network did not fit its finite training set: {accuracy}")
    neural = TrafficNeuralTransducer(
        tuple(tuple(float(value) for value in row) for row in codebook),
        tuple(tuple(float(value) for value in row) for row in model.first.weight.detach().numpy()),
        tuple(float(value) for value in model.first.bias.detach().numpy()),
        tuple(tuple(float(value) for value in row) for row in model.second.weight.detach().numpy()),
        tuple(float(value) for value in model.second.bias.detach().numpy()),
    )
    # sklearn's parallel inertia reduction can differ by a few ulps between
    # processes. Recompute the descriptive statistic in a fixed scalar order.
    stable_inertia = math.fsum(
        (float(samples[row, column]) - float(codebook[int(kmeans.labels_[row]), column])) ** 2
        for row in range(len(samples)) for column in range(samples.shape[1])
    )
    return neural, {
        "samples": int(len(samples)),
        "training_pairs": int(len(labels)),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "teacher_accuracy": accuracy,
        "inertia": stable_inertia,
        "teacher_transitions": teacher.tolist(),
    }


def run_traffic_case(config: TrafficConfig, index: int) -> dict:
    seed = config.seed + index * 1009
    neural, training = train_traffic_transducer(config, seed)
    system = neural.enumerate_system()
    oracle = oracle_minimize(system)
    oracle_check = verify_certificate(system, oracle["certificate"])
    active = recover_from_responses(ResponseOracle(system), strategy="active", seed=seed)
    quality = compare_partition(oracle["blocks"], active["blocks"])
    if not active["certified_complete"] or not quality["exact"]:
        raise RuntimeError("Traffic response recovery did not certify the oracle quotient")
    active_certificate = certificate_from_recovery(system.identity(), active)
    active_check = verify_certificate(system, active_certificate)
    budget = max(system.state_count, int(active["queries"] * config.budget_fraction))
    limited = recover_from_responses(
        ResponseOracle(system), strategy="active", query_budget=budget, seed=seed
    )
    return {
        "case": index,
        "seed": seed,
        "neural": neural.to_dict(),
        "training": training,
        "system": system.to_dict(),
        "oracle": oracle,
        "oracle_verification": oracle_check,
        "active": active,
        "active_certificate": active_certificate,
        "active_verification": active_check,
        "quality": quality,
        "limited": {
            "budget": budget,
            "result": limited,
            "quality": compare_partition(oracle["blocks"], limited["blocks"]),
        },
    }


def summarize_traffic(cases: list[dict]) -> dict:
    return {
        "schema": "ncd.certified-traffic-summary.v1",
        "cases": len(cases),
        "actions": list(ACTIONS),
        "all_training_pairs_fit": all(case["training"]["teacher_accuracy"] == 1.0 for case in cases),
        "all_certificates_verified": all(case["active_verification"]["minimal"] for case in cases),
        "active_exact_cases": sum(int(case["quality"]["exact"]) for case in cases),
        "concrete_states": sum(len(case["system"]["states"]) for case in cases),
        "minimal_states": sum(len(case["oracle"]["blocks"]) for case in cases),
        "response_queries": sum(case["active"]["queries"] for case in cases),
        "limited_unresolved_pairs": sum(
            case["limited"]["result"]["pair_status"]["unresolved"] for case in cases
        ),
        "scope": "learned quantized traffic dynamics; finite codebook and five declared control actions",
    }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _report(summary: dict) -> str:
    actions = ", ".join(html.escape(action) for action in summary["actions"])
    return f"""<!doctype html><meta charset="utf-8"><title>Certified traffic realization</title>
<style>body{{font:16px system-ui;max-width:900px;margin:40px auto;line-height:1.5}}</style>
<h1>Certified traffic realization</h1>
<p><strong>Scope:</strong> {html.escape(summary['scope'])}</p>
<p>Actions: {actions}</p>
<p>{summary['cases']} trained models; {summary['concrete_states']} quantized
states; {summary['minimal_states']} certified minimal states;
{summary['response_queries']} response queries.</p>
<p>Budget-limited runs retained {summary['limited_unresolved_pairs']}
unresolved pairs.</p>"""


def run_certified_traffic(output: Path, config: TrafficConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    started = time.perf_counter()
    cases = []
    for index in range(config.cases):
        case = run_traffic_case(config, index)
        cases.append(case)
        directory = output / "cases" / f"case_{index:03d}"
        save_json(directory / "neural.json", case["neural"])
        save_json(directory / "training.json", case["training"])
        save_json(directory / "system.json", case["system"])
        save_json(directory / "oracle_certificate.json", case["oracle"]["certificate"])
        save_json(directory / "active_recovery.json", case["active"])
        save_json(directory / "active_certificate.json", case["active_certificate"])
        save_json(directory / "limited.json", case["limited"])
        save_json(directory / "case.json", case)
    summary = summarize_traffic(cases)
    summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    (output / "report.html").write_text(_report(summary), encoding="utf-8", newline="\n")
    artifacts = {
        path.relative_to(output).as_posix(): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    save_json(output / "manifest.json", {
        "schema": "ncd.certified-traffic-manifest.v1",
        "artifacts": artifacts,
    })
    return summary


def verify_certified_traffic(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.certified-traffic-manifest.v1":
        raise ValueError("Unsupported traffic manifest")
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("Traffic artifact set mismatch")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Traffic artifact integrity failure: {relative}")
    config = TrafficConfig(**read_json(output / "config.json"))
    config.validate()
    cases = []
    for index in range(config.cases):
        directory = output / "cases" / f"case_{index:03d}"
        stored = read_json(directory / "case.json")
        neural = TrafficNeuralTransducer.from_dict(read_json(directory / "neural.json"))
        system = FiniteInterventionalSystem.from_dict(read_json(directory / "system.json"))
        if neural.enumerate_system().to_dict() != system.to_dict():
            raise ValueError(f"Traffic neural enumeration mismatch in case {index}")
        verify_certificate(system, read_json(directory / "oracle_certificate.json"))
        verify_certificate(system, read_json(directory / "active_certificate.json"))
        replayed = run_traffic_case(config, index)
        if _canonical(replayed) != _canonical(stored):
            raise ValueError(f"Traffic deterministic replay mismatch in case {index}")
        cases.append(replayed)
    replayed_summary = summarize_traffic(cases)
    stored_summary = read_json(output / "summary.json")
    stored_summary.pop("runtime_seconds", None)
    if _canonical(replayed_summary) != _canonical(stored_summary):
        raise ValueError("Traffic summary mismatch")
    return {
        "status": "verified",
        "cases_replayed": len(cases),
        "certificates_verified": 2 * len(cases),
        "active_exact_cases": replayed_summary["active_exact_cases"],
        "limited_unresolved_pairs": replayed_summary["limited_unresolved_pairs"],
    }
