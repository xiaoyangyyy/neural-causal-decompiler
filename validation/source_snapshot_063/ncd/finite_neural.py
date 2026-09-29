"""Finite quantized-state transducers executed by explicit ReLU networks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import numpy as np

from .certified_finite import FiniteInterventionalSystem


@dataclass(frozen=True)
class FiniteNeuralTransducer:
    states: tuple[str, ...]
    actions: tuple[str, ...]
    output_vocab: tuple[Any, ...]
    transition_hidden_weight: tuple[tuple[float, ...], ...]
    transition_hidden_bias: tuple[float, ...]
    transition_output_weight: tuple[tuple[float, ...], ...]
    observation_weight: tuple[tuple[float, ...], ...]

    @classmethod
    def compile(cls, system: FiniteInterventionalSystem) -> "FiniteNeuralTransducer":
        """Compile a table into a ReLU network; evaluation later queries the network."""
        n, action_count = system.state_count, len(system.actions)
        hidden = n * action_count
        first = np.zeros((hidden, n + action_count), dtype=float)
        second = np.zeros((n, hidden), dtype=float)
        for state in range(n):
            for action in range(action_count):
                unit = state * action_count + action
                first[unit, state] = 1.0
                first[unit, n + action] = 1.0
                second[system.transitions[state][action], unit] = 2.0
        output_vocab = []
        keys = []
        for output in system.outputs:
            key = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if key not in keys:
                keys.append(key)
                output_vocab.append(output)
        observation = np.zeros((len(output_vocab), n), dtype=float)
        for state, output in enumerate(system.outputs):
            key = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            observation[keys.index(key), state] = 2.0
        return cls(
            system.states,
            system.actions,
            tuple(output_vocab),
            tuple(tuple(float(x) for x in row) for row in first),
            tuple(-1.0 for _ in range(hidden)),
            tuple(tuple(float(x) for x in row) for row in second),
            tuple(tuple(float(x) for x in row) for row in observation),
        )

    def step(self, state: int, action: int) -> int:
        n = len(self.states)
        if not 0 <= state < n or not 0 <= action < len(self.actions):
            raise ValueError("Invalid neural transducer input")
        value = np.zeros(n + len(self.actions), dtype=float)
        value[state] = 1.0
        value[n + action] = 1.0
        hidden = np.maximum(
            np.asarray(self.transition_hidden_weight) @ value
            + np.asarray(self.transition_hidden_bias),
            0.0,
        )
        logits = np.asarray(self.transition_output_weight) @ hidden
        return int(np.argmax(logits))

    def observe(self, state: int) -> Any:
        if not 0 <= state < len(self.states):
            raise ValueError("Invalid neural transducer state")
        code = np.zeros(len(self.states), dtype=float)
        code[state] = 1.0
        logits = np.asarray(self.observation_weight) @ code
        return self.output_vocab[int(np.argmax(logits))]

    def enumerate_system(self) -> FiniteInterventionalSystem:
        n = len(self.states)
        transitions = tuple(
            tuple(self.step(state, action) for action in range(len(self.actions)))
            for state in range(n)
        )
        outputs = tuple(self.observe(state) for state in range(n))
        # Quantized hidden state is the one-hot neural state code.
        embeddings = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(n))
            for row in range(n)
        )
        return FiniteInterventionalSystem(self.states, self.actions, transitions, outputs, embeddings)

    def to_dict(self) -> dict:
        return {
            "schema": "ncd.finite-relu-transducer.v1",
            "states": list(self.states),
            "actions": list(self.actions),
            "output_vocab": list(self.output_vocab),
            "transition_hidden_weight": [list(row) for row in self.transition_hidden_weight],
            "transition_hidden_bias": list(self.transition_hidden_bias),
            "transition_output_weight": [list(row) for row in self.transition_output_weight],
            "observation_weight": [list(row) for row in self.observation_weight],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FiniteNeuralTransducer":
        if value.get("schema") != "ncd.finite-relu-transducer.v1":
            raise ValueError("Unsupported finite neural transducer schema")
        return cls(
            tuple(value["states"]),
            tuple(value["actions"]),
            tuple(value["output_vocab"]),
            tuple(tuple(float(x) for x in row) for row in value["transition_hidden_weight"]),
            tuple(float(x) for x in value["transition_hidden_bias"]),
            tuple(tuple(float(x) for x in row) for row in value["transition_output_weight"]),
            tuple(tuple(float(x) for x in row) for row in value["observation_weight"]),
        )
