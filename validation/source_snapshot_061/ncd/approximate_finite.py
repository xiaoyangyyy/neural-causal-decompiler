"""Sound lower/upper certificates for finite epsilon-realization under L-infinity."""
from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Any, Sequence
import numpy as np

from .certified_finite import FiniteInterventionalSystem, decode_word


def _vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("Approximate outputs must be finite scalars or vectors")
    return result


def linf(left: Any, right: Any) -> float:
    a, b = _vector(left), _vector(right)
    if a.shape != b.shape:
        raise ValueError("Output dimensions differ")
    return float(np.max(np.abs(a - b)))


def pair_distance(
    system: FiniteInterventionalSystem, left: int, right: int
) -> tuple[float, tuple[str, ...]]:
    """Exact sup over A* by exhaustive reachability in the finite product system."""
    queue = deque([(left, right, tuple())])
    seen = {(left, right)}
    best_distance = -1.0
    best_word: tuple[str, ...] = ()
    while queue:
        s, t, word = queue.popleft()
        distance = linf(system.outputs[s], system.outputs[t])
        if distance > best_distance:
            best_distance, best_word = distance, word
        for action_index, action in enumerate(system.actions):
            pair = (system.transitions[s][action_index], system.transitions[t][action_index])
            if pair not in seen:
                seen.add(pair)
                queue.append((pair[0], pair[1], word + (action,)))
    return best_distance, best_word


def incompatibility_graph(
    system: FiniteInterventionalSystem, epsilon: float
) -> tuple[list[set[int]], list[dict]]:
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    adjacency = [set() for _ in system.states]
    witnesses = []
    for left, right in combinations(range(system.state_count), 2):
        distance, word = pair_distance(system, left, right)
        if distance > 2 * epsilon:
            adjacency[left].add(right)
            adjacency[right].add(left)
            witnesses.append({
                "left": system.states[left],
                "right": system.states[right],
                "word": list(word),
                "distance": distance,
                "left_response": system.response(left, word),
                "right_response": system.response(right, word),
            })
    return adjacency, witnesses


def _k_coloring(adjacency: Sequence[set[int]], colors: int) -> list[int] | None:
    n = len(adjacency)
    assignment = [-1] * n

    def search(colored: int) -> bool:
        if colored == n:
            return True
        unassigned = [v for v in range(n) if assignment[v] < 0]
        vertex = max(
            unassigned,
            key=lambda v: (
                len({assignment[u] for u in adjacency[v] if assignment[u] >= 0}),
                len(adjacency[v]),
                -v,
            ),
        )
        forbidden = {assignment[u] for u in adjacency[vertex] if assignment[u] >= 0}
        for color in range(colors):
            if color in forbidden:
                continue
            assignment[vertex] = color
            if search(colored + 1):
                return True
            assignment[vertex] = -1
        return False

    return assignment if search(0) else None


def chromatic_number(adjacency: Sequence[set[int]]) -> tuple[int, list[int]]:
    for colors in range(1, len(adjacency) + 1):
        coloring = _k_coloring(adjacency, colors)
        if coloring is not None:
            return colors, coloring
    raise RuntimeError("Finite graph has no coloring")


def _valid_upper_partition(
    system: FiniteInterventionalSystem, assignment: Sequence[int], epsilon: float
) -> tuple[FiniteInterventionalSystem, list[int]] | None:
    colors = sorted(set(assignment))
    remap = {color: index for index, color in enumerate(colors)}
    encoding = [remap[color] for color in assignment]
    blocks = [[state for state, color in enumerate(encoding) if color == block] for block in range(len(colors))]
    outputs = []
    transitions = []
    for block in blocks:
        values = np.stack([_vector(system.outputs[state]) for state in block])
        center = (values.min(axis=0) + values.max(axis=0)) / 2
        if any(linf(system.outputs[state], center) > epsilon + 1e-12 for state in block):
            return None
        rows = {
            tuple(encoding[target] for target in system.transitions[state])
            for state in block
        }
        if len(rows) != 1:
            return None
        outputs.append(float(center[0]) if center.size == 1 else center.tolist())
        transitions.append(next(iter(rows)))
    candidate = FiniteInterventionalSystem(
        tuple(f"q{i}" for i in range(len(blocks))),
        system.actions,
        tuple(transitions),
        tuple(outputs),
    )
    return candidate, encoding


def _find_upper(
    system: FiniteInterventionalSystem,
    adjacency: Sequence[set[int]],
    epsilon: float,
    colors: int,
) -> tuple[FiniteInterventionalSystem, list[int]] | None:
    n = system.state_count
    assignment = [-1] * n
    block_members: list[list[int]] = [[] for _ in range(colors)]

    def search(state: int, used: int):
        if state == n:
            return _valid_upper_partition(system, assignment, epsilon)
        maximum = min(used + 1, colors)
        for color in range(maximum):
            if color == used and used == colors:
                continue
            if any(other in adjacency[state] for other in block_members[color]):
                continue
            proposed = block_members[color] + [state]
            values = np.stack([_vector(system.outputs[item]) for item in proposed])
            if float(np.max(values.max(axis=0) - values.min(axis=0))) > 2 * epsilon + 1e-12:
                continue
            assignment[state] = color
            block_members[color].append(state)
            result = search(state + 1, max(used, color + 1))
            if result is not None:
                return result
            block_members[color].pop()
            assignment[state] = -1
        return None

    return search(0, 0)


def approximate_minimize(system: FiniteInterventionalSystem, epsilon: float) -> dict:
    adjacency, witnesses = incompatibility_graph(system, epsilon)
    lower, coloring = chromatic_number(adjacency)
    upper_result = None
    for colors in range(lower, system.state_count + 1):
        upper_result = _find_upper(system, adjacency, epsilon, colors)
        if upper_result is not None:
            break
    if upper_result is None:
        raise RuntimeError("Singleton partition should always be a valid upper bound")
    candidate, encoding = upper_result
    return {
        "schema": "ncd.approximate-realization-certificate.v1",
        "source_identity": system.identity(),
        "metric": "linf",
        "epsilon": epsilon,
        "scope": "all finite action words A* over the declared finite system",
        "lower": {
            "bound": lower,
            "chromatic_coloring": coloring,
            "edges": witnesses,
        },
        "upper": {
            "bound": candidate.state_count,
            "candidate": candidate.to_dict(),
            "encoding": {state: encoding[i] for i, state in enumerate(system.states)},
            "proof": "output epsilon-cover plus exact labelled transition homomorphism",
        },
        "minimal": lower == candidate.state_count,
    }


def verify_approximate_certificate(
    system: FiniteInterventionalSystem, certificate: dict
) -> dict:
    if certificate.get("schema") != "ncd.approximate-realization-certificate.v1":
        raise ValueError("Unsupported approximate certificate schema")
    if certificate.get("source_identity") != system.identity():
        raise ValueError("Approximate certificate source mismatch")
    if certificate.get("metric") != "linf":
        raise ValueError("Only the L-infinity metric is supported")
    epsilon = float(certificate["epsilon"])
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("Invalid epsilon")

    lower_value = certificate["lower"]
    adjacency = [set() for _ in system.states]
    seen = set()
    for edge in lower_value["edges"]:
        left = system.state_index(edge["left"])
        right = system.state_index(edge["right"])
        pair = (min(left, right), max(left, right))
        if left == right or pair in seen:
            raise ValueError("Duplicate or reflexive incompatibility edge")
        word = decode_word(edge["word"])
        left_response = system.response(left, word)
        right_response = system.response(right, word)
        distance = linf(left_response, right_response)
        if distance <= 2 * epsilon:
            raise ValueError("Incompatibility witness does not exceed 2 epsilon")
        if abs(distance - float(edge["distance"])) > 1e-12:
            raise ValueError("Stored incompatibility distance is incorrect")
        if linf(left_response, edge["left_response"]) > 1e-12 or linf(
            right_response, edge["right_response"]
        ) > 1e-12:
            raise ValueError("Stored incompatibility response is incorrect")
        adjacency[left].add(right)
        adjacency[right].add(left)
        seen.add(pair)
    chromatic, coloring = chromatic_number(adjacency)
    if chromatic != lower_value["bound"]:
        raise ValueError("Claimed chromatic lower bound is incorrect")
    stored_coloring = lower_value["chromatic_coloring"]
    if len(stored_coloring) != system.state_count:
        raise ValueError("Stored coloring has the wrong size")
    if any(stored_coloring[a] == stored_coloring[b] for a, b in seen):
        raise ValueError("Stored coloring violates an incompatibility edge")
    if len(set(stored_coloring)) != chromatic:
        raise ValueError("Stored coloring does not use the claimed color count")

    upper_value = certificate["upper"]
    candidate = FiniteInterventionalSystem.from_dict(upper_value["candidate"])
    if candidate.actions != system.actions:
        raise ValueError("Approximate candidate changes action labels")
    raw_encoding = upper_value["encoding"]
    if set(raw_encoding) != set(system.states):
        raise ValueError("Approximate encoding does not cover all states")
    encoding = [raw_encoding[state] for state in system.states]
    if set(encoding) != set(range(candidate.state_count)):
        raise ValueError("Approximate candidate has invalid or unused states")
    maximum_output_error = 0.0
    for state in range(system.state_count):
        abstract = encoding[state]
        error = linf(system.outputs[state], candidate.outputs[abstract])
        maximum_output_error = max(maximum_output_error, error)
        if error > epsilon + 1e-12:
            raise ValueError("Approximate upper certificate exceeds epsilon")
        for action in range(len(system.actions)):
            if encoding[system.transitions[state][action]] != candidate.transitions[abstract][action]:
                raise ValueError("Approximate upper certificate fails transition homomorphism")
    if upper_value["bound"] != candidate.state_count:
        raise ValueError("Incorrect approximate upper bound")
    if lower_value["bound"] > upper_value["bound"]:
        raise ValueError("Approximate lower bound exceeds upper bound")
    minimal = lower_value["bound"] == upper_value["bound"]
    if bool(certificate.get("minimal")) != minimal:
        raise ValueError("Incorrect approximate minimality flag")
    return {
        "status": "verified",
        "lower_bound": lower_value["bound"],
        "upper_bound": upper_value["bound"],
        "minimal": minimal,
        "edges_verified": len(seen),
        "maximum_output_error": maximum_output_error,
        "homomorphism_checks": system.state_count * len(system.actions),
        "independent_coloring": coloring,
    }
