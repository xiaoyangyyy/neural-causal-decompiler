"""Behavior-only proposals, exact one-step ReLU intervention support certificates."""
from __future__ import annotations
import argparse
from functools import lru_cache
from fractions import Fraction as Q
from pathlib import Path
from typing import Callable
import numpy as np
from .continuous_separation import ContinuousReLUSystem, ReLUMLP, _digest
from .io import read_json, save_json

LEVELS = (Q(0), Q(1), Q(1, 4), Q(1, 2), Q(3, 4))

def propose_support(oracle: Callable, state_dim: int, action_dim: int,
                    *, threshold: float = 1e-10) -> dict:
    """Read only the callable transition. Finite queries can establish presence only."""
    if state_dim < 1 or action_dim < 1 or not 0 <= threshold < 1:
        raise ValueError("Invalid dimensions or threshold")
    width = state_dim + action_dim
    center = [Q(1, 2)] * width
    witnesses = {}
    calls = 0
    for j in range(width):
        points, outputs = [], []
        for level in LEVELS:
            point = center.copy()
            point[j] = level
            result = np.asarray(oracle([float(x) for x in point[:state_dim]],
                                       [float(x) for x in point[state_dim:]]),
                                dtype=np.float64)
            if result.shape != (state_dim,) or not np.isfinite(result).all():
                raise ValueError("Invalid oracle response")
            points.append(point)
            outputs.append(result)
            calls += 1
        for i in range(state_dim):
            for a in range(len(points)):
                if (i, j) in witnesses:
                    break
                for b in range(a + 1, len(points)):
                    if abs(outputs[a][i] - outputs[b][i]) > threshold:
                        witnesses[i, j] = {"output": i, "input": j,
                            "left": [str(x) for x in points[a]],
                            "right": [str(x) for x in points[b]]}
                        break
    return {"schema": "ncd.interventional-support-proposal.v1",
            "state_dim": state_dim, "action_dim": action_dim,
            "domain": "full unit state and action cubes",
            "query_calls": calls, "threshold": threshold,
            "witnesses": [witnesses[key] for key in sorted(witnesses)]}

def _possible(net: ReLUMLP) -> list[list[bool]]:
    reach = [[i == j for j in range(net.input_dim)]
             for i in range(net.input_dim)]
    for layer in net.weights:
        reach = [[any(w != 0 and reach[k][j] for k, w in enumerate(row))
                  for j in range(net.input_dim)] for row in layer]
    return reach

@lru_cache(maxsize=32)
def _sparse_layers(net: ReLUMLP):
    return tuple(tuple((Q(bias), tuple((k, Q(w)) for k, w in enumerate(row)
                                      if w != 0))
                       for row, bias in zip(layer, biases))
                 for layer, biases in zip(net.weights, net.biases))

def _exact_value(net: ReLUMLP, point: tuple[Q, ...]) -> tuple[Q, ...]:
    current = point
    layers = _sparse_layers(net)
    for index, layer in enumerate(layers):
        current = tuple(bias + sum((w * current[k] for k, w in links), Q(0))
                        for bias, links in layer)
        if index + 1 < len(layers):
            current = tuple(max(Q(0), x) for x in current)
    return current

def _checked_witnesses(system: ContinuousReLUSystem, proposal: dict) -> list[dict]:
    d, width = system.state_dim, system.state_dim + system.action_dim
    if (proposal.get("schema") != "ncd.interventional-support-proposal.v1"
            or proposal.get("state_dim") != d
            or proposal.get("action_dim") != system.action_dim
            or proposal.get("domain") != "full unit state and action cubes"
            or type(proposal.get("query_calls")) is not int
            or proposal["query_calls"] < 0
            or type(proposal.get("threshold")) not in (float, int)
            or not 0 <= proposal["threshold"] < 1):
        raise ValueError("Invalid intervention proposal")
    seen, checked = set(), []
    cache = {}
    def output(point):
        key = tuple(point)
        if key not in cache:
            cache[key] = _exact_value(system.transition, key)
        return cache[key]
    for item in proposal["witnesses"]:
        i, j = item["output"], item["input"]
        if (type(i) is not int or type(j) is not int or not 0 <= i < d
                or not 0 <= j < width or (i, j) in seen):
            raise ValueError("Invalid edge")
        seen.add((i, j))
        left, right = [Q(x) for x in item["left"]], [Q(x) for x in item["right"]]
        if (len(left) != width or len(right) != width
                or any(not 0 <= x <= 1 for x in left + right)
                or left[j] == right[j]
                or any(left[k] != right[k] for k in range(width) if k != j)):
            raise ValueError("Invalid coordinate intervention")
        delta = output(left)[i] - output(right)[i]
        if delta == 0:
            raise ValueError("Proposed edge lacks an exact witness")
        checked.append({"output": i, "input": j,
                        "left": [str(x) for x in left],
                        "right": [str(x) for x in right],
                        "exact_difference": str(delta)})
    return sorted(checked, key=lambda x: (x["output"], x["input"]))

def certify_support(system: ContinuousReLUSystem, proposal: dict) -> dict:
    """No structural path certifies absence; a missed possible path stays unresolved."""
    witnesses = _checked_witnesses(system, proposal)
    present = {(w["output"], w["input"]) for w in witnesses}
    possible = _possible(system.transition)
    unresolved = [[i, j] for i in range(system.state_dim)
                  for j in range(system.state_dim + system.action_dim)
                  if possible[i][j] and (i, j) not in present]
    total = system.state_dim * (system.state_dim + system.action_dim)
    return {"schema": "ncd.interventional-support-certificate.v1",
            "system_sha256": _digest(system.to_dict()),
            "state_dim": system.state_dim, "action_dim": system.action_dim,
            "domain": "full unit state and action cubes",
            "witnesses": witnesses, "unresolved": unresolved,
            "absent_count": total - len(witnesses) - len(unresolved),
            "status": "certified" if not unresolved else "unresolved"}

def verify_support(system: ContinuousReLUSystem, proposal: dict,
                   certificate: dict) -> dict:
    expected = certify_support(system, proposal)
    if certificate != expected:
        raise ValueError("Interventional support certificate replay mismatch")
    return {"status": expected["status"],
            "present_count": len(expected["witnesses"]),
            "absent_count": expected["absent_count"],
            "unresolved_count": len(expected["unresolved"])}

def run_case(system: ContinuousReLUSystem, output: Path) -> dict:
    output = Path(output)
    proposal = propose_support(system.step, system.state_dim, system.action_dim)
    certificate = certify_support(system, proposal)
    save_json(output / "proposal.json", proposal)
    save_json(output / "certificate.json", certificate)
    return verify_case(system, output)

def verify_case(system: ContinuousReLUSystem, output: Path) -> dict:
    output = Path(output)
    proposal = read_json(output / "proposal.json")
    expected = propose_support(system.step, system.state_dim, system.action_dim,
                               threshold=proposal["threshold"])
    if proposal != expected:
        raise ValueError("Behavior-query proposal replay mismatch")
    return verify_support(system, proposal, read_json(output / "certificate.json"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    system = ContinuousReLUSystem.from_dict(read_json(args.model))
    result = verify_case(system, args.output) if args.verify else run_case(system, args.output)
    print(result)

if __name__ == "__main__":
    main()
