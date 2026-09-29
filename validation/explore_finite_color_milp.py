"""Diagnostic MILP for finite-trace eight-state congruence; not a formal proof."""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ACTIONS = (Fraction(0), Fraction(1, 2), Fraction(1))
BETA = Fraction(0.4)
EPSILON = Fraction(0.101)


def step(x: Fraction, action: Fraction) -> Fraction:
    return Fraction(1, 2) * x + BETA * action


def instance(depth: int):
    roots = {Fraction(i, 20) for i in range(21)}
    levels = [roots]
    for _ in range(depth):
        levels.append({step(x, a) for x in levels[-1] for a in ACTIONS})
    nodes = sorted(set().union(*levels))
    parents = sorted(set().union(*levels[:-1]))
    lookup = {x: i for i, x in enumerate(nodes)}
    edges = [(lookup[x], a_index, lookup[step(x, a)])
             for x in parents for a_index, a in enumerate(ACTIONS)]
    incompatible = [(i, j) for i, j in combinations(range(len(nodes)), 2)
                    if abs(nodes[i] - nodes[j]) > 2 * EPSILON]
    anchors = [(lookup[Fraction(i, 4)], i) for i in range(5)]
    return nodes, edges, incompatible, anchors


def solve(depth: int, states: int, limit: float):
    nodes, edges, incompatible, anchors = instance(depth)
    n = len(nodes)
    action_count = len(ACTIONS)
    variables = n * states + states * action_count * states
    rows, cols, data = [], [], []
    lower, upper = [], []

    def add(terms, lo=-np.inf, hi=np.inf):
        row = len(lower)
        lower.append(lo)
        upper.append(hi)
        for index, coefficient in terms:
            rows.append(row)
            cols.append(index)
            data.append(coefficient)

    def z(node, color):
        return node * states + color

    def t(color, action, target):
        return n * states + (color * action_count + action) * states + target

    for node in range(n):
        add([(z(node, color), 1) for color in range(states)], 1, 1)
    for color in range(states):
        for action in range(action_count):
            add([(t(color, action, target), 1)
                 for target in range(states)], 1, 1)
    for left, right in incompatible:
        for color in range(states):
            add([(z(left, color), 1), (z(right, color), 1)], hi=1)
    for source, action, child in edges:
        for color in range(states):
            for target in range(states):
                add([(z(source, color), 1), (t(color, action, target), 1),
                     (z(child, target), -1)], hi=1)
    for node, color in anchors:
        add([(z(node, color), 1)], 1, 1)

    matrix = coo_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=(len(lower), variables)).tocsr()
    result = milp(
        c=np.zeros(variables), integrality=np.ones(variables),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": limit, "mip_rel_gap": 0.0},
    )
    payload = {
        "scope": "finite sampled traces only; infeasibility is not a verified global lower certificate",
        "depth": depth, "candidate_states": states,
        "concrete_nodes": n, "edges": len(edges),
        "incompatible_pairs": len(incompatible),
        "binary_variables": variables, "constraints": len(lower),
        "solver_status": int(result.status), "message": str(result.message),
    }
    if result.x is not None:
        assignment = [int(np.argmax(result.x[node * states:(node + 1) * states]))
                      for node in range(n)]
        transitions = [
            [int(np.argmax(result.x[
                n * states + (color * action_count + action) * states:
                n * states + (color * action_count + action + 1) * states]))
             for action in range(action_count)]
            for color in range(states)
        ]
        if any(assignment[left] == assignment[right]
               for left, right in incompatible):
            raise RuntimeError("MILP returned an incompatible color assignment")
        if any(transitions[assignment[source]][action] != assignment[child]
               for source, action, child in edges):
            raise RuntimeError("MILP returned a nondeterministic transition")
        payload["assignment_verified_on_sample"] = True
        payload["color_hulls"] = [
            [str(min(nodes[i] for i, c in enumerate(assignment) if c == color)),
             str(max(nodes[i] for i, c in enumerate(assignment) if c == color))]
            if color in assignment else None for color in range(states)]
        payload["transitions"] = transitions
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    print(json.dumps(solve(args.depth, args.states, args.seconds), indent=2))


if __name__ == "__main__":
    main()

