"""Exact all-horizon behavioral quotient for globally affine frozen ReLU systems.

The quotient is ker span{C, CA, ..., CA^(d-1)}. Actions cancel between two
runs given the same word. For a cyclic local network, a triangular delayed
observation proof establishes full rank without huge matrix powers.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from pathlib import Path

from .continuous_separation import ContinuousReLUSystem, ReLUMLP, _digest
from .io import read_json, save_json


class Unresolved(Exception):
    """The exact affine quotient checker cannot establish this case."""


def _format(x):
    return str(x)


def _combine(row, bias, forms):
    width = len(forms[0]) - 1
    result = [Q(0)] * (width + 1)
    result[-1] = Q(bias)
    for weight, form in zip(row, forms):
        if weight == 0:
            continue
        w = Q(weight)
        for j, value in enumerate(form):
            if value:
                result[j] += w * value
    return tuple(result)


def _affine_network(net: ReLUMLP):
    """Extract a global affine map only after proving every ReLU phase fixed."""
    width = net.input_dim
    forms = tuple(tuple(Q(int(i == j)) for j in range(width)) + (Q(0),)
                  for i in range(width))
    active, inactive = 0, 0
    for layer, (weights, biases) in enumerate(zip(net.weights, net.biases)):
        next_forms = tuple(_combine(row, bias, forms)
                           for row, bias in zip(weights, biases))
        if layer + 1 < len(net.weights):
            checked = []
            for form in next_forms:
                low = form[-1] + sum((min(Q(0), x) for x in form[:-1]), Q(0))
                high = form[-1] + sum((max(Q(0), x) for x in form[:-1]), Q(0))
                if low >= 0:
                    checked.append(form)
                    active += 1
                elif high <= 0:
                    checked.append((Q(0),) * (width + 1))
                    inactive += 1
                else:
                    raise Unresolved("Hidden ReLU phase changes on the full cube")
            forms = tuple(checked)
        else:
            forms = next_forms
    return tuple(tuple(form[:-1]) for form in forms), tuple(form[-1] for form in forms), {
        "active_hidden": active, "inactive_hidden": inactive}


def _matmul(left, right):
    if not left:
        return tuple()
    columns = len(right[0])
    return tuple(tuple(sum((a * row[j] for a, row in zip(left_row, right)), Q(0))
                       for j in range(columns))
                 for left_row in left)


def _rref(rows, width):
    """Canonical exact row-space basis and its pivot columns."""
    matrix = [list(row) for row in rows if any(row)]
    pivots = []
    rank = 0
    for col in range(width):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][col]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][col]
        matrix[rank] = [x / divisor for x in matrix[rank]]
        for i in range(len(matrix)):
            if i == rank or not matrix[i][col]:
                continue
            factor = matrix[i][col]
            matrix[i] = [x - factor * y
                         for x, y in zip(matrix[i], matrix[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(matrix):
            break
    return tuple(tuple(row) for row in matrix[:rank]), tuple(pivots)


def _ring_full(A, C):
    d = len(A)
    if not any(row[0] != 0 and all(row[j] == 0 for j in range(1, d))
               for row in C):
        return None
    edges = []
    for i, row in enumerate(A):
        predecessor = (i - 1) % d
        if row[predecessor] == 0 or any(
                value != 0 and j not in (i, predecessor)
                for j, value in enumerate(row)):
            return None
        edges.append(row[predecessor])
    return edges


def _support_closure(A, C):
    closure = {j for row in C for j, value in enumerate(row) if value}
    while True:
        larger = closure | {j for i in closure for j, value in enumerate(A[i]) if value}
        if larger == closure:
            return tuple(sorted(closure))
        closure = larger


def _quotient_data(T, A, B, b, C, c):
    """Exact induced affine dynamics on a canonical row-space basis."""
    basis, pivots = _rref(T, len(A))
    if tuple(T) != basis:
        raise AssertionError("Quotient rows must be canonical")
    r = len(T)
    TA = _matmul(T, A)
    Abar = tuple(tuple(row[pivot] for pivot in pivots) for row in TA)
    Cbar = tuple(tuple(row[pivot] for pivot in pivots) for row in C)
    if _matmul(Abar, T) != TA or _matmul(Cbar, T) != C:
        raise AssertionError("Quotient transition or output does not factor")
    Bbar = _matmul(T, B)
    bbar = tuple(sum((a * x for a, x in zip(row, b)), Q(0)) for row in T)
    return {
        "basis": [[_format(x) for x in row] for row in T],
        "pivots": list(pivots),
        "transition": [[_format(x) for x in row] for row in Abar],
        "control": [[_format(x) for x in row] for row in Bbar],
        "offset": [_format(x) for x in bbar],
        "observation": [[_format(x) for x in row] for row in Cbar],
        "observation_offset": [_format(x) for x in c],
        "state_domain": "linear image of full unit cube",
    }


def _general_basis(A, C, max_general_dim):
    d = len(A)
    if d > max_general_dim:
        raise Unresolved("General exact row-space cap")
    basis, _ = _rref(C, d)
    while True:
        grown, _ = _rref(basis + _matmul(basis, A), d)
        if grown == basis:
            return basis
        basis = grown


def _compute(system: ContinuousReLUSystem, max_general_dim=12):
    d, u = system.state_dim, system.action_dim
    base = {"schema": "ncd.affine-behavioral-quotient.v1",
            "system_sha256": _digest(system.to_dict()),
            "state_dim": d, "action_dim": u,
            "input_domain": "full unit state and action cubes",
            "continuations": "all finite words over continuous unit action cube",
            "max_general_dim": max_general_dim}
    try:
        transition, b, t_phase = _affine_network(system.transition)
        C, c, o_phase = _affine_network(system.observation)
        A = tuple(row[:d] for row in transition)
        B = tuple(row[d:] for row in transition)
        for row, offset in zip(transition, b):
            low = offset + sum((min(Q(0), x) for x in row), Q(0))
            high = offset + sum((max(Q(0), x) for x in row), Q(0))
            if low < 0 or high > 1:
                raise Unresolved("Transition does not preserve the full unit cube")
        phases = {"transition": t_phase, "observation": o_phase}
        if not any(value for row in C for value in row):
            return {**base, "status": "certified",
                    "proof": "constant-observation",
                    "quotient_dim": 0, "observation_dim": len(C),
                    "phase_summary": phases,
                    "constant_output": [_format(x) for x in c]}
        edges = _ring_full(A, C)
        if edges is not None:
            observed = next(i for i, row in enumerate(C)
                            if row[0] != 0 and all(x == 0 for x in row[1:]))
            source = min(4, d - 1)
            horizon = d - source
            coefficient = C[observed][0]
            for step in range(horizon):
                coefficient *= edges[(-step) % d]
            active = {source}
            first_possible = None
            for time in range(d):
                if any(C[row][j] != 0 for row in range(len(C)) for j in active):
                    first_possible = time
                    break
                active = {i for i, row in enumerate(A)
                          if any(row[j] != 0 for j in active)}
            first_distinction = horizon if first_possible == horizon else None
            return {**base, "status": "certified",
                    "proof": "cyclic-triangular-delayed-observation",
                    "quotient_dim": d, "observation_dim": len(C),
                    "phase_summary": phases,
                    "ring_edges": [_format(x) for x in edges],
                    "coordinate_map": "identity",
                    "delayed_witness": {
                        "source_coordinate": source,
                        "horizon": horizon,
                        "observed_output": observed,
                        "action_word": "all-zero",
                        "exact_response_difference": _format(coefficient),
                        "initially_unobserved": all(row[source] == 0 for row in C),
                        "first_distinction_horizon": first_distinction,
                    }}
        closure = _support_closure(A, C)
        restricted = tuple(tuple(row[j] for j in closure) for row in C)
        if len(closure) < d and len(_rref(restricted, len(closure))[0]) == len(closure):
            T = tuple(tuple(Q(int(j == index)) for j in range(d))
                      for index in closure)
            return {**base, "status": "certified",
                    "proof": "closed-coordinate-projection",
                    "quotient_dim": len(closure), "observation_dim": len(C),
                    "phase_summary": phases,
                    "coordinates": list(closure),
                    "quotient": _quotient_data(T, A, B, b, C, c)}
        T = _general_basis(A, C, max_general_dim)
        return {**base, "status": "certified",
                "proof": "exact-observability-row-space",
                "quotient_dim": len(T), "observation_dim": len(C),
                "phase_summary": phases,
                "quotient": _quotient_data(T, A, B, b, C, c)}
    except Unresolved as error:
        return {**base, "status": "unresolved", "reason": str(error)}


def certify_quotient(system: ContinuousReLUSystem, *, max_general_dim=12):
    return _compute(system, max_general_dim=max_general_dim)


def verify_quotient(system: ContinuousReLUSystem, certificate):
    if certificate.get("schema") != "ncd.affine-behavioral-quotient.v1":
        raise ValueError("Unsupported affine quotient certificate")
    expected = _compute(system, max_general_dim=certificate["max_general_dim"])
    if certificate != expected:
        raise ValueError("Affine quotient certificate replay mismatch")
    return {"status": expected["status"],
            "quotient_dim": expected.get("quotient_dim"),
            "proof": expected.get("proof")}


def run_case(system: ContinuousReLUSystem, output: Path):
    output = Path(output)
    certificate = certify_quotient(system)
    save_json(output / "certificate.json", certificate)
    return verify_case(system, output)


def verify_case(system: ContinuousReLUSystem, output: Path):
    return verify_quotient(system, read_json(Path(output) / "certificate.json"))


def main():
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
