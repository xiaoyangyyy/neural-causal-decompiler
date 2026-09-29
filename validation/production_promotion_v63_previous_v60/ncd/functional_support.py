"""Exact functional support for small ReLU systems via strict rational regions.

A finite collection of full-dimensional activation cells covers a dense
subset of the open input cube. Zero coordinate slope in every feasible cell
therefore proves coordinate independence on the whole closed cube by
continuity. A nonzero slope yields an exact in-cell intervention witness.
Resource limits return unresolved rather than an unsound classification.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from pathlib import Path

from .continuous_separation import ContinuousReLUSystem, ReLUMLP, _digest
from .interventional_support import (
    _checked_witnesses, _exact_value, _possible, propose_support)
from .io import read_json, save_json


class RegionLimit(Exception):
    """Exact enumeration reached its declared resource cap."""


def _negative(form):
    return tuple(-x for x in form)


def _zero(width):
    return (Q(0),) * (width + 1)


def _strict_point(constraints, width, *, max_constraints=5000):
    """Exact Fourier-Motzkin feasibility and rational point for affine > 0."""
    stages = []
    current = list(constraints)
    for dim in range(width, 0, -1):
        if len(current) > max_constraints:
            raise RegionLimit("Fourier-Motzkin constraint cap")
        stages.append(current)
        positive = [c for c in current if c[dim - 1] > 0]
        negative = [c for c in current if c[dim - 1] < 0]
        next_constraints = [c[:dim - 1] + (c[-1],)
                            for c in current if c[dim - 1] == 0]
        for lower in positive:
            for upper in negative:
                next_constraints.append(tuple(
                    lower[k] / lower[dim - 1] - upper[k] / upper[dim - 1]
                    for k in range(dim - 1)) +
                    (lower[-1] / lower[dim - 1] -
                     upper[-1] / upper[dim - 1],))
                if len(next_constraints) > max_constraints:
                    raise RegionLimit("Fourier-Motzkin constraint cap")
        current = next_constraints
        if any(c[-1] <= 0 for c in current if len(c) == 1):
            return None
    if any(c[0] <= 0 for c in current):
        return None
    point = []
    for dim, stage in enumerate(reversed(stages), 1):
        lower, upper = [], []
        for c in stage:
            rest = c[-1] + sum(c[k] * point[k] for k in range(dim - 1))
            coefficient = c[dim - 1]
            if coefficient > 0:
                lower.append(-rest / coefficient)
            elif coefficient < 0:
                upper.append(-rest / coefficient)
            elif rest <= 0:
                raise AssertionError("Invalid Fourier-Motzkin reconstruction")
        lo = max(lower) if lower else None
        hi = min(upper) if upper else None
        if lo is not None and hi is not None:
            if lo >= hi:
                raise AssertionError("Infeasible reconstructed bounds")
            x = (lo + hi) / 2
        elif lo is not None:
            x = lo + 1
        elif hi is not None:
            x = hi - 1
        else:
            x = Q(0)
        point.append(x)
    answer = tuple(point)
    if not all(sum(c[k] * answer[k] for k in range(width)) + c[-1] > 0
               for c in constraints):
        raise AssertionError("Fourier-Motzkin witness failed exact replay")
    return answer


def _domain(width):
    constraints = []
    for j in range(width):
        lower = [Q(0)] * (width + 1)
        upper = [Q(0)] * (width + 1)
        lower[j] = Q(1)
        upper[j], upper[-1] = Q(-1), Q(1)
        constraints.extend((tuple(lower), tuple(upper)))
    return tuple(constraints)


def _affine(row, bias, forms):
    width = len(forms[0]) - 1
    return tuple(sum((Q(w) * form[k] for w, form in zip(row, forms)), Q(0))
                 + (Q(bias) if k == width else Q(0))
                 for k in range(width + 1))


def _regions(net: ReLUMLP, *, max_input_dim=3, max_hidden=12,
             max_regions=4096, max_constraints=5000):
    width = net.input_dim
    hidden = sum(len(layer) for layer in net.weights[:-1])
    if width > max_input_dim or hidden > max_hidden:
        raise RegionLimit("Network exceeds exact-region scope")
    identity = tuple(tuple(Q(int(k == j)) for k in range(width)) + (Q(0),)
                     for j in range(width))
    initial = _domain(width)
    regions = [(initial, identity, _strict_point(initial, width,
                                                max_constraints=max_constraints))]
    feasible_branches = 0
    infeasible_branches = 0
    for layer, biases in zip(net.weights[:-1], net.biases[:-1]):
        next_regions = []
        for constraints, forms, point in regions:
            preactivations = tuple(_affine(row, bias, forms)
                                   for row, bias in zip(layer, biases))
            partial = [(constraints, tuple(), point)]
            for pre in preactivations:
                branches = []
                for current_constraints, current_forms, _ in partial:
                    if pre == _zero(width):
                        branches.append((current_constraints,
                                         current_forms + (_zero(width),),
                                         _strict_point(current_constraints, width,
                                                       max_constraints=max_constraints)))
                        continue
                    for sign, outgoing in ((1, pre), (-1, _zero(width))):
                        candidate = current_constraints + (
                            pre if sign == 1 else _negative(pre),)
                        inside = _strict_point(candidate, width,
                                               max_constraints=max_constraints)
                        if inside is None:
                            infeasible_branches += 1
                        else:
                            feasible_branches += 1
                            branches.append((candidate, current_forms + (outgoing,), inside))
                partial = branches
                if len(partial) + len(next_regions) > max_regions:
                    raise RegionLimit("Region count cap")
            next_regions.extend(partial)
            if len(next_regions) > max_regions:
                raise RegionLimit("Region count cap")
        regions = next_regions
    output = []
    for constraints, forms, point in regions:
        affine_outputs = tuple(_affine(row, bias, forms)
                               for row, bias in zip(net.weights[-1], net.biases[-1]))
        output.append((constraints, affine_outputs, point))
    return output, {"region_count": len(output),
                    "feasible_branches": feasible_branches,
                    "infeasible_branches": infeasible_branches}


def _in_region_witness(net, constraints, point, output, coordinate):
    bounds = []
    for c in constraints:
        slope = c[coordinate]
        if slope < 0:
            margin = c[-1] + sum(c[k] * point[k] for k in range(len(point)))
            bounds.append(margin / -slope)
    delta = min(bounds) / 2
    right = list(point)
    right[coordinate] += delta
    right = tuple(right)
    difference = _exact_value(net, point)[output] - _exact_value(net, right)[output]
    if difference == 0:
        raise AssertionError("A nonzero cell slope failed to produce an edge")
    return {"output": output, "input": coordinate,
            "left": [str(x) for x in point],
            "right": [str(x) for x in right],
            "exact_difference": str(difference),
            "origin": "exact-region-search"}


def certify_functional_support(system: ContinuousReLUSystem, proposal: dict,
                               *, max_input_dim=3, max_hidden=12,
                               max_regions=4096, max_constraints=5000) -> dict:
    """Extend query witnesses with exact region witnesses and absence proofs."""
    witnesses = _checked_witnesses(system, proposal)
    present = {(w["output"], w["input"]) for w in witnesses}
    possible = _possible(system.transition)
    candidates = [(i, j) for i in range(system.state_dim)
                  for j in range(system.state_dim + system.action_dim)
                  if possible[i][j] and (i, j) not in present]
    region_proof = None
    functionally_absent = []
    unresolved = []
    if candidates:
        try:
            regions, region_proof = _regions(
                system.transition, max_input_dim=max_input_dim,
                max_hidden=max_hidden, max_regions=max_regions,
                max_constraints=max_constraints)
        except RegionLimit as error:
            unresolved = [[i, j] for i, j in candidates]
            region_proof = {"status": "resource_limit", "reason": str(error)}
        else:
            for i, j in candidates:
                match = next(((constraints, point)
                              for constraints, affine, point in regions
                              if affine[i][j] != 0), None)
                if match is None:
                    functionally_absent.append([i, j])
                else:
                    witnesses.append(_in_region_witness(
                        system.transition, match[0], match[1], i, j))
    witnesses.sort(key=lambda x: (x["output"], x["input"]))
    total = system.state_dim * (system.state_dim + system.action_dim)
    return {"schema": "ncd.functional-support-certificate.v1",
            "system_sha256": _digest(system.to_dict()),
            "state_dim": system.state_dim,
            "action_dim": system.action_dim,
            "domain": "full unit state and action cubes",
            "limits": {"max_input_dim": max_input_dim, "max_hidden": max_hidden,
                       "max_regions": max_regions,
                       "max_constraints": max_constraints},
            "witnesses": witnesses,
            "functional_absence": functionally_absent,
            "region_proof": region_proof,
            "unresolved": unresolved,
            "absent_count": total - len(witnesses) - len(unresolved),
            "status": "certified" if not unresolved else "unresolved"}


def verify_functional_support(system: ContinuousReLUSystem, proposal: dict,
                              certificate: dict) -> dict:
    limits = certificate["limits"]
    expected = certify_functional_support(system, proposal, **limits)
    if certificate != expected:
        raise ValueError("Functional support certificate replay mismatch")
    return {"status": expected["status"],
            "present_count": len(expected["witnesses"]),
            "absent_count": expected["absent_count"],
            "unresolved_count": len(expected["unresolved"]),
            "functional_absence_count": len(expected["functional_absence"]),
            "region_witness_count": sum(w.get("origin") == "exact-region-search"
                                        for w in expected["witnesses"])}


def run_case(system: ContinuousReLUSystem, output: Path) -> dict:
    output = Path(output)
    proposal = propose_support(system.step, system.state_dim, system.action_dim)
    certificate = certify_functional_support(system, proposal)
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
    return verify_functional_support(
        system, proposal, read_json(output / "certificate.json"))


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
