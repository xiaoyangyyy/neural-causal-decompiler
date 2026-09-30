"""Construct a rational dual obstruction for the frozen one-parent grammar.

All binary32 checkpoint parameters are interpreted as exact rationals, with
ideal-real affine/Tanh operations. This excludes extra device rounding.
"""
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ncd.mechanisms import mechanism_library
from ncd.proof_intervals import (
    Interval, affine, hyperbolic_tangent, trigonometric,
)

CHECKPOINT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_0/trained/mixed/mechanism_1.pt"
RESULT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_0/result.json"
OUTPUT = ROOT / "validation/frozen_unary_grammar_obstruction_v1.json"
NARROW_CHECKPOINT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_5/trained/mixed/mechanism_1.pt"
NARROW_RESULT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_5/result.json"
NARROW_OUTPUT = ROOT / "validation/frozen_unary_grammar_obstruction_unit_box_v1.json"
POINTS = [Q(-2), Q(-1), Q(0), Q(1, 2), Q(1), Q(3, 2), Q(2)]
NARROW_POINTS = [Q(-1), Q(-3, 4), Q(-1, 2), Q(0), Q(1, 2), Q(3, 4), Q(1)]
BASIS = ["constant", "linear", "square", "sin", "cos", "tanh"]
MIDPOINT_DEN = 1 << 48


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def intervals_for_basis(x):
    p = Interval.point(x)
    return [Interval.point(1), p, p.square(),
            trigonometric(p), trigonometric(p, cosine=True),
            hyperbolic_tangent(p)]


def scalar(value):
    return Q(float(value.item()))


def vector(value):
    return [scalar(x) for x in value]


def matrix(value):
    return [vector(row) for row in value]


def neural_interval(state, x):
    z = (Interval.point(x) - scalar(state["mean"][0])) / scalar(state["std"][0])
    first = [hyperbolic_tangent(v) for v in affine(
        matrix(state["network.0.weight"]), vector(state["network.0.bias"]), [z])]
    second = [hyperbolic_tangent(v) for v in affine(
        matrix(state["network.2.weight"]), vector(state["network.2.bias"]), first)]
    value = affine(matrix(state["network.4.weight"]),
                   vector(state["network.4.bias"]), second)[0]
    return value * scalar(state["ystd"]) + scalar(state["ymean"])


def inverse(matrix_values):
    n = len(matrix_values)
    table = [[Q(x) for x in row] + [Q(int(i == j)) for j in range(n)]
             for i, row in enumerate(matrix_values)]
    for col in range(n):
        pivot = next((row for row in range(col, n) if table[row][col]), None)
        if pivot is None:
            raise ValueError("Basis anchor matrix singular")
        table[col], table[pivot] = table[pivot], table[col]
        factor = table[col][col]
        table[col] = [x / factor for x in table[col]]
        for row in range(n):
            if row == col:
                continue
            factor = table[row][col]
            table[row] = [x - factor * y for x, y in zip(table[row], table[col])]
    return [row[n:] for row in table]


def compute(checkpoint_path=CHECKPOINT, result_path=RESULT, points=POINTS,
            domain="[-2,2] in observed parent coordinate; the seven listed points suffice"):
    library_names = [name for name, _ in mechanism_library((0,))]
    if library_names != [
            "constant", "linear:0", "square:0", "sin:0", "cos:0", "tanh:0"]:
        raise ValueError("Declared unary grammar differs from implementation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["parents"]) != (0,) or checkpoint["width"] != 48:
        raise ValueError("Wrong frozen unary checkpoint")
    state = checkpoint["state_dict"]
    if not scalar(state["std"][0]) > 0 or not scalar(state["ystd"]) > 0:
        raise ValueError("Invalid frozen scales")
    basis_intervals = [intervals_for_basis(x) for x in points]
    target_intervals = [neural_interval(state, x) for x in points]
    approximated = [[Q(round((v.lo + v.hi) * MIDPOINT_DEN / 2), MIDPOINT_DEN)
                     for v in row] for row in basis_intervals]
    deviations = [[max(abs(v.lo - q), abs(v.hi - q))
                   for v, q in zip(row, approximate)]
                  for row, approximate in zip(basis_intervals, approximated)]
    anchor = approximated[:6]
    inv = inverse(anchor)
    right = approximated[6]
    dual = [-sum((right[j] * inv[j][i] for j in range(6)), Q(0))
            for i in range(6)] + [Q(1)]
    norm = sum(map(abs, dual))
    dual = [value / norm for value in dual]
    if any(sum(dual[i] * approximated[i][j] for i in range(7))
           for j in range(6)):
        raise ValueError("Rational dual does not annihilate approximate grammar")
    if sum(map(abs, dual)) != 1:
        raise ValueError("Dual normalization failed")
    inverse_norm = max(sum(map(abs, row)) for row in inv)
    error_norm = max(sum(row) for row in deviations)
    anchor_error_norm = max(sum(row) for row in deviations[:6])
    target_bound = max(max(abs(y.lo), abs(y.hi)) for y in target_intervals[:6])
    epsilon = scalar(state["ystd"]) / 100
    if inverse_norm * anchor_error_norm >= 1:
        raise ValueError("Perturbed basis inverse bound failed")
    stable_inverse_norm = inverse_norm / (
        1 - inverse_norm * anchor_error_norm)
    coefficient_bound = stable_inverse_norm * (target_bound + epsilon)
    dual_target = sum(
        (Interval.point(weight) * value for weight, value in
         zip(dual, target_intervals)), Interval.point(0))
    lower = dual_target.abs().lo
    admissible_upper = epsilon + error_norm * coefficient_bound
    global_lower = (
        lower - error_norm * stable_inverse_norm * target_bound
    ) / (1 + error_norm * stable_inverse_norm)
    if lower <= admissible_upper or global_lower <= epsilon:
        raise ValueError("No strict grammar-wide obstruction")
    certificate = {
        "schema": "ncd.frozen-unary-grammar-obstruction.v1",
        "status": "proved-for-declared-finite-grammar-and-ideal-real-network",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": digest(checkpoint_path),
        "source_result_sha256": digest(result_path),
        "generator_source_sha256": digest(__file__),
        "network_semantics": "binary32 parameters exact; ideal-real affine and tanh; no device rounding",
        "grammar": BASIS,
        "mechanism_library_source_sha256": digest(ROOT / "ncd/mechanisms.py"),
        "proof_interval_source_sha256": digest(ROOT / "ncd/proof_intervals.py"),
        "candidate_family": "all real linear combinations of the six basis functions; stronger than 8-term structured search",
        "domain": domain,
        "points": [str(x) for x in points],
        "basis_midpoint_denominator": MIDPOINT_DEN,
        "basis_midpoint": [[str(x) for x in row] for row in approximated],
        "basis_outward_intervals": [[v.to_dict() for v in row] for row in basis_intervals],
        "network_outward_intervals": [v.to_dict() for v in target_intervals],
        "dual_weights": [str(x) for x in dual],
        "dual_l1_norm": "1",
        "training_scale": str(scalar(state["ystd"])),
        "one_percent_absolute_threshold": str(epsilon),
        "anchor_inverse_infinity_norm": str(inverse_norm),
        "basis_error_infinity_norm": str(error_norm),
        "coefficient_infinity_bound_if_threshold_met": str(coefficient_bound),
        "dual_target_absolute_lower": str(lower),
        "dual_target_upper_if_threshold_met": str(admissible_upper),
        "strict_gap": str(lower - admissible_upper),
        "grid_max_error_lower": str(global_lower),
        "refutes_all_candidates_in_declared_grammar": True,
        "refutes_all_possible_program_grammars": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    return certificate


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("--write", "--verify") or (
            len(sys.argv) == 3 and sys.argv[2] != "--unit-box"):
        raise SystemExit("Usage: script --write|--verify [--unit-box]")
    unit_box = len(sys.argv) == 3
    output = NARROW_OUTPUT if unit_box else OUTPUT
    expected = (compute(NARROW_CHECKPOINT, NARROW_RESULT, NARROW_POINTS,
                        "[-1,1] observed parent box; seven rational points")
                if unit_box else compute())
    if sys.argv[1] == "--write":
        if output.exists():
            raise FileExistsError("Prior proof certificate is retained")
        output.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    else:
        observed = json.loads(output.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("Grammar obstruction certificate changed")
    print(json.dumps({
        "status": expected["status"],
        "strict_gap": float(Q(expected["strict_gap"])),
        "normalized_lower_error": float(
            Q(expected["grid_max_error_lower"]) / Q(expected["training_scale"])),
        "original_claim_closed": False,
    }, sort_keys=True))
