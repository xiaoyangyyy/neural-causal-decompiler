"""Read-only independent rational replay of the finite-unary-grammar obstruction."""
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ncd.mechanisms import mechanism_library
from ncd.proof_intervals import Interval, hyperbolic_tangent, trigonometric

CERT = ROOT / "validation/frozen_unary_grammar_obstruction_v1.json"
RECEIPT = ROOT / "validation/frozen_unary_grammar_obstruction_verification_v1.json"
CHECKPOINT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_0/trained/mixed/mechanism_1.pt"
SOURCE = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_0/result.json"
SOURCE_VERIFICATION = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_0/verification.json"
NARROW_CERT = ROOT / "validation/frozen_unary_grammar_obstruction_unit_box_v1.json"
NARROW_RECEIPT = ROOT / "validation/frozen_unary_grammar_obstruction_unit_box_verification_v1.json"
NARROW_CHECKPOINT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_5/trained/mixed/mechanism_1.pt"
NARROW_SOURCE = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_5/result.json"
NARROW_SOURCE_VERIFICATION = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_5/verification.json"
POINTS = (Q(-2), Q(-1), Q(0), Q(1, 2), Q(1), Q(3, 2), Q(2))
NARROW_POINTS = (Q(-1), Q(-3, 4), Q(-1, 2), Q(0), Q(1, 2), Q(3, 4), Q(1))
BASIS = ["constant", "linear", "square", "sin", "cos", "tanh"]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(tensor):
    return Q(float(tensor.item()))


def evaluate_layer(weights, biases, inputs, activation):
    output = []
    for row, bias in zip(weights, biases):
        value = Interval.point(number(bias))
        for weight, input_value in zip(row, inputs):
            value = value + Interval.point(number(weight)) * input_value
        output.append(hyperbolic_tangent(value) if activation else value)
    return output


def network_at(state, x):
    z = (Interval.point(x) - number(state["mean"][0])) / number(state["std"][0])
    a = evaluate_layer(
        state["network.0.weight"], state["network.0.bias"], [z], True)
    b = evaluate_layer(
        state["network.2.weight"], state["network.2.bias"], a, True)
    c = evaluate_layer(
        state["network.4.weight"], state["network.4.bias"], b, False)[0]
    return c * number(state["ystd"]) + number(state["ymean"])


def gram_at(x):
    p = Interval.point(x)
    return (Interval.point(1), p, p.square(), trigonometric(p),
            trigonometric(p, cosine=True), hyperbolic_tangent(p))


def independent_inverse(a):
    n = len(a)
    aug = [[Q(a[i][j]) for j in range(n)] +
           [Q(int(i == j)) for j in range(n)] for i in range(n)]
    for j in range(n):
        chosen = next((i for i in range(j, n) if aug[i][j] != 0), None)
        if chosen is None:
            raise ValueError("Anchor basis is singular")
        aug[j], aug[chosen] = aug[chosen], aug[j]
        pivot = aug[j][j]
        for k in range(2 * n):
            aug[j][k] /= pivot
        for i in range(n):
            if i != j:
                factor = aug[i][j]
                for k in range(2 * n):
                    aug[i][k] -= factor * aug[j][k]
    return [row[n:] for row in aug]


def verify(unit_box=False):
    cert_path = NARROW_CERT if unit_box else CERT
    checkpoint_path = NARROW_CHECKPOINT if unit_box else CHECKPOINT
    source_path = NARROW_SOURCE if unit_box else SOURCE
    source_verification_path = (
        NARROW_SOURCE_VERIFICATION if unit_box else SOURCE_VERIFICATION)
    points = NARROW_POINTS if unit_box else POINTS
    expected_unit = [8301, 3, "test_id", 5 if unit_box else 0]
    expected_domain = ("[-1,1] observed parent box; seven rational points"
                       if unit_box else
                       "[-2,2] in observed parent coordinate; the seven listed points suffice")
    cert = read(cert_path)
    if (cert["schema"] != "ncd.frozen-unary-grammar-obstruction.v1"
            or cert["status"] != "proved-for-declared-finite-grammar-and-ideal-real-network"
            or cert["checkpoint"] != str(checkpoint_path.relative_to(ROOT)).replace("\\", "/")
            or cert["checkpoint_sha256"] != digest(checkpoint_path)
            or cert["source_result_sha256"] != digest(source_path)
            or cert["generator_source_sha256"] != digest(
                ROOT / "validation/certify_frozen_unary_grammar_v1.py")
            or cert["grammar"] != BASIS
            or cert["mechanism_library_source_sha256"] != digest(ROOT / "ncd/mechanisms.py")
            or cert["proof_interval_source_sha256"] != digest(ROOT / "ncd/proof_intervals.py")
            or cert["points"] != [str(x) for x in points]
            or cert["basis_midpoint_denominator"] != 1 << 48
            or cert["domain"] != expected_domain
            or cert["refutes_all_candidates_in_declared_grammar"] is not True
            or cert["refutes_all_possible_program_grammars"] is not False
            or cert["original_claim_closed"] is not False
            or cert["original_objective_achieved"] is not False):
        raise ValueError("Certificate identity/scope changed")
    if [name for name, _ in mechanism_library((0,))] != [
            "constant", "linear:0", "square:0", "sin:0", "cos:0", "tanh:0"]:
        raise ValueError("Implemented unary grammar changed")
    source = read(source_path)
    receipt = read(source_verification_path)
    if (source["unit"] != expected_unit
            or source["selected_arms"][1] != "mixed"
            or source["inferred_graph"][0][1] != 1
            or sum(row[1] for row in source["inferred_graph"]) != 1
            or source["training"]["mixed"][1]["checkpoint_sha256"] != digest(checkpoint_path)
            or receipt["status"] != "verified-one-independent-world"
            or receipt["result_sha256"] != digest(source_path)):
        raise ValueError("Frozen source network not independently bound")
    package = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(package["parents"]) != (0,) or package["width"] != 48:
        raise ValueError("Wrong frozen mechanism architecture")
    state = package["state_dict"]
    if number(state["std"][0]) <= 0 or number(state["ystd"]) <= 0:
        raise ValueError("Invalid frozen model scales")
    gram = [gram_at(x) for x in points]
    network = [network_at(state, x) for x in points]
    denominator = 1 << 48
    approx = [[Q(round((v.lo + v.hi) * denominator / 2), denominator)
               for v in row] for row in gram]
    if cert["basis_midpoint"] != [[str(x) for x in row] for row in approx]:
        raise ValueError("Basis midpoint changed")
    if cert["basis_outward_intervals"] != [
            [v.to_dict() for v in row] for row in gram]:
        raise ValueError("Basis enclosure changed")
    if cert["network_outward_intervals"] != [v.to_dict() for v in network]:
        raise ValueError("Neural enclosure changed")
    lam = [Q(x) for x in cert["dual_weights"]]
    if len(lam) != 7 or sum(abs(x) for x in lam) != 1:
        raise ValueError("Invalid dual normalization")
    for j in range(6):
        if sum((lam[i] * approx[i][j] for i in range(7)), Q(0)) != 0:
            raise ValueError("Dual does not annihilate full grammar")
    inverse = independent_inverse(approx[:6])
    if any(sum(approx[i][k] * inverse[k][j] for k in range(6)) != Q(int(i == j))
           for i in range(6) for j in range(6)):
        raise ValueError("Rational anchor inverse is wrong")
    inv_norm = max(sum(abs(x) for x in row) for row in inverse)
    error = [[max(abs(v.lo - q), abs(v.hi - q)) for v, q in zip(row, mid)]
             for row, mid in zip(gram, approx)]
    max_error = max(sum(row) for row in error)
    anchor_error = max(sum(row) for row in error[:6])
    if inv_norm * anchor_error >= 1:
        raise ValueError("Perturbation stability failed")
    target_norm = max(max(abs(y.lo), abs(y.hi)) for y in network[:6])
    dual_target = sum(
        (Interval.point(weight) * interval for weight, interval in zip(lam, network)),
        Interval.point(0))
    lower = dual_target.abs().lo
    epsilon = number(state["ystd"]) / 100
    stable = inv_norm / (1 - inv_norm * anchor_error)
    conditional_coefficient_bound = stable * (target_norm + epsilon)
    conditional_upper = epsilon + max_error * conditional_coefficient_bound
    unconditional_lower = (lower - max_error * stable * target_norm) / (
        1 + max_error * stable)
    asserted = {
        "training_scale": number(state["ystd"]),
        "one_percent_absolute_threshold": epsilon,
        "anchor_inverse_infinity_norm": inv_norm,
        "basis_error_infinity_norm": max_error,
        "coefficient_infinity_bound_if_threshold_met": conditional_coefficient_bound,
        "dual_target_absolute_lower": lower,
        "dual_target_upper_if_threshold_met": conditional_upper,
        "strict_gap": lower - conditional_upper,
        "grid_max_error_lower": unconditional_lower,
    }
    for key, actual in asserted.items():
        if Q(cert[key]) != actual:
            raise ValueError("Derived rational proof field changed: " + key)
    if not (lower > conditional_upper and unconditional_lower > epsilon):
        raise ValueError("Grammar obstruction fails")
    return {
        "schema": "ncd.frozen-unary-grammar-obstruction-verification.v1",
        "status": "verified-finite-grammar-obstruction",
        "certificate_sha256": digest(cert_path),
        "verifier_sha256": digest(__file__),
        "checkpoint_sha256": digest(checkpoint_path),
        "domain": expected_domain,
        "grid_max_error_lower": str(unconditional_lower),
        "normalized_grid_lower": str(unconditional_lower / number(state["ystd"])),
        "threshold": str(epsilon),
        "all_real_coefficients_covered": True,
        "larger_program_grammars_covered": False,
        "device_rounding_covered": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("--write", "--verify") or (
            len(sys.argv) == 3 and sys.argv[2] != "--unit-box"):
        raise SystemExit("Usage: verifier --write|--verify [--unit-box]")
    unit_box = len(sys.argv) == 3
    output = NARROW_RECEIPT if unit_box else RECEIPT
    result = verify(unit_box=unit_box)
    if sys.argv[1] == "--write":
        if output.exists():
            raise FileExistsError("Prior verifier receipt is retained")
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")
    elif read(output) != result:
        raise ValueError("Verifier receipt changed")
    print(json.dumps({
        "status": result["status"],
        "normalized_grid_lower": float(Q(result["normalized_grid_lower"])),
        "original_claim_closed": False,
    }, sort_keys=True))
