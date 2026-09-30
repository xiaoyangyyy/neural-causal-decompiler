"""Independent replay of the archived Tanh-network/symbolic mismatch."""
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.mechanisms import NeuralMechanism

PACKAGE = ROOT / "validation/continuous_noise_posthoc_package_v1"
CERTIFICATE = ROOT / "validation/fixed_tanh_neural_symbolic_obstruction_v1.json"
OUTPUT = ROOT / "validation/fixed_tanh_neural_symbolic_obstruction_verification_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
GENERATOR_SHA256 = "047d65fd3e8e1a261bb73e7af1baec5d2d44e0dac10f5f3bb1d9793a2d139ec2"
MODEL = "model/baseline/mechanism_1.pt"


def hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite stored parameter")
    return Fraction(Decimal.from_float(number))


def structural_degree(node):
    op = node["op"]
    if op == "constant":
        return 0
    if op == "var":
        if node["index"] not in (0, 2):
            raise ValueError("Unexpected parent")
        return 1
    if op not in ("add", "mul") or len(node["args"]) < 2:
        raise ValueError("Unexpected symbolic operator")
    child_degrees = [structural_degree(child) for child in node["args"]]
    result = max(child_degrees) if op == "add" else sum(child_degrees)
    if result > 2:
        raise ValueError("Symbolic ray degree exceeds two")
    return result


def evaluate(node, t):
    op = node["op"]
    if op == "constant":
        return exact(node["value"])
    if op == "var":
        if node["index"] not in (0, 2):
            raise ValueError("Unexpected parent")
        return t
    parts = [evaluate(child, t) for child in node["args"]]
    if op == "add":
        return sum(parts, Fraction(0))
    if op == "mul":
        total = Fraction(1)
        for part in parts:
            total *= part
        return total
    raise ValueError("Unexpected symbolic operator")


def verify(certificate=CERTIFICATE, package=PACKAGE):
    package = Path(package)
    if hash_file(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original archive manifest changed")
    if hash_file(ROOT / "validation/fixed_tanh_neural_symbolic_obstruction_v1.py") != GENERATOR_SHA256:
        raise ValueError("Generator source changed")
    manifest = load(package / "manifest.json")
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Wrong archive schema")
    for name, expected in manifest["files"].items():
        if hash_file(package / name) != expected:
            raise ValueError("Archived payload changed")
    for name, expected in manifest["source_files"].items():
        if hash_file(ROOT / name) != expected:
            raise ValueError("Archived source changed")
    world = load(package / "world.json")
    scm = load(package / "model/structured/explicit_scm.json")
    result = load(package / "result.json")
    graph = [[0, 1, 1], [0, 0, 0], [0, 1, 0]]
    if (world["graph"] != graph or world["split"] != "test_id"
            or scm["source_graph"] != graph or scm["effective_graph"] != graph
            or result["world_id"] != world["world_id"]
            or result["status"] != "posthoc-development-only"
            or result["same_world_across_splits"] is not True
            or result["original_claim_closed"] is not False):
        raise ValueError("Wrong archived claim scope")
    saved = torch.load(package / MODEL, map_location="cpu", weights_only=True)
    if (set(saved) != {"parents", "width", "state_dict"}
            or saved["parents"] != (0, 2) or saved["width"] != 48):
        raise ValueError("Wrong neural architecture metadata")
    weights = saved["state_dict"]
    names = {"mean", "std", "ymean", "ystd",
             "network.0.weight", "network.0.bias",
             "network.2.weight", "network.2.bias",
             "network.4.weight", "network.4.bias"}
    if set(weights) != names or any(
        not torch.is_tensor(v) or v.dtype != torch.float32 or not bool(torch.isfinite(v).all())
        for v in weights.values()
    ):
        raise ValueError("Bad frozen tensor set")
    model = NeuralMechanism((0, 2), width=48)
    model.load_state_dict(weights, strict=True)
    if [type(layer) for layer in model.network] != [
        nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear
    ] or tuple(weights["network.4.weight"].shape) != (1, 48):
        raise ValueError("Frozen network not a bounded Tanh stack")
    if any(exact(v) <= 0 for v in weights["std"]) or exact(weights["ystd"]) <= 0:
        raise ValueError("Invalid normalization scale")
    scale = exact(weights["ystd"])
    bound = abs(exact(weights["ymean"])) + scale * (
        abs(exact(weights["network.4.bias"][0]))
        + sum((abs(exact(v)) for v in weights["network.4.weight"][0]), Fraction(0))
    )
    expression = scm["equations"][1]
    if structural_degree(expression) != 2:
        raise ValueError("Symbolic ray is not quadratic")
    at_negative = evaluate(expression, Fraction(-1))
    at_zero = evaluate(expression, Fraction(0))
    at_positive = evaluate(expression, Fraction(1))
    at_two = evaluate(expression, Fraction(2))
    a = (at_positive + at_negative - 2 * at_zero) / 2
    b = (at_positive - at_negative) / 2
    if not a or at_two != 4 * a + 2 * b + at_zero:
        raise ValueError("No verified nonzero quadratic ray")
    gap = abs(at_two) - bound
    if gap <= scale / 100:
        raise ValueError("Finite witness fails training-scale threshold")
    expected = {
        "schema": "ncd.fixed-tanh-neural-symbolic-obstruction.v1",
        "status": "verified-fixed-candidate-neural-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "frozen_model_sha256": hash_file(package / MODEL),
        "candidate_scm_sha256": hash_file(package / "model/structured/explicit_scm.json"),
        "world_sha256": hash_file(package / "world.json"),
        "checker_source_sha256": GENERATOR_SHA256,
        "world_id": world["world_id"],
        "network_semantics": "real arithmetic using exact stored binary32 parameters and real tanh",
        "frozen_architecture": "Linear(2,48)-Tanh-Linear(48,48)-Tanh-Linear(48,1), outer affine yscale/ymean",
        "network_global_absolute_upper": str(bound),
        "frozen_training_scale": str(scale),
        "symbolic_ray_coefficients": {"0": str(at_zero), "1": str(b), "2": str(a)},
        "witness_parent_values": {"0": "2", "2": "2"},
        "witness_symbolic_value": str(at_two),
        "witness_absolute_error_lower": str(gap),
        "witness_normalized_error_lower": str(gap / scale),
        "witness_exceeds_1_over_100_training_scale": True,
        "all_real_network_symbolic_supremum_infinite": True,
        "torch_float32_rounding_bound_proved": False,
        "candidate_specific_only": True,
        "oracle_true_mechanism_used": False,
        "intervention_family_includes_witness_proved": False,
        "independent_worlds_for_inference": 0,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
    if load(certificate) != expected:
        raise ValueError("Independent neural fidelity certificate mismatch")
    return {
        "schema": "ncd.fixed-tanh-neural-symbolic-obstruction-verification.v1",
        "status": "independently-verified-fixed-candidate-neural-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "generator_source_sha256": GENERATOR_SHA256,
        "verifier_source_sha256": hash_file(__file__),
        "certificate_sha256": hash_file(certificate),
        "witness_normalized_error_lower": str(gap / scale),
        "all_real_network_symbolic_supremum_infinite": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous verification receipt")
        OUTPUT.write_text(json.dumps(verify(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    receipt = verify()
    if load(OUTPUT) != receipt:
        raise ValueError("Independent neural fidelity receipt mismatch")
    print(json.dumps(receipt, sort_keys=True))
