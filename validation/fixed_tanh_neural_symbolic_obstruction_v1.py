"""Exact real-semantic neural/symbolic mismatch for one frozen Tanh mechanism."""
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
OUTPUT = ROOT / "validation/fixed_tanh_neural_symbolic_obstruction_v1.json"
MANIFEST_SHA256 = "cb67b63f8fdfddbfc921dd8db00bd4afb135f31e48e9b29bec751f5c74931e40"
MODEL = "model/baseline/mechanism_1.pt"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def q(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite frozen coefficient")
    return Fraction(*number.as_integer_ratio())


def ray_poly(node):
    op = node["op"]
    if op == "constant":
        return {0: q(node["value"])}
    if op == "var":
        if node["index"] not in (0, 2):
            raise ValueError("Unexpected symbolic parent")
        return {1: Fraction(1)}
    if op not in ("add", "mul") or len(node["args"]) < 2:
        raise ValueError("Unsupported symbolic operator")
    out = {0: Fraction(0) if op == "add" else Fraction(1)}
    for arg in node["args"]:
        part = ray_poly(arg)
        if op == "add":
            for exponent, value in part.items():
                out[exponent] = out.get(exponent, Fraction(0)) + value
        else:
            product = {}
            for ea, va in out.items():
                for eb, vb in part.items():
                    if ea + eb > 2:
                        raise ValueError("Symbolic degree exceeds certificate scope")
                    product[ea + eb] = product.get(ea + eb, Fraction(0)) + va * vb
            out = product
    return {exponent: value for exponent, value in out.items() if value}


def compute(package=PACKAGE):
    package = Path(package)
    if digest(package / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original bundle manifest identity changed")
    manifest = read(package / "manifest.json")
    if manifest["schema"] != "ncd.continuous-noise-posthoc-bundle.v1":
        raise ValueError("Wrong archive schema")
    for name, expected in manifest["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Frozen payload changed: " + name)
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    world = read(package / "world.json")
    scm = read(package / "model/structured/explicit_scm.json")
    result = read(package / "result.json")
    graph = [[0, 1, 1], [0, 0, 0], [0, 1, 0]]
    if (scm["source_graph"] != graph or scm["effective_graph"] != graph
            or world["graph"] != graph or world["split"] != "test_id"
            or result["world_id"] != world["world_id"]
            or result["status"] != "posthoc-development-only"
            or result["same_world_across_splits"] is not True
            or result["original_claim_closed"] is not False):
        raise ValueError("Wrong frozen candidate scope")
    checkpoint = torch.load(package / MODEL, map_location="cpu", weights_only=True)
    if (set(checkpoint) != {"parents", "width", "state_dict"}
            or checkpoint["parents"] != (0, 2) or checkpoint["width"] != 48):
        raise ValueError("Wrong frozen neural mechanism")
    state = checkpoint["state_dict"]
    expected_keys = {"mean", "std", "ymean", "ystd",
                     "network.0.weight", "network.0.bias",
                     "network.2.weight", "network.2.bias",
                     "network.4.weight", "network.4.bias"}
    if (set(state) != expected_keys
            or any(not torch.is_tensor(value) or value.dtype != torch.float32
                   or not bool(torch.isfinite(value).all()) for value in state.values())
            or tuple(state["mean"].shape) != (2,)
            or tuple(state["std"].shape) != (2,)
            or tuple(state["ymean"].shape) != ()
            or tuple(state["ystd"].shape) != ()
            or tuple(state["network.4.weight"].shape) != (1, 48)
            or tuple(state["network.4.bias"].shape) != (1,)
            or any(q(value) <= 0 for value in state["std"])
            or q(state["ystd"]) <= 0):
        raise ValueError("Invalid finite float32 checkpoint schema")
    model = NeuralMechanism((0, 2), width=48)
    model.load_state_dict(state, strict=True)
    if [type(layer) for layer in model.network] != [
        nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear
    ]:
        raise ValueError("Frozen network is not the expected Tanh stack")
    yscale = q(state["ystd"])
    output_bound = (abs(q(state["ymean"])) + yscale * (
        abs(q(state["network.4.bias"][0]))
        + sum((abs(q(value)) for value in state["network.4.weight"][0]), Fraction(0))
    ))
    poly = ray_poly(scm["equations"][1])
    if set(poly) != {0, 1, 2} or poly[2] == 0:
        raise ValueError("No nonzero quadratic symbolic ray")
    symbolic = sum((coef * 2**degree for degree, coef in poly.items()), Fraction(0))
    lower = abs(symbolic) - output_bound
    if lower <= yscale / 100:
        raise ValueError("Finite neural-symbolic witness not separated")
    return {
        "schema": "ncd.fixed-tanh-neural-symbolic-obstruction.v1",
        "status": "verified-fixed-candidate-neural-counterexample",
        "bundle_manifest_sha256": MANIFEST_SHA256,
        "frozen_model_sha256": digest(package / MODEL),
        "candidate_scm_sha256": digest(package / "model/structured/explicit_scm.json"),
        "world_sha256": digest(package / "world.json"),
        "checker_source_sha256": digest(__file__),
        "world_id": world["world_id"],
        "network_semantics": "real arithmetic using exact stored binary32 parameters and real tanh",
        "frozen_architecture": "Linear(2,48)-Tanh-Linear(48,48)-Tanh-Linear(48,1), outer affine yscale/ymean",
        "network_global_absolute_upper": str(output_bound),
        "frozen_training_scale": str(yscale),
        "symbolic_ray_coefficients": {str(key): str(poly[key]) for key in (0, 1, 2)},
        "witness_parent_values": {"0": "2", "2": "2"},
        "witness_symbolic_value": str(symbolic),
        "witness_absolute_error_lower": str(lower),
        "witness_normalized_error_lower": str(lower / yscale),
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


def verify(path=OUTPUT, package=PACKAGE):
    actual = read(path)
    expected = compute(package)
    if actual != expected:
        raise ValueError("Frozen neural obstruction certificate mismatch")
    return {"status": expected["status"], "certificate_sha256": digest(path),
            "witness_normalized_error_lower": expected["witness_normalized_error_lower"],
            "original_claim_closed": False}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Retain previous certificate")
        OUTPUT.write_text(json.dumps(compute(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(verify(), sort_keys=True))
