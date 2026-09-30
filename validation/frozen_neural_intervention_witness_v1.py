"""Rational enclosure of one frozen two-Tanh mechanism at two interventions.

The checker extracts binary32 weights, then evaluates ideal real arithmetic with
outward rational rounding. PyTorch forward is not used to establish the bound.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "validation/continuous_noise_posthoc_package_v1/model/baseline/mechanism_2.pt"
WORLD = ROOT / "validation/continuous_noise_posthoc_package_v1/world.json"
MECHANISMS = ROOT / "ncd/mechanisms.py"
Q = 10**9
TAYLOR_DEGREE = 64
EXPECTED_MODEL_SHA256 = "81802ba722f173f35594d02f738e78ba8975284a12099af25bbb5b1a73f146eb"
EXPECTED_WORLD_SHA256 = "0ad38c1b7844caca150f6c38e90b4ccd08d8a3ee9421ddba944b3d9757a192cf"
EXPECTED_FORWARD_SHA256 = "35f78ebe2e2051a15d0b22e145b702146d169e53e4dce349c8bdc9276b389624"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fraction(value):
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("Frozen parameter must be finite binary32")
    return Fraction.from_float(value)


def outward(lower, upper):
    if lower > upper:
        raise ValueError("Invalid interval")
    lo = lower.numerator * Q // lower.denominator
    hi = -((-upper.numerator * Q) // upper.denominator)
    return Fraction(lo, Q), Fraction(hi, Q)


def affine(coeffs, intervals, bias):
    lo = hi = bias
    for coefficient, (a, b) in zip(coeffs, intervals, strict=True):
        if coefficient >= 0:
            lo += coefficient * a
            hi += coefficient * b
        else:
            lo += coefficient * b
            hi += coefficient * a
    return outward(lo, hi)


def exp_positive_bounds(value):
    if value < 0 or value >= 66:
        raise ValueError("Taylor bound supports nonnegative arguments below 66")
    term = total = Fraction(1)
    for k in range(1, TAYLOR_DEGREE + 1):
        term *= value / k
        total += term
    first_omitted = term * value / (TAYLOR_DEGREE + 1)
    ratio = value / (TAYLOR_DEGREE + 2)
    return total, total + first_omitted / (1 - ratio)


def tanh_point_bounds(value):
    if value < 0:
        lo, hi = tanh_point_bounds(-value)
        return -hi, -lo
    e_lo, e_hi = exp_positive_bounds(2 * value)
    transform = lambda e: (e - 1) / (e + 1)
    return transform(e_lo), transform(e_hi)


def tanh_interval(interval):
    lo = tanh_point_bounds(interval[0])[0]
    hi = tanh_point_bounds(interval[1])[1]
    return outward(lo, hi)


def parameters(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != {"parents", "width", "state_dict"}:
        raise ValueError("Unexpected checkpoint schema")
    if tuple(state["parents"]) != (0,) or state["width"] != 48:
        raise ValueError("Wrong frozen mechanism binding")
    weights = state["state_dict"]
    shapes = {
        "mean": (1,), "std": (1,), "ymean": (), "ystd": (),
        "network.0.weight": (48, 1), "network.0.bias": (48,),
        "network.2.weight": (48, 48), "network.2.bias": (48,),
        "network.4.weight": (1, 48), "network.4.bias": (1,),
    }
    if set(weights) != set(shapes):
        raise ValueError("Unexpected parameter keys")
    for key, shape in shapes.items():
        value = weights[key]
        if tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError("Unexpected parameter shape or type: " + key)
    scalar = lambda key: fraction(float(weights[key]))
    vector = lambda key: [fraction(x) for x in weights[key].tolist()]
    matrix = lambda key: [[fraction(x) for x in row]
                          for row in weights[key].tolist()]
    return {
        "mean": fraction(weights["mean"][0].item()),
        "std": fraction(weights["std"][0].item()),
        "ymean": scalar("ymean"),
        "ystd": scalar("ystd"),
        "w0": matrix("network.0.weight"),
        "b0": vector("network.0.bias"),
        "w2": matrix("network.2.weight"),
        "b2": vector("network.2.bias"),
        "w4": matrix("network.4.weight")[0],
        "b4": vector("network.4.bias")[0],
    }


def forward_interval(params, x):
    if params["std"] <= 0 or params["ystd"] <= 0:
        raise ValueError("Frozen normalizers must be positive")
    normalized = outward((x - params["mean"]) / params["std"],
                         (x - params["mean"]) / params["std"])
    h1 = [tanh_interval(affine(row, [normalized], bias))
          for row, bias in zip(params["w0"], params["b0"], strict=True)]
    h2 = [tanh_interval(affine(row, h1, bias))
          for row, bias in zip(params["w2"], params["b2"], strict=True)]
    lo, hi = affine(params["w4"], h2, params["b4"])
    return outward(lo * params["ystd"] + params["ymean"],
                   hi * params["ystd"] + params["ymean"])


def derive(model_path=MODEL, world_path=WORLD):
    if (digest(model_path) != EXPECTED_MODEL_SHA256
            or digest(world_path) != EXPECTED_WORLD_SHA256
            or digest(MECHANISMS) != EXPECTED_FORWARD_SHA256):
        raise ValueError("Frozen neural/world/source hash mismatch")
    world = json.loads(Path(world_path).read_text(encoding="utf-8"),
                       parse_float=Decimal)
    if ([row[2] for row in world["graph"]] != [1, 0, 0]
            or [Fraction(v) for v in world["scales"]] != [1, 1, 1]
            or world["noise_family"] != "gaussian"
            or Fraction(world["noise_scale"]) <= 0):
        raise ValueError("True-world witness premises changed")
    terms = world["equations"][2]
    if (len(terms) != 1 or terms[0]["operator"] != "linear"
            or terms[0]["parents"] != [0]):
        raise ValueError("True node-2 mechanism changed")
    true_slope = Fraction.from_float(float(terms[0]["coefficient"]))
    params = parameters(model_path)
    minus = forward_interval(params, Fraction(-1))
    plus = forward_interval(params, Fraction(1))
    neural_contrast = outward(plus[0] - minus[1], plus[1] - minus[0])
    true_contrast = 2 * true_slope
    gap = (true_contrast - neural_contrast[1],
           true_contrast - neural_contrast[0])
    minus_point_gap = (minus[0] + true_slope, minus[1] + true_slope)
    plus_point_gap = (true_slope - plus[1], true_slope - plus[0])
    normalized_minus = (minus_point_gap[0] / params["ystd"],
                        minus_point_gap[1] / params["ystd"])
    normalized_plus = (plus_point_gap[0] / params["ystd"],
                       plus_point_gap[1] / params["ystd"])
    if (minus_point_gap[0] <= Fraction(8, 25)
            or plus_point_gap[0] <= Fraction(8, 25)
            or normalized_minus[0] <= Fraction(3, 5)
            or normalized_plus[0] <= Fraction(3, 5)):
        raise ValueError("Strict pointwise neural-truth gap not established")
    if gap[0] <= Fraction(13, 20):
        raise ValueError("Strict neural-truth intervention gap not established")
    fmt = lambda interval: [str(v) for v in interval]
    return {
        "schema": "ncd.frozen-neural-intervention-witness.v1",
        "status": "refuted-scoped",
        "claim": "This frozen neural mechanism's node-2 paired response contrast differs from the true mechanism by at most 13/20",
        "model_sha256": digest(model_path),
        "world_sha256": digest(world_path),
        "forward_source_sha256": digest(MECHANISMS),
        "proof_source_sha256": digest(__file__),
        "semantics": "Ideal real arithmetic with the exact binary64 true coefficient loaded from JSON, exact binary32 neural parameters, and rational Tanh enclosures; not device-rounding equality",
        "intervention_pair": [{"0": "-1"}, {"0": "1"}],
        "taylor_degree": TAYLOR_DEGREE,
        "outward_grid_denominator": Q,
        "neural_at_minus_one_interval": fmt(minus),
        "neural_at_plus_one_interval": fmt(plus),
        "neural_contrast_interval": fmt(neural_contrast),
        "true_contrast": str(true_contrast),
        "frozen_training_ystd": str(params["ystd"]),
        "minus_one_neural_minus_true_gap_interval": fmt(minus_point_gap),
        "plus_one_true_minus_neural_gap_interval": fmt(plus_point_gap),
        "minus_one_normalized_gap_interval": fmt(normalized_minus),
        "plus_one_normalized_gap_interval": fmt(normalized_plus),
        "true_minus_neural_gap_interval": fmt(gap),
        "strict_pointwise_normalized_gap_lower_gt_3_over_5": True,
        "strict_gap_lower_gt_13_over_20": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(certificate, model_path=MODEL, world_path=WORLD):
    if certificate != derive(model_path, world_path):
        raise ValueError("Frozen neural intervention certificate mismatch")
    return {"status": "verified-refuted-scoped",
            "original_claim_closed": False,
            "original_objective_achieved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prove", "verify"))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--world", default=str(WORLD))
    parser.add_argument("--certificate", required=True)
    args = parser.parse_args()
    path = Path(args.certificate)
    if args.mode == "prove":
        certificate = derive(args.model, args.world)
        path.write_text(json.dumps(certificate, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    else:
        certificate = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify(certificate, args.model, args.world),
                         sort_keys=True))


if __name__ == "__main__":
    main()