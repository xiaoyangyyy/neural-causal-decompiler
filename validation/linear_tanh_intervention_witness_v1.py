"""Exact rational witness for one frozen SCM's two-do response contrast.

This refutes a fixed candidate under stated ideal additive SCM semantics.
It does not refute all programs, all noise models, or original R10.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

WORLD = Path("validation/continuous_noise_posthoc_package_v1/world.json")
SCM = Path("validation/continuous_noise_posthoc_package_v1/model/structured/explicit_scm.json")
SELF = Path(__file__)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise ValueError("Expected exact JSON number")
    return Fraction(value)


def require_dict(value, op, args):
    if not isinstance(value, dict) or set(value) != {"op", "args"}:
        raise ValueError("Unexpected expression syntax")
    if value["op"] != op or not isinstance(value["args"], list) or len(value["args"]) != args:
        raise ValueError("Unexpected expression operator")
    return value["args"]


def constant(value):
    if not isinstance(value, dict) or set(value) != {"op", "value"} or value["op"] != "constant":
        raise ValueError("Expected a constant")
    return rational(value["value"])


def extract(world, scm):
    if len(world["graph"]) != 3 or len(scm["source_graph"]) != 3:
        raise ValueError("This witness binds one three-node instance")
    if [row[2] for row in world["graph"]] != [1, 0, 0]:
        raise ValueError("True node 2 parent set changed")
    if [row[2] for row in scm["source_graph"]] != [1, 0, 0]:
        raise ValueError("Estimated node 2 parent set changed")
    if [rational(v) for v in world["scales"]] != [1, 1, 1]:
        raise ValueError("Witness requires observed scale one")
    if world["noise_family"] != "gaussian" or rational(world["noise_scale"]) <= 0:
        raise ValueError("True noise premise changed")
    terms = world["equations"][2]
    if (not isinstance(terms, list) or len(terms) != 1
            or set(terms[0]) != {"operator", "parents", "coefficient"}
            or terms[0]["operator"] != "linear" or terms[0]["parents"] != [0]):
        raise ValueError("True node 2 equation changed")
    true_slope = rational(terms[0]["coefficient"])
    summands = require_dict(scm["equations"][2], "add", 2)
    left = require_dict(summands[0], "mul", 2)
    offset = constant(left[0])
    if constant(left[1]) != 1:
        raise ValueError("Expected identity constant")
    right = require_dict(summands[1], "mul", 2)
    tanh_slope = constant(right[0])
    tanh = require_dict(right[1], "tanh", 1)[0]
    if tanh != {"op": "var", "index": 0}:
        raise ValueError("Expected tanh of intervened root")
    if true_slope <= 0 or tanh_slope <= 0:
        raise ValueError("Positive-slope witness premise changed")
    return true_slope, offset, tanh_slope


def tanh_one_bounds():
    """Bound e^2 by Taylor terms 0..16 and a geometric majorant."""
    term = Fraction(1)
    lower = term
    for k in range(1, 17):
        term *= Fraction(2, k)
        lower += term
    first_omitted = term * Fraction(2, 17)
    upper = lower + first_omitted / (1 - Fraction(2, 18))
    transform = lambda e: (e - 1) / (e + 1)
    return transform(lower), transform(upper)


def derive(world_path=WORLD, scm_path=SCM):
    world = json.loads(Path(world_path).read_text(encoding="utf-8"), parse_float=Decimal)
    scm = json.loads(Path(scm_path).read_text(encoding="utf-8"), parse_float=Decimal)
    true_slope, offset, tanh_slope = extract(world, scm)
    tanh_lower, tanh_upper = tanh_one_bounds()
    true_contrast = 2 * true_slope
    candidate_lower = 2 * tanh_slope * tanh_lower
    candidate_upper = 2 * tanh_slope * tanh_upper
    gap_lower = true_contrast - candidate_upper
    gap_upper = true_contrast - candidate_lower
    if not (0 < tanh_lower < tanh_upper < 1 and gap_lower > Fraction(1, 2)):
        raise ValueError("Strict fixed-candidate witness does not hold")
    return {
        "schema": "ncd.linear-tanh-intervention-witness.v1",
        "status": "refuted-scoped",
        "claim": "This frozen recovered SCM matches the true node-2 paired response contrast between do(X0=-1) and do(X0=1) within absolute error 1/2",
        "world_sha256": digest(world_path),
        "estimated_scm_sha256": digest(scm_path),
        "proof_source_sha256": digest(SELF),
        "intervention_pair": [{"0": "-1"}, {"0": "1"}],
        "true_slope": str(true_slope),
        "candidate_offset": str(offset),
        "candidate_tanh_slope": str(tanh_slope),
        "tanh_one_interval": [str(tanh_lower), str(tanh_upper)],
        "true_pathwise_contrast": str(true_contrast),
        "estimated_pathwise_contrast_interval": [str(candidate_lower), str(candidate_upper)],
        "true_minus_estimated_gap_interval": [str(gap_lower), str(gap_upper)],
        "strict_gap_lower_gt_half": True,
        "noise_assumption": "The same additive node-2 exogenous value is coupled across both interventions; its value cancels pathwise",
        "semantic_scope": "Ideal real-valued interpretation of the serialized decimal coefficients; not floating-device equality",
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(certificate, world_path=WORLD, scm_path=SCM):
    expected = derive(world_path, scm_path)
    if certificate != expected:
        raise ValueError("Fixed-intervention witness certificate mismatch")
    return {"status": "verified-refuted-scoped",
            "original_claim_closed": False,
            "original_objective_achieved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prove", "verify"))
    parser.add_argument("--world", default=str(WORLD))
    parser.add_argument("--scm", default=str(SCM))
    parser.add_argument("--certificate", required=True)
    args = parser.parse_args()
    path = Path(args.certificate)
    if args.mode == "prove":
        certificate = derive(args.world, args.scm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(certificate, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    else:
        certificate = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(verify(certificate, args.world, args.scm), sort_keys=True))


if __name__ == "__main__":
    main()