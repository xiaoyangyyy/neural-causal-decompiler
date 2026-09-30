"""Generate one exact nonlinear paired-do development certificate."""
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ncd.paired_nonlinear_graph import recover_local_graph

OUTPUT = ROOT / "validation/paired_nonlinear_graph_case_v1.json"
SOURCES = ("ncd/paired_linear_graph.py", "ncd/paired_nonlinear_graph.py",
           "validation/paired_nonlinear_graph_case_v1.py",
           "validation/verify_paired_nonlinear_graph_case_v1.py")


def response(source, h):
    x = [Q(0)] * 3
    for j in range(3):
        if j == source:
            x[j] = h
        elif j == 1:
            x[j] = Q(3, 5) * x[0] + Q(1, 10) * x[0] ** 2
        elif j == 2:
            x[j] = Q(2, 5) * x[0] + Q(1, 2) * x[1] + Q(1, 20) * x[1] ** 2
    return [str(v) for v in x]


def compute():
    h = Q(1, 16)
    inputs = {
        "baseline": ["0"] * 3,
        "responses": [response(i, h) for i in range(3)],
        "step": str(h),
        "curvature_bound": "1/5",
        "response_error_bound": "0",
        "minimum_visible_direct_effect": "2/5",
    }
    result = recover_local_graph(**inputs)
    return {
        "schema": "ncd.paired-nonlinear-development-certificate.v1",
        "status": "exact-rational-conditional-graph-certificate",
        "source_sha256": {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        "estimator_input": inputs,
        "estimator_result": result,
        "truth_only_equations": [
            "X0=U0", "X1=3/5 X0+1/10 X0^2+U1",
            "X2=2/5 X0+1/2 X1+1/20 X1^2+U2",
        ],
        "truth_only_graph_source_target": [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
        "exogenous_state": ["0"] * 3,
        "conditioning": "one shared exogenous state for baseline and every do response",
        "device_rounding_certified": False,
        "independent_noise_draws_certified": False,
        "original_claim_closed": False,
    }


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError("Existing exact certificate retained")
    record = compute()
    OUTPUT.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"],
                      "graph": record["estimator_result"]["graph_source_target"]},
                     sort_keys=True))