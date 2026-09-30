"""Independent exact-arithmetic check of the nonlinear paired-do certificate."""
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "validation/paired_nonlinear_graph_case_v1.json"
OUTPUT = ROOT / "validation/paired_nonlinear_graph_case_verification_v1.json"
SOURCES = ("ncd/paired_linear_graph.py", "ncd/paired_nonlinear_graph.py",
           "validation/paired_nonlinear_graph_case_v1.py",
           "validation/verify_paired_nonlinear_graph_case_v1.py")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inverse3(a):
    z = a
    c = [
        [z[1][1]*z[2][2]-z[1][2]*z[2][1], z[1][2]*z[2][0]-z[1][0]*z[2][2], z[1][0]*z[2][1]-z[1][1]*z[2][0]],
        [z[0][2]*z[2][1]-z[0][1]*z[2][2], z[0][0]*z[2][2]-z[0][2]*z[2][0], z[0][1]*z[2][0]-z[0][0]*z[2][1]],
        [z[0][1]*z[1][2]-z[0][2]*z[1][1], z[0][2]*z[1][0]-z[0][0]*z[1][2], z[0][0]*z[1][1]-z[0][1]*z[1][0]],
    ]
    det = sum((z[0][j] * c[0][j] for j in range(3)), Q(0))
    if det == 0:
        raise ValueError("Singular finite-difference matrix")
    return [[c[j][i] / det for j in range(3)] for i in range(3)]


def matmul(a, b):
    return [[sum((a[i][k]*b[k][j] for k in range(3)), Q(0))
             for j in range(3)] for i in range(3)]


def norm(a):
    return max(sum((abs(v) for v in row), Q(0)) for row in a)


def verify(path=CERTIFICATE):
    record = read(path)
    expected_sources = {p: digest(ROOT / p) for p in SOURCES}
    if record.get("source_sha256") != expected_sources:
        raise ValueError("Certificate source binding changed")
    h = Q(1, 16)
    def response(i):
        x0 = h if i == 0 else Q(0)
        x1 = h if i == 1 else Q(3, 5)*x0 + Q(1, 10)*x0*x0
        x2 = h if i == 2 else Q(2, 5)*x0 + Q(1, 2)*x1 + Q(1, 20)*x1*x1
        return [str(x0), str(x1), str(x2)]
    inputs = {
        "baseline": ["0"]*3, "responses": [response(i) for i in range(3)],
        "step": "1/16", "curvature_bound": "1/5",
        "response_error_bound": "0", "minimum_visible_direct_effect": "2/5",
    }
    if record.get("estimator_input") != inputs:
        raise ValueError("Paired response or assumption changed")
    # Along do(X0=t), g(t)=3t/5+t^2/10 and
    # X2''=1/10+1/10[(g')^2+g*g'']. For t in [0,h], both g and g'
    # are nonnegative and increasing. Other sources have curvature <= 1/10.
    gmax = Q(3, 5)*h + Q(1, 10)*h*h
    gpmax = Q(3, 5) + Q(1, 5)*h
    x2_curvature_upper = Q(1, 10) + Q(1, 10)*(gpmax*gpmax + gmax/Q(5))
    if x2_curvature_upper > Q(1, 5):
        raise ValueError("Curvature assumption is false")
    direct_true = [[Q(0), Q(0), Q(0)],
                   [Q(3, 5), Q(0), Q(0)],
                   [Q(2, 5), Q(1, 2), Q(0)]]
    visible = [abs(direct_true[i][j]) for i in range(3) for j in range(3)
               if i != j and direct_true[i][j] != 0]
    if not visible or min(visible) < Q(2, 5):
        raise ValueError("Direct-effect visibility assumption is false")
    total_true = [[Q(1), Q(0), Q(0)],
                  [Q(3, 5), Q(1), Q(0)],
                  [Q(7, 10), Q(1, 2), Q(1)]]
    identity = [[Q(int(i == j)) for j in range(3)] for i in range(3)]
    if matmul([[identity[i][j]-direct_true[i][j] for j in range(3)]
               for i in range(3)], total_true) != identity:
        raise ValueError("Local total-effect identity failed")
    total_hat = [[Q(response(source)[target])/h for source in range(3)]
                 for target in range(3)]
    inv = inverse3(total_hat)
    if matmul(total_hat, inv) != identity or matmul(inv, total_hat) != identity:
        raise ValueError("Finite-difference inverse failed")
    k = norm(inv)
    entry_bound = Q(1, 5)*h/2
    matrix_bound = 3*entry_bound
    if norm([[total_hat[i][j]-total_true[i][j] for j in range(3)]
             for i in range(3)]) > matrix_bound or k*matrix_bound >= 1:
        raise ValueError("Total-effect perturbation bound failed")
    direct_error = k*k*matrix_bound/(1-k*matrix_bound)
    if 2*direct_error >= Q(2, 5):
        raise ValueError("Direct-effect separation failed")
    direct_hat = [[identity[i][j]-inv[i][j] for j in range(3)]
                  for i in range(3)]
    if norm([[direct_hat[i][j]-direct_true[i][j] for j in range(3)]
             for i in range(3)]) > direct_error:
        raise ValueError("Direct-effect error bound failed")
    graph = [[0 if i == j else int(abs(direct_hat[j][i]) > Q(1, 5))
              for j in range(3)] for i in range(3)]
    expected_graph = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    if graph != expected_graph:
        raise ValueError("Graph does not follow from separated effects")
    expected_result = {
        "schema": "ncd.paired-nonlinear-local-graph.v1",
        "status": "conditional-on-pairing-curvature-and-direct-effect-margin",
        "total_effect_estimate": [[str(v) for v in row] for row in total_hat],
        "inverse_total_effect_estimate": [[str(v) for v in row] for row in inv],
        "direct_effect_estimate": [[str(v) for v in row] for row in direct_hat],
        "total_matrix_error_bound_inf": str(matrix_bound),
        "direct_matrix_error_bound_inf": str(direct_error),
        "graph_source_target": graph,
        "truth_graph_read_by_estimator": False,
        "shared_exogenous_state_assumed": True,
        "original_claim_closed": False,
    }
    expected = {
        "schema": "ncd.paired-nonlinear-development-certificate.v1",
        "status": "exact-rational-conditional-graph-certificate",
        "source_sha256": expected_sources,
        "estimator_input": inputs,
        "estimator_result": expected_result,
        "truth_only_equations": [
            "X0=U0", "X1=3/5 X0+1/10 X0^2+U1",
            "X2=2/5 X0+1/2 X1+1/20 X1^2+U2",
        ],
        "truth_only_graph_source_target": expected_graph,
        "exogenous_state": ["0"]*3,
        "conditioning": "one shared exogenous state for baseline and every do response",
        "device_rounding_certified": False,
        "independent_noise_draws_certified": False,
        "original_claim_closed": False,
    }
    if record != expected:
        raise ValueError("Certificate result or scope changed")
    return {
        "schema": "ncd.paired-nonlinear-development-verification.v1",
        "status": "independently-verified-conditional-nonlinear-graph",
        "certificate_sha256": digest(path),
        "verifier_sha256": digest(__file__),
        "curvature_bound": str(Q(1, 5)),
        "actual_x2_curvature_upper": str(x2_curvature_upper),
        "direct_error_bound_inf": str(direct_error),
        "graph_source_target": graph,
        "paired_exogenous_required": True,
        "original_claim_closed": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("--write", "--verify"):
        raise SystemExit("Usage: verifier --write|--verify")
    receipt = verify()
    if sys.argv[1] == "--write":
        if OUTPUT.exists():
            raise FileExistsError("Existing verification retained")
        OUTPUT.write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n",
                          encoding="utf-8")
    elif read(OUTPUT) != receipt:
        raise ValueError("Verification receipt changed")
    print(json.dumps(receipt, sort_keys=True))