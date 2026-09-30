"""Independent exact-arithmetic replay of the Gaussian joint-SCM example."""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "validation/gaussian_joint_scm_certificate_v1.json"
RECEIPT = ROOT / "validation/gaussian_joint_scm_verification_v1.json"
SOURCE = ROOT / "ncd/gaussian_joint_scm_certificate.py"
ASSUMPTIONS = [
    "true and model exogenous laws are the declared affine images of the same-dimensional standard Gaussian",
    "both SCMs use the same correct acyclic graph and additive unit-gain exogenous noise",
    "both SCMs apply identical observed-coordinate intervention values",
    "at every common parent state in the domain, true and model structural functions differ by at most the declared local error",
    "declared coordinatewise Lipschitz bounds hold for the model structural functions over that domain",
]
UNCOVERED = [
    "identification of the declared true noise law from data",
    "correctness of the graph, mechanism error and Lipschitz premises",
    "coverage of the coupled execution domain",
    "other noise families and general neural-network fidelity",
    "original R0-R13 claim closure",
]
EXPECTED_FIELDS = {
    "schema", "status", "graph", "topological_order", "interventions",
    "local_errors", "lipschitz", "true_noise_mean", "true_noise_mix",
    "true_noise_covariance", "model_noise_mean", "model_noise_mix",
    "model_noise_covariance", "true_noise_has_cross_node_correlation",
    "model_noise_has_cross_node_correlation", "gaussian_sigma_upper",
    "gaussian_row_difference_squared", "verified_shared_gaussian_coupling",
    "noise_coordinate_l1_upper", "noise_joint_wasserstein_l1_upper",
    "coordinate_intervention_error_upper",
    "joint_intervention_wasserstein_l1_upper", "premises",
    "premises_verified", "assumptions", "uncovered",
    "original_claim_closed", "original_objective_achieved",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def fraction(value):
    if type(value) is not str:
        raise ValueError("Certificate arithmetic must be exact strings")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid rational in certificate") from exc
    if str(parsed) != value:
        raise ValueError("Noncanonical rational in certificate")
    return parsed


def vector(values, n):
    if type(values) is not list or len(values) != n:
        raise ValueError("Certificate vector dimension")
    return [fraction(value) for value in values]


def matrix(values, n):
    if type(values) is not list or len(values) != n:
        raise ValueError("Certificate matrix dimension")
    return [vector(row, n) for row in values]


def sorted_dag(graph):
    n = len(graph)
    degree = [sum(graph[parent][child] for parent in range(n))
              for child in range(n)]
    ready = [node for node in range(n) if degree[node] == 0]
    order = []
    while ready:
        node = min(ready)
        ready.remove(node)
        order.append(node)
        for child in range(n):
            if graph[node][child]:
                degree[child] -= 1
                if degree[child] == 0:
                    ready.append(child)
    if len(order) != n:
        raise ValueError("Certificate graph contains a cycle")
    return order


def audit(certificate_path=CERTIFICATE):
    certificate_path = Path(certificate_path)
    cert = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (type(cert) is not dict or set(cert) != EXPECTED_FIELDS
            or cert["schema"] != "ncd.gaussian-joint-scm-coupling.v1"
            or cert["status"] != "proved-conditionally"
            or cert["verified_shared_gaussian_coupling"] is not True
            or cert["premises_verified"] is not False
            or cert["original_claim_closed"] is not False
            or cert["original_objective_achieved"] is not False):
        raise ValueError("Changed Gaussian coupling certificate scope")
    graph = cert["graph"]
    if (type(graph) is not list or not 1 <= len(graph) <= 8
            or any(type(row) is not list or len(row) != len(graph)
                   or any(type(edge) is not int or edge not in (0, 1)
                          for edge in row)
                   for row in graph)):
        raise ValueError("Wrong certificate DAG")
    n = len(graph)
    order = sorted_dag(graph)
    if cert["topological_order"] != order:
        raise ValueError("Wrong topological order")
    eta = vector(cert["local_errors"], n)
    L = matrix(cert["lipschitz"], n)
    mu = vector(cert["true_noise_mean"], n)
    A = matrix(cert["true_noise_mix"], n)
    nu = vector(cert["model_noise_mean"], n)
    B = matrix(cert["model_noise_mix"], n)
    sigma = vector(cert["gaussian_sigma_upper"], n)
    if any(value < 0 for value in eta + sigma):
        raise ValueError("Negative error or Gaussian bound")
    for i in range(n):
        for j in range(n):
            if L[i][j] < 0 or (L[i][j] and not graph[i][j]):
                raise ValueError("Wrong Lipschitz support")
    do = {}
    for node, value in cert["interventions"].items():
        if (type(node) is not str or not node.isdecimal()
                or str(int(node)) != node):
            raise ValueError("Wrong intervention node")
        index = int(node)
        if not 0 <= index < n:
            raise ValueError("Unknown intervention node")
        do[index] = fraction(value)
    if type(cert["premises"]) is not list or any(
        type(item) is not str or not item for item in cert["premises"]
    ) or len(set(cert["premises"])) != len(cert["premises"]):
        raise ValueError("Invalid premise references")
    if cert["assumptions"] != ASSUMPTIONS:
        raise ValueError("Changed theorem assumptions")
    if cert["uncovered"] != UNCOVERED:
        raise ValueError("Changed unresolved boundaries")
    true_cov = [
        [sum((A[i][k] * A[j][k] for k in range(n)), Fraction(0))
         for j in range(n)] for i in range(n)
    ]
    model_cov = [
        [sum((B[i][k] * B[j][k] for k in range(n)), Fraction(0))
         for j in range(n)] for i in range(n)
    ]
    if (cert["true_noise_covariance"]
            != [[str(value) for value in row] for row in true_cov]
            or cert["model_noise_covariance"]
            != [[str(value) for value in row] for row in model_cov]):
        raise ValueError("Gaussian covariance mismatch")
    true_correlated = any(
        true_cov[i][j] != 0 for i in range(n) for j in range(i + 1, n))
    model_correlated = any(
        model_cov[i][j] != 0 for i in range(n) for j in range(i + 1, n))
    if (cert["true_noise_has_cross_node_correlation"] is not true_correlated
            or cert["model_noise_has_cross_node_correlation"] is not model_correlated):
        raise ValueError("Gaussian correlation claim mismatch")
    squared = [
        sum(((A[j][k] - B[j][k]) ** 2 for k in range(n)), Fraction(0))
        for j in range(n)
    ]
    if (cert["gaussian_row_difference_squared"] != list(map(str, squared))
            or any(sigma[j] ** 2 < squared[j] for j in range(n))):
        raise ValueError("Invalid shared-Gaussian coupling bound")
    noise = [abs(mu[j] - nu[j]) + sigma[j] for j in range(n)]
    if (cert["noise_coordinate_l1_upper"] != list(map(str, noise))
            or cert["noise_joint_wasserstein_l1_upper"]
            != str(sum(noise, Fraction(0)))):
        raise ValueError("Noise transport bound mismatch")
    error = [Fraction(0) for _ in range(n)]
    for child in order:
        if child not in do:
            error[child] = (
                eta[child] + noise[child]
                + sum((L[parent][child] * error[parent]
                       for parent in range(n)), Fraction(0))
            )
    if (cert["coordinate_intervention_error_upper"] != list(map(str, error))
            or cert["joint_intervention_wasserstein_l1_upper"]
            != str(sum(error, Fraction(0)))):
        raise ValueError("Intervention distribution bound mismatch")
    return {
        "schema": "ncd.gaussian-joint-scm-verification.v1",
        "status": "verified-conditionally",
        "certificate_sha256": digest(certificate_path),
        "producer_source_sha256": digest(SOURCE),
        "independent_verifier_sha256": digest(__file__),
        "true_noise_correlated": true_correlated,
        "verified_joint_noise_coupling": True,
        "premises_verified": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def replay_receipt():
    observed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = audit()
    if observed != expected:
        raise ValueError("Gaussian coupling verification receipt mismatch")
    return expected


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        if RECEIPT.exists():
            raise FileExistsError("Retain prior exact coupling receipt")
        result = audit()
        RECEIPT.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
    elif len(sys.argv) == 1:
        result = replay_receipt()
    else:
        raise SystemExit("Usage: script [--write]")
    print(json.dumps(result, sort_keys=True))
