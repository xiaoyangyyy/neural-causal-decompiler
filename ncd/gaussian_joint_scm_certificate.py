"""Exact joint-Gaussian noise coupling with conditional SCM error propagation.

U = mu + A Z and V = nu + B Z use the same vector of independent standard
Gaussians Z. Each generated law may have correlated coordinates. The coupling
and its coordinate L1 upper bounds are checked exactly over rational inputs.
Correct graph, local mechanism errors, Lipschitz bounds and common execution
domain remain separate proof obligations.
"""
from fractions import Fraction
from .graphs import topological_order


def rational(value):
    if type(value) not in (int, str, Fraction):
        raise ValueError("Certificate scalars must be exact integers or rationals")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid exact rational") from exc


def vector(values, n, name):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise ValueError(name + " has wrong dimension")
    return [rational(value) for value in values]


def matrix(values, n, name):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise ValueError(name + " has wrong row count")
    return [vector(row, n, name + " row") for row in values]


def covariance(mix):
    n = len(mix)
    return [
        [sum((mix[i][k] * mix[j][k] for k in range(n)), Fraction(0))
         for j in range(n)]
        for i in range(n)
    ]


def exact(value):
    return str(value)


def certify_gaussian_joint_scm_error(
    graph, local_errors, lipschitz,
    true_mean, true_mix, model_mean, model_mix,
    sigma_upper, interventions=None, premises=None,
):
    if not isinstance(graph, (list, tuple)) or not graph:
        raise ValueError("Nonempty DAG required")
    n = len(graph)
    if n > 8 or any(
        not isinstance(row, (list, tuple)) or len(row) != n
        or any(type(edge) not in (int, bool) or edge not in (0, 1)
               for edge in row)
        for row in graph
    ):
        raise ValueError("Expected unweighted DAG of at most eight nodes")
    graph = [[int(edge) for edge in row] for row in graph]
    order = topological_order(graph)
    eta = vector(local_errors, n, "local errors")
    lip = matrix(lipschitz, n, "Lipschitz matrix")
    mu = vector(true_mean, n, "true means")
    A = matrix(true_mix, n, "true Gaussian mixing matrix")
    nu = vector(model_mean, n, "model means")
    B = matrix(model_mix, n, "model Gaussian mixing matrix")
    upper = vector(sigma_upper, n, "Gaussian standard-deviation bounds")
    if any(value < 0 for value in eta + upper):
        raise ValueError("Error and sigma bounds must be nonnegative")
    for i in range(n):
        for j in range(n):
            if lip[i][j] < 0 or (lip[i][j] and not graph[i][j]):
                raise ValueError("Lipschitz bound outside declared DAG")
    if interventions is None:
        interventions = {}
    if not isinstance(interventions, dict):
        raise ValueError("Interventions must map node indices to exact values")
    do = {}
    for node, value in interventions.items():
        if type(node) is int:
            index = node
        elif type(node) is str and node.isdecimal() and str(int(node)) == node:
            index = int(node)
        else:
            raise ValueError("Invalid intervention target")
        if not 0 <= index < n or index in do:
            raise ValueError("Duplicate or unknown intervention target")
        do[index] = rational(value)
    if premises is None:
        premises = []
    if (not isinstance(premises, list)
            or any(type(item) is not str or not item for item in premises)):
        raise ValueError("Premise references must be nonempty strings")
    if len(set(premises)) != len(premises):
        raise ValueError("Duplicate premise reference")
    squared = [
        sum(((A[j][k] - B[j][k]) ** 2 for k in range(n)), Fraction(0))
        for j in range(n)
    ]
    if any(upper[j] ** 2 < squared[j] for j in range(n)):
        raise ValueError("Gaussian row-difference norm exceeds claimed bound")
    noise = [abs(mu[j] - nu[j]) + upper[j] for j in range(n)]
    errors = [Fraction(0) for _ in range(n)]
    for child in order:
        if child not in do:
            errors[child] = (
                eta[child] + noise[child]
                + sum((lip[parent][child] * errors[parent]
                       for parent in range(n)), Fraction(0))
            )
    cov_true = covariance(A)
    cov_model = covariance(B)
    correlated_true = any(
        cov_true[i][j] != 0 for i in range(n) for j in range(i + 1, n))
    correlated_model = any(
        cov_model[i][j] != 0 for i in range(n) for j in range(i + 1, n))
    return {
        "schema": "ncd.gaussian-joint-scm-coupling.v1",
        "status": "proved-conditionally",
        "graph": graph,
        "topological_order": order,
        "interventions": {str(j): exact(do[j]) for j in sorted(do)},
        "local_errors": list(map(exact, eta)),
        "lipschitz": [[exact(value) for value in row] for row in lip],
        "true_noise_mean": list(map(exact, mu)),
        "true_noise_mix": [[exact(value) for value in row] for row in A],
        "true_noise_covariance": [
            [exact(value) for value in row] for row in cov_true],
        "model_noise_mean": list(map(exact, nu)),
        "model_noise_mix": [[exact(value) for value in row] for row in B],
        "model_noise_covariance": [
            [exact(value) for value in row] for row in cov_model],
        "true_noise_has_cross_node_correlation": correlated_true,
        "model_noise_has_cross_node_correlation": correlated_model,
        "gaussian_sigma_upper": list(map(exact, upper)),
        "gaussian_row_difference_squared": list(map(exact, squared)),
        "verified_shared_gaussian_coupling": True,
        "noise_coordinate_l1_upper": list(map(exact, noise)),
        "noise_joint_wasserstein_l1_upper": exact(
            sum(noise, Fraction(0))),
        "coordinate_intervention_error_upper": list(map(exact, errors)),
        "joint_intervention_wasserstein_l1_upper": exact(
            sum(errors, Fraction(0))),
        "premises": premises,
        "premises_verified": False,
        "assumptions": [
            "true and model exogenous laws are the declared affine images of the same-dimensional standard Gaussian",
            "both SCMs use the same correct acyclic graph and additive unit-gain exogenous noise",
            "both SCMs apply identical observed-coordinate intervention values",
            "at every common parent state in the domain, true and model structural functions differ by at most the declared local error",
            "declared coordinatewise Lipschitz bounds hold for the model structural functions over that domain",
        ],
        "uncovered": [
            "identification of the declared true noise law from data",
            "correctness of the graph, mechanism error and Lipschitz premises",
            "coverage of the coupled execution domain",
            "other noise families and general neural-network fidelity",
            "original R0-R13 claim closure",
        ],
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify_gaussian_joint_scm_error(certificate):
    if (not isinstance(certificate, dict)
            or certificate.get("schema") != "ncd.gaussian-joint-scm-coupling.v1"):
        raise ValueError("Wrong Gaussian joint SCM certificate")
    expected = certify_gaussian_joint_scm_error(
        certificate["graph"],
        certificate["local_errors"],
        certificate["lipschitz"],
        certificate["true_noise_mean"],
        certificate["true_noise_mix"],
        certificate["model_noise_mean"],
        certificate["model_noise_mix"],
        certificate["gaussian_sigma_upper"],
        certificate["interventions"],
        certificate["premises"],
    )
    if certificate != expected:
        raise ValueError("Gaussian joint SCM certificate mismatch")
    return {
        "status": "verified-conditionally",
        "verified_shared_gaussian_coupling": True,
        "premises_verified": False,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }
