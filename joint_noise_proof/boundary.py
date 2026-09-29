from fractions import Fraction as Q


def determinant3(matrix):
    a, b, c = matrix
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def certify():
    return {
        "schema": "ncd.correlated-noise-marginal-boundary.v1",
        "status": "refuted-in-stated-model-pair",
        "claim_refuted": "For all explicit SCM pairs, identical graph, equations and exogenous marginal laws imply identical joint intervention distributions without a joint-noise premise.",
        "reference": {
            "graph": [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            "equations": ["X0=U0", "X1=X0+U1", "X2=X0+U2"],
            "noise": "centered joint Gaussian, independent coordinates",
            "noise_covariance": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
        },
        "realization": {
            "graph": [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            "equations": ["X0=U0", "X1=X0+U1", "X2=X0+U2"],
            "noise": "centered joint Gaussian, correlation(U1,U2)=1/2",
            "noise_covariance": [["1", "0", "0"], ["0", "1", "1/2"], ["0", "1/2", "1"]],
        },
        "positive_definite_leading_minors": [["1", "1", "1"], ["1", "1", "3/4"]],
        "exogenous_marginal_laws": ["N(0,1)"] * 3,
        "each_exogenous_marginal_w1": ["0"] * 3,
        "intervention": {"targets": [0], "values": ["0"]},
        "intervened_marginal_laws": ["point_mass_0", "N(0,1)", "N(0,1)"],
        "each_output_marginal_w1": ["0"] * 3,
        "local_mechanism_errors": ["0"] * 3,
        "difference_variances": ["2", "1"],
        "dual_witness": "h(x)=abs(x1-x2)",
        "dual_lipschitz_constant_l1": "1",
        "reference_witness_expectation": "2/sqrt(pi)",
        "realization_witness_expectation": "sqrt(2/pi)",
        "expectation_gap": "sqrt(2/pi)*(sqrt(2)-1)",
        "rational_bounds": {"pi_upper": "4", "sqrt2_lower": "7/5", "sqrt_half_lower": "7/10", "gap_factor_lower": "2/5", "joint_w1_l1_lower": "7/25"},
        "lipschitz_derivation": "abs(abs(x1-x2)-abs(y1-y2)) <= abs(x1-y1)+abs(x2-y2) <= sum_i abs(xi-yi); every joint coupling therefore has expected l1 cost at least the expectation gap.",
        "moment_derivation": "For centered Gaussian D of variance v, E abs(D)=sqrt(2*v/pi), obtained by integrating x*exp(-x*x/(2*v)) on the positive half-line.",
        "rational_derivation": "pi<4 from pi/4=integral_0^1 1/(1+x*x) dx<1; (7/5)^2<2 and (7/10)^2<1/2. Thus the positive expectation gap is strictly greater than (7/10)*(2/5)=7/25.",
        "conclusion": "joint W1 under l1 cost is strictly greater than 7/25 despite every coordinate marginal W1 and local mechanism error being zero",
        "not_refuted": ["Bounds with a certified joint-noise coupling", "The declared independent-Gaussian product-noise propagation theorem", "End-to-end recovery under separately validated noise assumptions"],
        "actual_learned_residuals_certified": False,
        "classical_probability_facts_not_project_innovation": True,
        "original_objective_achieved": False,
        "original_claim_closed": False,
    }


def verify(certificate):
    if certificate.get("schema") != "ncd.correlated-noise-marginal-boundary.v1":
        raise ValueError("Wrong counterexample schema")
    moments, minors = [], []
    for model in (certificate["reference"], certificate["realization"]):
        matrix = [[Q(v) for v in row] for row in model["noise_covariance"]]
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("Covariance dimension")
        if any(matrix[i][j] != matrix[j][i] for i in range(3) for j in range(3)):
            raise ValueError("Non-symmetric covariance")
        leading = [matrix[0][0], matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2, determinant3(matrix)]
        if any(v <= 0 for v in leading):
            raise ValueError("Joint Gaussian not positive definite")
        if any(matrix[i][i] != 1 for i in range(3)):
            raise ValueError("Marginal law mismatch")
        moments.append(matrix[1][1] + matrix[2][2] - 2 * matrix[1][2])
        minors.append(list(map(str, leading)))
    if moments != [Q(2), Q(1)] or list(map(str, moments)) != certificate["difference_variances"] or minors != certificate["positive_definite_leading_minors"]:
        raise ValueError("Intervention variance proof mismatch")
    bounds = {key: Q(value) for key, value in certificate["rational_bounds"].items()}
    if bounds["sqrt2_lower"] ** 2 >= 2 or bounds["sqrt_half_lower"] ** 2 >= Q(1, 2):
        raise ValueError("Invalid radical enclosure")
    lower = bounds["sqrt_half_lower"] * (bounds["sqrt2_lower"] - 1)
    if lower != bounds["joint_w1_l1_lower"] or lower <= 0:
        raise ValueError("Invalid Wasserstein lower bound")
    if certificate != certify():
        raise ValueError("Counterexample scope, equations, noise law or proof changed")
    return {"status": "verified", "conclusion": "refuted", "joint_w1_l1_strict_lower": str(lower), "coordinate_marginal_w1": ["0"] * 3, "graph_and_equations_equal": True, "both_joint_gaussians_full_support": True, "independent_noise_propagation_refuted": False, "actual_learned_residuals_certified": False, "original_objective_achieved": False}
