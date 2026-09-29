from fractions import Fraction as Q
from math import isqrt
import hashlib, json
from finite_graph_proof.boundary import certify as base_certify, verify as base_verify

POWER_BIT_BUDGET = 262144


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def power_compare(base, exponent, threshold, multiplier=Q(1)):
    # An independent repeated-squaring route, with no float or log in any gate.
    numerator, denominator = multiplier.numerator, multiplier.denominator
    nn, dd, remaining = base.numerator, base.denominator, exponent
    while remaining:
        if remaining & 1:
            numerator *= nn
            denominator *= dd
        remaining //= 2
        if remaining:
            nn *= nn
            dd *= dd
    return numerator * threshold.denominator <= denominator * threshold.numerator


def power_bound(base, exponent, multiplier, threshold):
    estimate = exponent * max(base.numerator.bit_length(), base.denominator.bit_length()) + max(multiplier.numerator.bit_length(), multiplier.denominator.bit_length())
    supported = estimate <= POWER_BIT_BUDGET
    upper = min(Q(1), multiplier * base ** exponent) if supported else None
    return {
        "rational_power_base": str(base), "integer_exponent": exponent,
        "multiplier": str(multiplier), "clamped_by_probability_one": True,
        "power_bit_estimate": estimate, "comparison_supported": supported,
        "short_exact_fraction": str(upper) if upper is not None and max(upper.numerator.bit_length(), upper.denominator.bit_length()) <= 4096 else None,
        "gate": "proved" if upper is not None and upper <= threshold else "unresolved-by-this-bound" if supported else "unresolved-arithmetic-budget",
    }, upper


def certify(a="1/2", samples=96, delta="1/100", family_size=1):
    previous = base_certify(a, samples, delta, family_size)
    base_verify(previous)
    a = Q(previous["a"])
    t = abs(a) / 2
    conditional = t * t - abs(a) * t
    denominator = 1 - 2 * conditional
    threshold = Q(previous["delta"]) / family_size
    chernoff, upper_c = power_bound(1 / denominator, samples // 2, Q(1), threshold)
    chernoff["odd_half_power_dropped_conservatively"] = bool(samples % 2)
    pi_lower = 4 * sum((Q((-1) ** k, 2 * k + 1) for k in range(8)), Q(0))
    if samples > 2:
        square_root_floor = isqrt(3 * (samples - 2))
        multiplier = 1 / (abs(a) * square_root_floor)
        mills, upper_m = power_bound(1 / (1 + a * a), samples // 4, multiplier, threshold)
        mills["quarter_power_dropped_conservatively"] = samples % 4
        mills["sqrt_3n_minus6_integer_lower"] = square_root_floor
    else:
        mills, upper_m = power_bound(Q(1), 0, Q(1), threshold)
        mills["not_used_reason"] = "Inverse chi-square moment requires samples>2; probability-one bound retained"
    supported = [("chernoff", upper_c), ("mills_cauchy_schwarz", upper_m)]
    supported = [(name, value) for name, value in supported if value is not None]
    choice = min(supported, key=lambda item: (item[1], item[0])) if supported else None
    gate = "proved" if choice is not None and choice[1] <= threshold else "unresolved-by-this-bound" if choice is not None else "unresolved-arithmetic-budget"
    return {
        "schema": "ncd.finite-gaussian-chernoff.v1",
        "status": "analytic-bounds-proved-in-stated-two-model-family",
        "a": str(a), "samples": samples, "delta": previous["delta"], "family_size": family_size,
        "base_certificate": previous, "base_certificate_sha256": canonical_digest(previous),
        "statistic_unchanged": previous["estimator"]["statistic"],
        "signed_row_law": "For each model reflect the signed error event so W=abs(a)*X0^2+sqrt(2)*X0*Z, with X0,Z independent N(0,1)",
        "tilt": str(t), "conditional_gaussian_exponent": str(conditional), "gaussian_mgf_denominator": str(denominator),
        "mgf_identity": "E exp(-t*W) = E exp((t*t-abs(a)*t)*X0^2) = (1+2*abs(a)*t-2*t*t)^(-1/2)",
        "mgf_condition": "The denominator is positive; here 1+a*a/2>1. Completing the Gaussian square proves the identity.",
        "markov_derivation": "The error event is contained in sum W_i<=0. Markov on exp(-t*sum W_i), using IID rows, gives error <=(1+a*a/2)^(-samples/2).",
        "uniform_error_upper": chernoff,
        "exact_half_power_bound": "(1+a*a/2)^(-samples/2)",
        "mills_cauchy_schwarz_upper": mills,
        "conditional_error": "With S=sum X0_i^2~chi_square(samples), error conditional on S is Phi(-abs(a)*sqrt(S/2)); the Gaussian noise has conditional variance 2*S.",
        "mills_derivation": "For x>0, Phi(-x)<=exp(-x*x/2)/(x*sqrt(2*pi)) by integrating t*exp(-t*t/2) and t/x>=1. Cauchy-Schwarz gives error<=1/(abs(a)*sqrt(pi*(samples-2)))*(1+a*a)^(-samples/4), for samples>2.",
        "inverse_chisquare_moment": "E(1/S)=1/(samples-2), obtained from the chi-square density by the Gamma recurrence; E exp(-a*a*S/2)=(1+a*a)^(-samples/2).",
        "pi_lower_rational": str(pi_lower),
        "pi_lower_proof": "pi/4=integral_0^1 1/(1+x*x) dx; sum_{k=0}^7 (-x*x)^k is a lower polynomial, with nonnegative remainder x^16/(1+x*x). Its exact integral times four is >3.",
        "best_supported_bound": choice[0] if choice is not None else None,
        "best_short_exact_fraction": str(choice[1]) if choice is not None and max(choice[1].numerator.bit_length(), choice[1].denominator.bit_length()) <= 4096 else None,
        "per_event_error_threshold": str(threshold),
        "requested_recovery_success_probability_gate": gate,
        "power_bit_budget": POWER_BIT_BUDGET,
        "old_second_moment_gate": previous["estimator"]["requested_confidence_gate"],
        "union_bound": "For K fixed recovery events satisfying these premises, per-event failure<=delta/K implies family failure<=delta; events need not be mutually independent.",
        "probability_semantics": "Recovery probability over ideal IID observation rows for a fixed estimator and known two-model family; not a confidence interval for trained-network accuracy across evaluation worlds.",
        "not_proved": previous["not_proved"] + ["A 99 percent guarantee for the actual frozen neural graph networks"],
        "base_failure_artifacts_not_rewritten": True,
        "arithmetic_budget_exhaustion_does_not_refute_recovery": True,
        "original_claim_closed": False, "original_objective_achieved": False,
    }


def verify(c):
    if c.get("schema") != "ncd.finite-gaussian-chernoff.v1":
        raise ValueError("Wrong Gaussian-tail certificate schema")
    previous = c["base_certificate"]
    base_verify(previous)
    if canonical_digest(previous) != c["base_certificate_sha256"]:
        raise ValueError("Base theorem changed")
    for field in ("a", "samples", "delta", "family_size"):
        if c[field] != previous[field]:
            raise ValueError("Frozen estimator parameter changed")
    a, n, t = Q(c["a"]), c["samples"], Q(c["tilt"])
    # Independent covariance route verifies the residual Gaussian variance and independence.
    for model in previous["models"]:
        covariance = [[Q(v) for v in row] for row in model["covariance"]]
        mu = covariance[0][1] - covariance[0][2]
        var = covariance[1][1] + covariance[2][2] - 2 * covariance[1][2]
        if covariance[0][0] != 1 or abs(mu) != abs(a) or var - mu * mu != 2:
            raise ValueError("Conditional Gaussian variance premise changed")
    conditional = t * t - abs(a) * t
    denominator = 1 - 2 * conditional
    if t != abs(a) / 2 or denominator <= 1 or denominator != 1 + a * a / 2:
        raise ValueError("Invalid Gaussian exponential moment")
    if str(conditional) != c["conditional_gaussian_exponent"] or str(denominator) != c["gaussian_mgf_denominator"]:
        raise ValueError("MGF coefficient mismatch")
    pi_lower = sum((Q(4 * (-1) ** k, 2 * k + 1) for k in range(8)), Q(0))
    if pi_lower <= 3 or str(pi_lower) != c["pi_lower_rational"]:
        raise ValueError("Invalid pi enclosure")
    threshold = Q(c["delta"]) / c["family_size"]
    decisions = []
    for name, expected_base, expected_power, multiplier in (
        ("uniform_error_upper", 1 / denominator, n // 2, Q(1)),
        ("mills_cauchy_schwarz_upper", 1 / (1 + a * a) if n > 2 else Q(1), n // 4 if n > 2 else 0, 1 / (abs(a) * isqrt(3 * (n - 2))) if n > 2 else Q(1))):
        bound = c[name]
        if Q(bound["rational_power_base"]) != expected_base or bound["integer_exponent"] != expected_power or Q(bound["multiplier"]) != multiplier:
            raise ValueError("Invalid tail power or prefactor")
        if n > 2 and name == "mills_cauchy_schwarz_upper":
            integer = bound["sqrt_3n_minus6_integer_lower"]
            if integer * integer > 3 * (n - 2) or (integer + 1) ** 2 <= 3 * (n - 2):
                raise ValueError("Invalid radical enclosure")
        estimate = expected_power * max(expected_base.numerator.bit_length(), expected_base.denominator.bit_length()) + max(multiplier.numerator.bit_length(), multiplier.denominator.bit_length())
        if bound["comparison_supported"] != (estimate <= POWER_BIT_BUDGET):
            raise ValueError("Hidden arithmetic budget exhaustion")
        if estimate <= POWER_BIT_BUDGET:
            decisions.append(threshold >= 1 or power_compare(expected_base, expected_power, threshold, multiplier))
    gate = "proved" if any(decisions) else "unresolved-by-this-bound" if decisions else "unresolved-arithmetic-budget"
    if c["requested_recovery_success_probability_gate"] != gate:
        raise ValueError("Recovery gate does not follow from exact comparison")
    if c != certify(previous["a"], n, previous["delta"], previous["family_size"]):
        raise ValueError("Theorem, estimator, noise premise or scope changed")
    return {"status": "verified", "conclusion": "proved-in-declared-family", "requested_recovery_success_probability_gate": gate, "uniform_error_upper": c["uniform_error_upper"], "mills_cauchy_schwarz_upper": c["mills_cauchy_schwarz_upper"], "best_supported_bound": c["best_supported_bound"], "best_short_exact_fraction": c["best_short_exact_fraction"], "old_second_moment_gate": c["old_second_moment_gate"], "recovery_estimator_unchanged": True, "comparison_without_float_or_log": True, "original_objective_achieved": False}
