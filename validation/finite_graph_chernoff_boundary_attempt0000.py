from fractions import Fraction as Q
import hashlib, json
from finite_graph_proof.boundary import certify as base_certify, verify as base_verify

POWER_BIT_BUDGET = 262144


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def power_compare(base, exponent, threshold):
    # Independent repeated-squaring comparison; no float or logarithm in the decision.
    numerator, denominator = 1, 1
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


def certify(a="1/2", samples=96, delta="1/100", family_size=1):
    previous = base_certify(a, samples, delta, family_size)
    base_verify(previous)
    a = Q(previous["a"])
    t = abs(a) / 2
    gaussian_square_coefficient = t * t - abs(a) * t
    denominator = 1 - 2 * gaussian_square_coefficient
    power_base = 1 / denominator
    exponent = samples // 2
    bit_estimate = exponent * max(power_base.numerator.bit_length(), power_base.denominator.bit_length())
    threshold = Q(previous["delta"]) / family_size
    supported = bit_estimate <= POWER_BIT_BUDGET
    if supported:
        exact_upper = power_base ** exponent
        gate = "proved" if exact_upper <= threshold else "unresolved-by-this-bound"
        short_exact = str(exact_upper) if max(exact_upper.numerator.bit_length(), exact_upper.denominator.bit_length()) <= 4096 else None
    else:
        gate, short_exact = "unresolved-arithmetic-budget", None
    return {
        "schema": "ncd.finite-gaussian-chernoff.v1",
        "status": "analytic-bound-proved-in-stated-two-model-family",
        "a": str(a), "samples": samples, "delta": previous["delta"], "family_size": family_size,
        "base_certificate": previous,
        "base_certificate_sha256": canonical_digest(previous),
        "statistic_unchanged": previous["estimator"]["statistic"],
        "signed_row_law": "For each model reflect the signed error event so W=abs(a)*X0^2+sqrt(2)*X0*Z, with X0,Z independent N(0,1)",
        "tilt": str(t),
        "conditional_gaussian_exponent": str(gaussian_square_coefficient),
        "gaussian_mgf_denominator": str(denominator),
        "mgf_identity": "E exp(-t*W) = E exp((t*t-abs(a)*t)*X0^2) = (1+2*abs(a)*t-2*t*t)^(-1/2)",
        "mgf_condition": "The denominator is positive; here 1+a*a/2>1. Completing the Gaussian square proves the displayed identity.",
        "markov_derivation": "The error event is contained in sum_i W_i<=0; exp(-t*sum_i W_i)>=1 there. Markov plus IID rows gives error <=(1+a*a/2)^(-samples/2).",
        "uniform_error_upper": {"rational_power_base": str(power_base), "integer_exponent": exponent, "odd_half_power_dropped_conservatively": bool(samples % 2), "short_exact_fraction": short_exact},
        "exact_half_power_bound": "(1+a*a/2)^(-samples/2)",
        "per_event_error_threshold": str(threshold),
        "requested_recovery_success_probability_gate": gate,
        "comparison_supported": supported,
        "power_bit_estimate": bit_estimate,
        "power_bit_budget": POWER_BIT_BUDGET,
        "old_second_moment_gate": previous["estimator"]["requested_confidence_gate"],
        "union_bound": "For K fixed recovery events each satisfying these premises, per-event failure <=delta/K implies family failure <=delta; events need not be mutually independent.",
        "probability_semantics": "Recovery probability over ideal IID observation rows for a fixed estimator and known two-model family; not a confidence interval for trained-network accuracy across evaluation worlds.",
        "not_proved": previous["not_proved"] + ["A 99 percent guarantee for the actual frozen neural graph networks"],
        "base_failure_artifacts_not_rewritten": True,
        "arithmetic_budget_exhaustion_does_not_refute_recovery": True,
        "original_claim_closed": False,
        "original_objective_achieved": False,
    }


def verify(certificate):
    if certificate.get("schema") != "ncd.finite-gaussian-chernoff.v1":
        raise ValueError("Wrong Chernoff certificate schema")
    previous = certificate["base_certificate"]
    base_verify(previous)
    if canonical_digest(previous) != certificate["base_certificate_sha256"]:
        raise ValueError("Base theorem changed")
    for field in ("a", "samples", "delta", "family_size"):
        if certificate[field] != previous[field]:
            raise ValueError("Frozen estimator parameter changed")
    a = Q(certificate["a"])
    t = Q(certificate["tilt"])
    # Integrate the conditional Gaussian noise first, then complete the square in X0.
    conditional = t * t - abs(a) * t
    denominator = 1 - 2 * conditional
    if t != abs(a) / 2 or denominator <= 1 or denominator != 1 + a * a / 2:
        raise ValueError("Invalid Gaussian exponential moment")
    if str(conditional) != certificate["conditional_gaussian_exponent"] or str(denominator) != certificate["gaussian_mgf_denominator"]:
        raise ValueError("MGF coefficient mismatch")
    power = certificate["uniform_error_upper"]
    base, exponent = 1 / denominator, certificate["samples"] // 2
    if str(base) != power["rational_power_base"] or exponent != power["integer_exponent"]:
        raise ValueError("Invalid tail power")
    estimate = exponent * max(base.numerator.bit_length(), base.denominator.bit_length())
    threshold = Q(certificate["delta"]) / certificate["family_size"]
    if estimate <= POWER_BIT_BUDGET:
        supported_gate = "proved" if power_compare(base, exponent, threshold) else "unresolved-by-this-bound"
    else:
        supported_gate = "unresolved-arithmetic-budget"
    if certificate["requested_recovery_success_probability_gate"] != supported_gate:
        raise ValueError("Recovery gate does not follow from exact comparison")
    if certificate != certify(previous["a"], previous["samples"], previous["delta"], previous["family_size"]):
        raise ValueError("Theorem, estimator, noise premise or scope changed")
    return {"status": "verified", "conclusion": "proved-in-declared-family", "requested_recovery_success_probability_gate": supported_gate, "uniform_error_upper": power, "old_second_moment_gate": certificate["old_second_moment_gate"], "recovery_estimator_unchanged": True, "comparison_without_float_or_log": True, "original_objective_achieved": False}
