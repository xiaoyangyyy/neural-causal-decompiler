from copy import deepcopy
from fractions import Fraction as Q
import pytest
from finite_graph_chernoff.boundary import certify, verify, power_compare


def test_original_96_rows_now_satisfy_single_event_recovery_gate():
    c = certify()
    r = verify(c)
    assert r["requested_recovery_success_probability_gate"] == "proved"
    assert r["old_second_moment_gate"] == "unresolved-by-this-bound"
    assert r["uniform_error_upper"]["rational_power_base"] == "8/9"
    assert r["uniform_error_upper"]["integer_exponent"] == 48
    assert Q(r["uniform_error_upper"]["short_exact_fraction"]) < Q(1, 100)
    assert r["recovery_estimator_unchanged"] and not r["original_objective_achieved"]


def test_six_events_need_the_declared_correction():
    assert verify(certify(family_size=6))["requested_recovery_success_probability_gate"] == "unresolved-by-this-bound"
    assert verify(certify(samples=110, family_size=6))["requested_recovery_success_probability_gate"] == "proved"
    assert verify(certify(samples=108, family_size=6))["requested_recovery_success_probability_gate"] == "unresolved-by-this-bound"


def test_odd_rows_drop_only_the_beneficial_half_power():
    c = certify(samples=97)
    assert c["uniform_error_upper"]["odd_half_power_dropped_conservatively"]
    assert c["uniform_error_upper"]["integer_exponent"] == 48
    assert verify(c)["requested_recovery_success_probability_gate"] == "proved"
    assert verify(certify(samples=1))["uniform_error_upper"]["short_exact_fraction"] == "1"


def test_known_negative_coefficient_changes_no_probability_bound():
    a, b = certify(), certify(a="-1/2")
    assert a["uniform_error_upper"] == b["uniform_error_upper"]
    assert verify(b)["requested_recovery_success_probability_gate"] == "proved"


def test_exact_compare_includes_equality_without_float_log():
    assert power_compare(Q(2, 3), 3, Q(8, 27))
    assert not power_compare(Q(2, 3), 3, Q(8, 27) - Q(1, 10**30))


def test_large_arithmetic_budget_retains_unresolved_gate():
    c = certify(a="1/2", samples=1000000)
    r = verify(c)
    assert not c["comparison_supported"]
    assert r["requested_recovery_success_probability_gate"] == "unresolved-arithmetic-budget"
    assert c["arithmetic_budget_exhaustion_does_not_refute_recovery"]


@pytest.mark.parametrize("change", ["tilt", "variance", "power", "gate", "noise", "scope"])
def test_mgf_premises_tail_and_scope_cannot_be_forged(change):
    c = certify()
    if change == "tilt":
        c["tilt"] = "1/2"
    elif change == "variance":
        c["gaussian_mgf_denominator"] = "5/4"
    elif change == "power":
        c["uniform_error_upper"]["integer_exponent"] = 96
    elif change == "gate":
        c = certify(family_size=6)
        c["requested_recovery_success_probability_gate"] = "proved"
    elif change == "noise":
        c["base_certificate"]["noise_law"] = "Correlated exogenous noise"
    else:
        c["original_claim_closed"] = True
    with pytest.raises(ValueError):
        verify(c)
