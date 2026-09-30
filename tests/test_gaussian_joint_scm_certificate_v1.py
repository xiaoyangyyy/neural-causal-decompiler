"""Exact correlated-Gaussian coupling and conditional SCM propagation."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ncd.gaussian_joint_scm_certificate import (
    certify_gaussian_joint_scm_error, verify_gaussian_joint_scm_error,
)


def certificate(interventions=None):
    return certify_gaussian_joint_scm_error(
        graph=[[0, 1], [0, 0]],
        local_errors=["1/10", "1/20"],
        lipschitz=[[0, 2], [0, 0]],
        true_mean=[0, 0],
        true_mix=[[1, 0], [1, 1]],
        model_mean=[0, 0],
        model_mix=[[1, 0], [0, 1]],
        sigma_upper=[0, 1],
        interventions=interventions,
        premises=["correct_graph", "local_mechanism", "domain_and_lipschitz"],
    )


def test_correlated_true_gaussian_coupling_is_exactly_checked():
    result = certificate()
    assert result["true_noise_covariance"] == [["1", "1"], ["1", "2"]]
    assert result["model_noise_covariance"] == [["1", "0"], ["0", "1"]]
    assert result["true_noise_has_cross_node_correlation"] is True
    assert result["model_noise_has_cross_node_correlation"] is False
    assert result["gaussian_row_difference_squared"] == ["0", "1"]
    assert result["noise_coordinate_l1_upper"] == ["0", "1"]
    assert result["coordinate_intervention_error_upper"] == ["1/10", "5/4"]
    assert result["joint_intervention_wasserstein_l1_upper"] == "27/20"
    assert verify_gaussian_joint_scm_error(result)["premises_verified"] is False


def test_intervention_cuts_mechanism_noise_and_parent_error():
    do_root = certificate({0: 2})
    assert do_root["coordinate_intervention_error_upper"] == ["0", "21/20"]
    assert do_root["joint_intervention_wasserstein_l1_upper"] == "21/20"
    do_child = certificate({"1": "-2"})
    assert do_child["coordinate_intervention_error_upper"] == ["1/10", "0"]
    assert do_child["joint_intervention_wasserstein_l1_upper"] == "1/10"
    both = certificate({0: 2, 1: -2})
    assert both["joint_intervention_wasserstein_l1_upper"] == "0"


def test_same_correlated_joint_law_has_zero_noise_coupling_cost():
    result = certify_gaussian_joint_scm_error(
        [[0, 1], [0, 0]], [0, 0], [[0, 0], [0, 0]],
        [0, 0], [[1, 0], [1, 1]],
        [0, 0], [[1, 0], [1, 1]], [0, 0],
    )
    assert result["true_noise_has_cross_node_correlation"] is True
    assert result["noise_joint_wasserstein_l1_upper"] == "0"
    assert result["joint_intervention_wasserstein_l1_upper"] == "0"
    assert verify_gaussian_joint_scm_error(result)["status"] == "verified-conditionally"


@pytest.mark.parametrize("field,value", [
    ("true_noise_covariance", [["1", "0"], ["0", "2"]]),
    ("noise_coordinate_l1_upper", ["0", "0"]),
    ("joint_intervention_wasserstein_l1_upper", "1"),
    ("verified_shared_gaussian_coupling", False),
    ("premises_verified", True),
    ("original_claim_closed", True),
])
def test_rehashed_or_edited_scientific_conclusion_rejected(field, value):
    altered = copy.deepcopy(certificate())
    altered[field] = value
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify_gaussian_joint_scm_error(altered)


def test_unsupported_or_underbounded_contract_rejected():
    with pytest.raises(ValueError, match="row-difference norm"):
        certify_gaussian_joint_scm_error(
            [[0, 1], [0, 0]], [0, 0], [[0, 0], [0, 0]],
            [0, 0], [[1, 0], [1, 1]],
            [0, 0], [[1, 0], [0, 1]], [0, "1/2"],
        )
    with pytest.raises(ValueError, match="exact integers or rationals"):
        certify_gaussian_joint_scm_error(
            [[0]], [0.01], [[0]], [0], [[1]], [0], [[1]], [0],
        )
    with pytest.raises(ValueError, match="outside declared DAG"):
        certify_gaussian_joint_scm_error(
            [[0, 0], [0, 0]], [0, 0], [[0, 1], [0, 0]],
            [0, 0], [[1, 0], [0, 1]],
            [0, 0], [[1, 0], [0, 1]], [0, 0],
        )
