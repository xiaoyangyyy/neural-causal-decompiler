"""Posthoc role-hybrid replay and scope-tamper checks."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from evaluate_interventional_mechanism_hybrid_dev_v1 import (
    OUTPUT, choose_roles, verify as primary_verify,
)
from check_interventional_mechanism_hybrid_dev_v1 import (
    OUTPUT as RECEIPT, verify as independent_verify,
)


def test_role_rule_uses_only_inferred_parent_counts():
    assert choose_roles([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) == [
        "control", "mixed", "control",
    ]
    assert choose_roles([[0, 0, 0], [0, 0, 1], [0, 0, 0]]) == [
        "control", "control", "mixed",
    ]
    with pytest.raises(ValueError, match="self edge"):
        choose_roles([[1, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_hybrid_replays_from_frozen_models():
    certificate = json.loads(OUTPUT.read_text(encoding="utf-8"))
    receipt = independent_verify()
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert primary_verify()["status"] == "posthoc-development-only"
    assert certificate["selected_model_arm_by_node"] == ["control", "mixed", "control"]
    assert certificate["metrics"]["max_executed_normalized_mse"] < .01
    assert certificate["metrics"]["max_paired_contrast_normalized_abs_error"] < .01
    assert certificate["independent_confirmation_worlds"] == 0
    assert certificate["full_scm_rollout_evaluated"] is False
    assert certificate["original_claim_closed"] is False


@pytest.mark.parametrize("field,replacement", [
    ("independent_confirmation_worlds", 1),
    ("rule_selected_after_development_metrics_seen", False),
    ("original_claim_closed", True),
    ("passes_local_max_normalized_mse_on_this_world", False),
])
def test_hybrid_checker_rejects_scope_inflation(tmp_path, field, replacement):
    certificate = json.loads(OUTPUT.read_text(encoding="utf-8"))
    certificate[field] = replacement
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(ValueError, match="scope changed"):
        independent_verify(certificate=path)
