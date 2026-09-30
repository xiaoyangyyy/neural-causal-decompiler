"""Checks for the frozen independent-world role experiment."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

from confirm_interventional_role_v1 import batch_layout, masks, preflight
import verify_interventional_role_v1 as independent
from verify_interventional_role_v1 import assert_same, verify

UNIT = ROOT / "runs/interventional_role_confirmation_v1/units/seed_8301_n3_test_id_0"


@pytest.mark.parametrize("nodes", [3, 5, 8])
def test_exact_budget_and_distinct_do_sources(nodes):
    layout = batch_layout(nodes)
    assert sum(fit + val for _, _, fit, val in layout) == 512
    assert len(layout) == 1 + 2 * nodes
    assert len({name for name, _, _, _ in layout}) == len(layout)
    assert all(fit > 0 and val >= 6 for _, _, fit, val in layout)


@pytest.mark.parametrize("nodes", [3, 5, 8])
def test_all_declared_masks_include_joint_interventions(nodes):
    family = masks(nodes)
    assert len(family) == 2 * nodes + 3
    assert len({tuple(sorted(mask.items())) for mask in family}) == len(family)
    assert sum(len(mask) == 2 for mask in family) == 2


def test_frozen_cohort_has_300_unique_new_worlds():
    plan = preflight()
    assert len(plan["seeds"]) * len(plan["nodes"]) * len(plan["environments"]) * plan["worlds_per_cell"] == 300


def test_independent_replay_rejects_metric_tamper():
    with pytest.raises(ValueError, match="metrics"):
        assert_same({"metrics": [{"normalized_mse": 0.0386}]},
                    {"metrics": [{"normalized_mse": 0.01}]})


@pytest.mark.skipif(not UNIT.exists(), reason="first confirmation unit not yet computed")
def test_first_new_world_fails_original_local_gate_under_independent_replay():
    receipt = verify(UNIT)
    assert receipt["status"] == "verified-one-independent-world"
    assert receipt["hybrid_meets_local_0_01"] is False
    assert receipt["hybrid_max_local_normalized_mse"] > 0.01
    assert receipt["original_claim_closed"] is False



@pytest.mark.skipif(not UNIT.exists(), reason="first confirmation unit not yet computed")
def test_full_verifier_rejects_falsified_gate(monkeypatch):
    original = independent.read

    def changed(path):
        value = original(path)
        if Path(path) == UNIT / "result.json":
            value["metrics"]["hybrid"]["max_local_normalized_mse"] = .01
        return value

    monkeypatch.setattr(independent, "read", changed)
    with pytest.raises(ValueError, match="metrics"):
        verify(UNIT)


@pytest.mark.skipif(not UNIT.exists(), reason="first confirmation unit not yet computed")
def test_full_verifier_rejects_claim_inflation(monkeypatch):
    original = independent.read

    def changed(path):
        value = original(path)
        if Path(path) == UNIT / "result.json":
            value["original_claim_closed"] = True
        return value

    monkeypatch.setattr(independent, "read", changed)
    with pytest.raises(ValueError, match="Untrusted unit"):
        verify(UNIT)
