"""Frozen handoff validation is safe before either training stage starts."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import advance_interventional_mechanism_dev_v1 as handoff


def test_live_handoff_precheck_binds_both_training_arms():
    result = handoff.precheck()
    assert result["proof_pid"] > 0
    assert len(result["evaluation_plan_sha256"]) == 64
    plan = handoff.read(handoff.EVALUATION_PLAN)
    assert handoff.artifact_bytes(plan) >= 0


def test_handoff_rejects_relaxed_evaluation_threshold(tmp_path, monkeypatch):
    plan = handoff.read(handoff.EVALUATION_PLAN)
    plan["original_normalized_mse_threshold"] = .1
    copied = tmp_path / "weakened_plan.json"
    copied.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(handoff, "EVALUATION_PLAN", copied)
    with pytest.raises(ValueError, match="Evaluation threshold changed"):
        handoff.precheck()
