"""Queued independent replay remains bound to the frozen handoff."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import advance_interventional_mechanism_dev_result_verify_v1 as follower


def test_independent_replay_precheck_binds_handoff_and_source():
    frozen = follower.precheck()
    assert len(frozen["checker_sha256"]) == 64
    assert len(frozen["evaluation_plan_sha256"]) == 64


def test_replay_precheck_rejects_changed_handoff_scope(tmp_path, monkeypatch):
    record = follower.read(follower.HANDOFF)
    record["original_objective_achieved"] = True
    copy = tmp_path / "handoff.json"
    copy.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(follower, "HANDOFF", copy)
    with pytest.raises(ValueError, match="Frozen development handoff changed"):
        follower.precheck()
