"""World-level uncertainty and strict completion checks."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
import analyze_interventional_role_confirmation_v1 as analysis


def test_four_rate_joint_99_percent_interval_uses_300_worlds():
    bound = analysis.simultaneous_hoeffding(0, 300)
    assert bound["worlds"] == 300
    assert bound["comparisons"] == 4
    assert bound["confidence_joint"] == .99
    assert 0.10 < bound["upper"] < 0.11


@pytest.mark.parametrize("successes,worlds", [(-1, 300), (301, 300), (0, 0)])
def test_invalid_world_level_counts_rejected(successes, worlds):
    with pytest.raises(ValueError):
        analysis.simultaneous_hoeffding(successes, worlds)


def test_partial_confirmation_cannot_be_analyzed_as_complete(tmp_path, monkeypatch):
    partial = tmp_path / "summary.json"
    partial.write_text(json.dumps({
        "status": "verified-partial", "verified_worlds": 12,
        "declared_worlds": 300, "worlds": [],
        "original_claim_closed": False,
    }), encoding="utf-8")
    monkeypatch.setattr(analysis, "SUMMARY", partial)
    with pytest.raises(ValueError, match="entire"):
        analysis.compute()
