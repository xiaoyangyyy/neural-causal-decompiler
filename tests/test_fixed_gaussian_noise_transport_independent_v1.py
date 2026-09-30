"""Separate replay of the fixed archived Gaussian noise transport claim."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from check_fixed_gaussian_noise_transport_v1 import (
    CERTIFICATE, PACKAGE, check, replay_receipt,
)


def test_independent_fixed_noise_receipt_replays():
    result = replay_receipt()
    assert result["status"] == "verified-fixed-candidate-noise-bound"
    assert result["same_world_across_splits"] is True
    assert result["intervention_distribution_bound_proved"] is False
    assert result["original_claim_closed"] is False


def test_independent_replay_rejects_result_metric_tamper(tmp_path):
    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    certificate["coordinate_wasserstein_l1_upper"][0] = "0"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises(ValueError, match="transport arithmetic mismatch"):
        check(changed)


def test_independent_replay_rejects_rehashed_candidate_payload(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(PACKAGE, copied)
    result_path = copied / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["candidate"]["node_models"][0]["scale"] *= 2
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="Original candidate file changed"):
        check(package=copied)
