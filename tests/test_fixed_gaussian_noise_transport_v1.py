"""Fixed-world ideal Gaussian noise transport: scope and identity checks."""
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from fixed_gaussian_noise_transport_v1 import (
    OUTPUT, PACKAGE, compute, verify,
)


def test_fixed_candidate_joint_noise_distance_replays_exactly():
    record = compute()
    assert record["status"] == "verified-fixed-candidate-noise-bound"
    assert Fraction(record["joint_wasserstein_l1_upper"]) == Fraction(
        35430223889551975, 288230376151711744)
    assert record["candidate_joint_product_law_declared"] is True
    assert record["true_law_oracle_metadata_used"] is True
    assert record["candidate_residual_joint_independence_proved"] is False
    assert record["mechanism_error_bound_proved"] is False
    assert record["intervention_distribution_bound_proved"] is False
    assert verify()["status"] == "verified-fixed-candidate-noise-bound"


@pytest.mark.parametrize("field,value", [
    ("joint_wasserstein_l1_upper", "0"),
    ("intervention_distribution_bound_proved", True),
    ("candidate_residual_joint_independence_proved", True),
    ("true_law_oracle_metadata_used", False),
    ("original_claim_closed", True),
])
def test_certificate_tampering_rejected(tmp_path, field, value):
    record = json.loads(OUTPUT.read_text(encoding="utf-8"))
    record[field] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify(changed)


def test_rehashed_candidate_payload_cannot_replace_frozen_identity(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(PACKAGE, copied)
    result_path = copied / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["candidate"]["node_models"][0]["loc"] += .1
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["result.json"] = sha256(result_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Wrong frozen noise candidate manifest"):
        compute(copied)


def test_preseal_attempt_with_missing_manifest_anchor_is_retained_and_rejected():
    old = ROOT / "validation/fixed_gaussian_noise_transport_v1_preseal_attempt0000.json"
    assert old.is_file()
    with pytest.raises(ValueError, match="certificate mismatch"):
        verify(old)
