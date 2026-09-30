"""Tests for the archived candidate's exact intervention counterexample."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from fixed_scm_intervention_obstruction_v1 import (
    PACKAGE, OUTPUT, compute, verify as primary_verify,
)
from check_fixed_scm_intervention_obstruction_v1 import (
    OUTPUT as RECEIPT, verify as independent_verify,
)


def test_exact_witness_and_independent_replay():
    certificate = compute()
    assert certificate["delta_mean_on_do_x0_x2_equal_t"]["t2"] != "0"
    assert certificate["witness_joint_w1_l1_exceeds_1_over_100"] is True
    assert certificate["all_real_joint_w1_l1_supremum_infinite"] is True
    assert certificate["all_candidates_refuted"] is False
    assert certificate["original_claim_closed"] is False
    assert primary_verify()["status"] == "verified-fixed-candidate-counterexample"
    assert independent_verify()["witness_joint_w1_l1_lower"] == certificate["witness_joint_w1_l1_lower"]
    assert json.loads(RECEIPT.read_text(encoding="utf-8")) == independent_verify()


@pytest.mark.parametrize("field,replacement", [
    ("witness_joint_w1_l1_lower", "0"),
    ("all_candidates_refuted", True),
    ("original_claim_closed", True),
])
def test_independent_replay_rejects_claim_tamper(tmp_path, field, replacement):
    altered = json.loads(OUTPUT.read_text(encoding="utf-8"))
    altered[field] = replacement
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        independent_verify(certificate=path)


def test_rehashed_archive_mutation_rejected(tmp_path):
    copied = tmp_path / "archive"
    shutil.copytree(PACKAGE, copied)
    path = copied / "model/structured/explicit_scm.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["equations"][1]["args"][-1]["args"][0]["value"] = 0.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from hashlib import sha256
    manifest["files"]["model/structured/explicit_scm.json"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        independent_verify(package=copied)
