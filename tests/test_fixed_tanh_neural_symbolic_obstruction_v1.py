"""Exact real-semantic Tanh network versus archived symbolic candidate."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from fixed_tanh_neural_symbolic_obstruction_v1 import (
    OUTPUT, PACKAGE, compute, verify as primary_verify,
)
from check_fixed_tanh_neural_symbolic_obstruction_v1 import (
    OUTPUT as RECEIPT, verify as independent_verify,
)


def test_frozen_network_witness_replays():
    certificate = compute()
    assert certificate["symbolic_ray_coefficients"]["2"] != "0"
    assert certificate["all_real_network_symbolic_supremum_infinite"] is True
    assert certificate["witness_exceeds_1_over_100_training_scale"] is True
    assert certificate["oracle_true_mechanism_used"] is False
    assert certificate["original_claim_closed"] is False
    assert primary_verify()["status"] == "verified-fixed-candidate-neural-counterexample"
    assert independent_verify()["witness_normalized_error_lower"] == certificate["witness_normalized_error_lower"]
    assert json.loads(RECEIPT.read_text(encoding="utf-8")) == independent_verify()


@pytest.mark.parametrize("field,replacement", [
    ("witness_absolute_error_lower", "0"),
    ("oracle_true_mechanism_used", True),
    ("intervention_family_includes_witness_proved", True),
    ("original_claim_closed", True),
])
def test_independent_verifier_rejects_certificate_tamper(tmp_path, field, replacement):
    altered = json.loads(OUTPUT.read_text(encoding="utf-8"))
    altered[field] = replacement
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        independent_verify(certificate=path)


def test_independent_verifier_rejects_rehashed_model_mutation(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(PACKAGE, copied)
    model_path = copied / "model/baseline/mechanism_1.pt"
    with model_path.open("ab") as handle:
        handle.write(b"tampered")
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from hashlib import sha256
    manifest["files"]["model/baseline/mechanism_1.pt"] = sha256(model_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        independent_verify(package=copied)
